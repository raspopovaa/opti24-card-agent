"""Хранилище операций: ответы API как есть плюс отметки, какой период загружен.

Строка хранит JSON операции из ответа API (TransactionItemV2), поэтому инструменты
получают из базы тот же объект SDK, что и из API, с теми же полями. Номер договора
и ID в базу не пишутся: договор — хеш его ID.

    инструменты ──► ToolContext.transactions ──► OperationStore ──► SQLite
                                  │ нет в базе или копия устарела
                                  └──► API (тарифицируемо) ──► OperationStore.save
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

# Сторно и корректировки приходят в течение нескольких дней после операции.
SETTLE_DAYS = 3
DEFAULT_TTL = timedelta(hours=1)
DEFAULT_PATH = Path("data/agent.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS operations (
    contract_key TEXT NOT NULL,
    id TEXT NOT NULL,
    local_day TEXT NOT NULL,
    card_id TEXT NOT NULL,
    raw TEXT NOT NULL,
    loaded_at TEXT NOT NULL,
    PRIMARY KEY (contract_key, id)
);
CREATE INDEX IF NOT EXISTS ix_operations_day ON operations (contract_key, local_day);
CREATE TABLE IF NOT EXISTS loaded_periods (
    contract_key TEXT NOT NULL,
    date_from TEXT NOT NULL,
    date_to TEXT NOT NULL,
    card_id TEXT NOT NULL,
    loaded_at TEXT NOT NULL,
    PRIMARY KEY (contract_key, date_from, date_to, card_id)
);
"""
_ALL_CARDS = ""  # card_id в loaded_periods: пустая строка — загружены все карты


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Операции из базы и момент, когда период был загружен из API."""

    items: list[dict[str, Any]]
    loaded_at: datetime


class OperationStore:
    def __init__(self, path: Path, contract_id: str, ttl: timedelta = DEFAULT_TTL) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.executescript(_SCHEMA)
        self._key = contract_key_for(contract_id)
        self._ttl = ttl

    def close(self) -> None:
        self._db.close()

    def lookup(
        self, date_from: date, date_to: date, card_id: str | None, now: datetime
    ) -> Snapshot | None:
        """Операции периода, если он загружен целиком и копия ещё годится; иначе None."""
        loaded_at = self._loaded_at(date_from, date_to, card_id)
        if loaded_at is None or not _usable(loaded_at, date_to, now, self._ttl):
            return None
        return Snapshot(self.read(date_from, date_to, card_id), loaded_at)

    def read(self, date_from: date, date_to: date, card_id: str | None = None) -> list[dict[str, Any]]:
        """Операции из базы за дни периода, без проверки, загружен ли он целиком."""
        query = "SELECT raw FROM operations WHERE contract_key = ? AND local_day BETWEEN ? AND ?"
        params: list[str] = [self._key, date_from.isoformat(), date_to.isoformat()]
        if card_id:
            query += " AND card_id = ?"
            params.append(card_id)
        rows = self._db.execute(query + " ORDER BY local_day, id", params).fetchall()
        return [json.loads(raw) for (raw,) in rows]

    def by_id(self, operation_id: str) -> dict[str, Any] | None:
        """Операция по ID или None."""
        row = self._db.execute(
            "SELECT raw FROM operations WHERE contract_key = ? AND id = ?", (self._key, operation_id)
        ).fetchone()
        return None if row is None else json.loads(row[0])

    def save(
        self,
        date_from: date,
        date_to: date,
        card_id: str | None,
        items: Iterable[dict[str, Any]],
        now: datetime,
    ) -> None:
        """Записать операции периода и отметить период загруженным.

        Повторная загрузка того же периода обновляет строки по ID операции, а не дублирует.
        """
        stamp = now.isoformat(timespec="seconds")
        rows = [
            (self._key, str(item["id"]), str(item["timestamp"])[:10], str(item["card_id"]),
             json.dumps(item, ensure_ascii=False), stamp)
            for item in items
        ]
        with self._db:
            self._db.executemany(
                "INSERT INTO operations VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (contract_key, id) DO UPDATE SET "
                "local_day = excluded.local_day, card_id = excluded.card_id, "
                "raw = excluded.raw, loaded_at = excluded.loaded_at",
                rows,
            )
            self._db.execute(
                "INSERT OR REPLACE INTO loaded_periods VALUES (?, ?, ?, ?, ?)",
                (self._key, date_from.isoformat(), date_to.isoformat(), card_id or _ALL_CARDS, stamp),
            )

    def _loaded_at(self, date_from: date, date_to: date, card_id: str | None) -> datetime | None:
        # Подходит период, который накрывает запрошенный: по той же карте или по всем картам.
        cards = [_ALL_CARDS] if not card_id else [_ALL_CARDS, card_id]
        marks = ",".join("?" * len(cards))
        row = self._db.execute(
            f"SELECT MAX(loaded_at) FROM loaded_periods WHERE contract_key = ? "
            f"AND date_from <= ? AND date_to >= ? AND card_id IN ({marks})",
            [self._key, date_from.isoformat(), date_to.isoformat(), *cards],
        ).fetchone()
        return None if row[0] is None else datetime.fromisoformat(row[0])


def contract_key_for(contract_id: str) -> str:
    """Тот же непрозрачный ключ, что у хранилища дашборда (fuel_portal.run_sync)."""
    return "c-" + hashlib.sha256(contract_id.encode()).hexdigest()[:16]


def open_store(contract_id: str | None) -> OperationStore | None:
    """База из AGENT_DB_PATH (по умолчанию data/agent.db); ``off`` или нет договора — без базы."""
    setting = os.environ.get("AGENT_DB_PATH", str(DEFAULT_PATH)).strip()
    if not contract_id or setting.lower() in {"", "off", "0", "false"}:
        return None
    minutes = os.environ.get("AGENT_DB_TTL_MINUTES", "").strip()
    ttl = timedelta(minutes=int(minutes)) if minutes else DEFAULT_TTL
    return OperationStore(Path(setting).expanduser(), contract_id, ttl)


def _usable(loaded_at: datetime, date_to: date, now: datetime, ttl: timedelta) -> bool:
    # Период закрыт, если загружен позже, чем могли прийти его сторно: такая копия не устаревает.
    if loaded_at.date() > date_to + timedelta(days=SETTLE_DAYS):
        return True
    return now - loaded_at < ttl
