"""Учёт расхода агентов: токены модели, вызовы API и время на каждый вопрос.

База отдельная от ``data/agent.db``: там кеш операций клиента, который можно стереть, а здесь
журнал только на добавление, без данных клиента, общий для нескольких агентов (``agent_id``
в каждой строке). Тексты вопросов и ответов не хранятся: только тема, размеры и итоги.
Имена полей токенов взяты из соглашений OpenTelemetry для ИИ (gen_ai.usage.*).

    ask() ─► begin_turn ─► llm_call / tool_call / embedding … ─► end_turn ─► одна транзакция

    AGENT_USAGE_DB   путь к базе, по умолчанию data/usage.db; ``off`` отключает учёт

Отчёт: ``python -m agent.usage`` (по темам за текущий месяц), цены моделей:
``python -m agent.usage price ...``. Сбой записи не мешает ответу клиенту.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sqlite3
import sys
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path("data/usage.db")
SCHEMA_VERSION = 1
AGENT_ID = "opti24-card-agent"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL, applied_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    agent_version TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    interface TEXT NOT NULL,
    mode TEXT NOT NULL,
    provider TEXT NOT NULL,
    contract_key TEXT,
    operator TEXT,
    started_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS turns (
    turn_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions (session_id),
    agent_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    topic TEXT,
    section TEXT,
    confident INTEGER NOT NULL,
    question_chars INTEGER NOT NULL,
    answer_chars INTEGER NOT NULL,
    steps INTEGER NOT NULL,
    tool_count INTEGER NOT NULL,
    billable_calls INTEGER NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cache_read_tokens INTEGER NOT NULL,
    duration_ms INTEGER NOT NULL,
    unverified_numbers INTEGER NOT NULL,
    outcome TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_turns_started ON turns (started_at);
CREATE TABLE IF NOT EXISTS llm_calls (
    turn_id TEXT NOT NULL REFERENCES turns (turn_id),
    step INTEGER NOT NULL,
    purpose TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cache_read_tokens INTEGER NOT NULL,
    cache_write_tokens INTEGER NOT NULL,
    reasoning_tokens INTEGER NOT NULL,
    tools_visible INTEGER,
    duration_ms INTEGER,
    finish_reason TEXT
);
CREATE INDEX IF NOT EXISTS ix_llm_calls_turn ON llm_calls (turn_id);
CREATE TABLE IF NOT EXISTS tool_calls (
    turn_id TEXT NOT NULL REFERENCES turns (turn_id),
    seq INTEGER NOT NULL,
    tool TEXT NOT NULL,
    billable_calls INTEGER NOT NULL,
    result_chars INTEGER NOT NULL,
    error_reason TEXT,
    duration_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_tool_calls_turn ON tool_calls (turn_id);
CREATE TABLE IF NOT EXISTS prices (
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    valid_from TEXT NOT NULL,
    rub_per_1m_input TEXT NOT NULL,
    rub_per_1m_output TEXT NOT NULL,
    rub_per_1m_cache_read TEXT,
    source TEXT,
    PRIMARY KEY (provider, model, valid_from)
);
"""

# Учёт текущего вопроса для мест, куда ссылку не передать явно (эмбеддинги в поиске по знаниям).
_current: ContextVar[UsageLedger | None] = ContextVar("usage_ledger", default=None)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def prompt_version(text: str) -> str:
    """Короткий отпечаток инструкции: сравнивать расход до и после её правки."""
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def usage_path_from_env() -> Path | None:
    setting = os.environ.get("AGENT_USAGE_DB", str(DEFAULT_PATH)).strip()
    if setting.lower() in {"", "off", "0", "false"}:
        return None
    return Path(setting).expanduser()


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=5)
    db.execute("PRAGMA journal_mode=WAL")  # несколько агентов пишут в одну базу
    db.executescript(_SCHEMA)
    if db.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 0:
        db.execute("INSERT INTO schema_version VALUES (?, ?)", (SCHEMA_VERSION, _now()))
    db.commit()
    return db


@dataclass(slots=True)
class _LlmCall:
    step: int
    purpose: str
    model: str
    input_tokens: int  # весь вход, вместе с токенами из кеша
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    reasoning_tokens: int
    tools_visible: int | None
    duration_ms: int | None
    finish_reason: str | None


@dataclass(slots=True)
class _ToolCall:
    seq: int
    tool: str
    billable_calls: int
    result_chars: int
    error_reason: str | None
    duration_ms: int


@dataclass(slots=True)
class _Turn:
    turn_id: str
    started_at: str
    started: float
    topic: str | None
    section: str | None
    confident: bool
    question_chars: int
    llm: list[_LlmCall] = field(default_factory=list)
    tools: list[_ToolCall] = field(default_factory=list)


@dataclass(slots=True)
class Price:
    input: Decimal
    output: Decimal
    cache_read: Decimal | None


def call_cost(call: _LlmCall, price: Price | None) -> Decimal | None:
    """Стоимость одного обращения в рублях; без цены — None, а не выдуманная сумма."""
    if price is None:
        return None
    cached = call.cache_read_tokens
    cache_price = price.cache_read if price.cache_read is not None else price.input
    fresh = max(call.input_tokens - cached, 0)
    total = fresh * price.input + cached * cache_price + call.output_tokens * price.output
    return (total / Decimal(1_000_000)).quantize(Decimal("0.0001"))


class UsageLedger:
    """Собирает расход вопроса в памяти и пишет его в базу одной транзакцией в конце вопроса."""

    def __init__(
        self,
        path: Path | None,
        *,
        provider: str,
        mode: str,
        interface: str,
        prompt: str,
        contract_key: str | None = None,
        operator: str | None = None,
        agent_id: str = AGENT_ID,
    ) -> None:
        self.path = path
        self.session_id = uuid.uuid4().hex
        self.agent_id = agent_id
        self.provider = provider
        self.contract_key = contract_key
        self.operator = operator
        self._agent_version = os.environ.get("AGENT_VERSION", "")
        self._prompt_version = prompt_version(prompt)
        self._interface = interface
        self._mode = mode
        self._started_at = _now()
        self._session_saved = False
        self._turn: _Turn | None = None
        self._db: sqlite3.Connection | None = None
        self.last_error: str | None = None

    # --- запись событий вопроса ---

    def begin_turn(self, *, topic: str | None, section: str | None, confident: bool, question: str) -> None:
        self._turn = _Turn(
            turn_id=uuid.uuid4().hex,
            started_at=_now(),
            started=time.monotonic(),
            topic=topic,
            section=section,
            confident=confident,
            question_chars=len(question),
        )
        _current.set(self)

    def llm_call(
        self,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        reasoning_tokens: int = 0,
        tools_visible: int | None = None,
        duration_ms: int | None = None,
        finish_reason: str | None = None,
        purpose: str = "answer",
    ) -> None:
        turn = self._turn
        if turn is None:
            return
        step = sum(1 for c in turn.llm if c.purpose == purpose) + 1
        turn.llm.append(
            _LlmCall(
                step, purpose, model, input_tokens, output_tokens, cache_read_tokens,
                cache_write_tokens, reasoning_tokens, tools_visible, duration_ms, finish_reason,
            )
        )

    def tool_call(
        self, *, tool: str, billable_calls: int, result_chars: int, error_reason: str | None, duration_ms: int
    ) -> None:
        turn = self._turn
        if turn is None:
            return
        turn.tools.append(_ToolCall(len(turn.tools) + 1, tool, billable_calls, result_chars, error_reason, duration_ms))

    def tool_ms(self) -> int:
        """Сколько миллисекунд вопроса ушло на инструменты (чтобы вычесть их из времени шага модели)."""
        return sum(t.duration_ms for t in self._turn.tools) if self._turn else 0

    def end_turn(self, *, answer: str, outcome: str, unverified_numbers: int = 0) -> None:
        turn, self._turn = self._turn, None
        _current.set(None)
        if turn is None or self.path is None:
            return
        try:
            self._save(turn, answer, outcome, unverified_numbers)
        except (sqlite3.Error, OSError) as exc:
            # Учёт вспомогательный: ответ клиенту важнее, поэтому ошибка только в stderr, один раз.
            if self.last_error is None:
                print(f"Учёт расхода не записан ({self.path}): {exc}", file=sys.stderr)
            self.last_error = str(exc)

    # --- база ---

    def _connection(self) -> sqlite3.Connection:
        if self._db is None:
            assert self.path is not None  # nosec B101 — вызывается только при заданном пути
            self._db = connect(self.path)
        return self._db

    def _save(self, turn: _Turn, answer: str, outcome: str, unverified_numbers: int) -> None:
        db = self._connection()
        answer_calls = [c for c in turn.llm if c.purpose == "answer"]
        with db:
            if not self._session_saved:
                db.execute(
                    "INSERT OR IGNORE INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        self.session_id, self.agent_id, self._agent_version, self._prompt_version, self._interface,
                        self._mode, self.provider, self.contract_key, self.operator, self._started_at,
                    ),
                )
                self._session_saved = True
            db.execute(
                "INSERT INTO turns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    turn.turn_id, self.session_id, self.agent_id, turn.started_at, turn.topic, turn.section,
                    int(turn.confident), turn.question_chars, len(answer), len(answer_calls), len(turn.tools),
                    sum(t.billable_calls for t in turn.tools),
                    sum(c.input_tokens for c in turn.llm), sum(c.output_tokens for c in turn.llm),
                    sum(c.cache_read_tokens for c in turn.llm),
                    int((time.monotonic() - turn.started) * 1000), unverified_numbers, outcome,
                ),
            )
            db.executemany(
                "INSERT INTO llm_calls VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        turn.turn_id, c.step, c.purpose, self.provider, c.model, c.input_tokens, c.output_tokens,
                        c.cache_read_tokens, c.cache_write_tokens, c.reasoning_tokens, c.tools_visible,
                        c.duration_ms, c.finish_reason,
                    )
                    for c in turn.llm
                ],
            )
            db.executemany(
                "INSERT INTO tool_calls VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (turn.turn_id, t.seq, t.tool, t.billable_calls, t.result_chars, t.error_reason, t.duration_ms)
                    for t in turn.tools
                ],
            )

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None


def record_embedding(model: str, input_tokens: int, duration_ms: int) -> None:
    """Эмбеддинги поиска по знаниям: пишутся в текущий вопрос, если он есть."""
    ledger = _current.get()
    if ledger is not None:
        ledger.llm_call(
            model=model, input_tokens=input_tokens, output_tokens=0, duration_ms=duration_ms, purpose="embedding"
        )


def _price(db: sqlite3.Connection, provider: str, model: str, day: str) -> Price | None:
    row = db.execute(
        "SELECT rub_per_1m_input, rub_per_1m_output, rub_per_1m_cache_read FROM prices "
        "WHERE provider = ? AND model = ? AND valid_from <= ? ORDER BY valid_from DESC LIMIT 1",
        (provider, model, day),
    ).fetchone()
    if row is None:
        return None
    return Price(Decimal(row[0]), Decimal(row[1]), None if row[2] is None else Decimal(row[2]))


# --- отчёт ---

GROUPS = {
    "topic": "COALESCE(t.topic, '(без темы)')",
    "model": "l.model",
    "agent": "t.agent_id",
    "interface": "s.interface",
    "prompt": "s.prompt_version",
}


def report(db: sqlite3.Connection, month: str, by: str = "topic") -> list[dict[str, Any]]:
    """Сводка за месяц YYYY-MM по группе: вопросы, шаги, токены, вызовы API, время и рубли.

    Рубли считаются при чтении по таблице prices, действовавшей в день вопроса: цену можно
    внести и после замеров. Если хотя бы для одного обращения группы цены нет, сумма пустая.
    """
    key = GROUPS[by]
    sql = (
        f"SELECT {key}, t.turn_id, t.steps, t.billable_calls, t.duration_ms, t.unverified_numbers, "  # nosec B608
        "substr(t.started_at, 1, 10), l.provider, l.model, l.input_tokens, l.output_tokens, "
        "l.cache_read_tokens FROM turns t JOIN sessions s USING (session_id) "
        "LEFT JOIN llm_calls l USING (turn_id) WHERE substr(t.started_at, 1, 7) = ?"
    )
    groups: dict[str, dict[str, Any]] = {}
    seen: set[tuple[str, str]] = set()
    prices: dict[tuple[str, str, str], Price | None] = {}
    for row in db.execute(sql, (month,)):
        group, turn_id, steps, billable, duration, unverified, day, provider, model, tin, tout, tcache = row
        g = groups.setdefault(
            str(group),
            {"group": str(group), "questions": 0, "steps": 0, "input_tokens": 0, "output_tokens": 0,
             "cache_read_tokens": 0, "billable_calls": 0, "seconds": 0.0, "unverified": 0,
             "cost_rub": Decimal(0)},
        )
        # При группировке не по модели строки вопроса повторяются по числу обращений к модели.
        if (str(group), turn_id) not in seen:
            seen.add((str(group), turn_id))
            g["questions"] += 1
            g["steps"] += steps
            g["billable_calls"] += billable
            g["seconds"] += duration / 1000
            g["unverified"] += int(unverified > 0)
        if model is None:
            continue
        g["input_tokens"] += tin
        g["output_tokens"] += tout
        g["cache_read_tokens"] += tcache
        price_key = (provider, model, day)
        if price_key not in prices:
            prices[price_key] = _price(db, provider, model, day)
        call = _LlmCall(0, "", model, tin, tout, tcache, 0, 0, None, None, None)
        cost = call_cost(call, prices[price_key])
        g["cost_rub"] = None if cost is None or g["cost_rub"] is None else g["cost_rub"] + cost
    rows = []
    for g in sorted(groups.values(), key=lambda v: v["input_tokens"], reverse=True):
        n = g.pop("questions")
        steps, seconds, cost = g.pop("steps"), g.pop("seconds"), g.pop("cost_rub")
        rows.append(
            {"group": g["group"], "questions": n, "avg_steps": round(steps / n, 1),
             "input_tokens": g["input_tokens"], "output_tokens": g["output_tokens"],
             "cache_read_tokens": g["cache_read_tokens"], "billable_calls": g["billable_calls"],
             "avg_seconds": round(seconds / n, 1), "unverified": g["unverified"],
             "cost_rub": None if cost is None else str(cost.quantize(Decimal("0.01")))}
        )
    return rows


def _print_report(rows: list[dict[str, Any]], month: str, by: str) -> None:
    if not rows:
        print(f"За {month} записей нет.")
        return
    columns = list(rows[0])
    print(f"Расход агентов за {month}, по {by}:")
    print("\t".join(columns))
    for row in rows:
        print("\t".join("—" if row[c] is None else str(row[c]) for c in columns))
    if any(row["cost_rub"] is None for row in rows):
        print("— в cost_rub: для части моделей нет цены; задайте её командой price.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent.usage", description="Расход токенов и запросов агентов")
    parser.add_argument("--db", type=Path, help="база учёта (по умолчанию AGENT_USAGE_DB или data/usage.db)")
    sub = parser.add_subparsers(dest="command")
    rep = sub.add_parser("report", help="сводка за месяц (команда по умолчанию)")
    rep.add_argument("--month", default=date.today().strftime("%Y-%m"), help="месяц YYYY-MM")
    rep.add_argument("--by", choices=sorted(GROUPS), default="topic")
    price = sub.add_parser("price", help="цена модели в рублях за 1 млн токенов с даты")
    price.add_argument("--provider", required=True)
    price.add_argument("--model", required=True)
    price.add_argument("--from", dest="valid_from", default=date.today().isoformat(), help="дата YYYY-MM-DD")
    price.add_argument("--input", required=True, type=Decimal)
    price.add_argument("--output", required=True, type=Decimal)
    price.add_argument("--cache-read", type=Decimal)
    price.add_argument("--source", help="ссылка на прайс")
    args = parser.parse_args(argv)

    path = args.db or usage_path_from_env()
    if path is None:
        print("Учёт выключен: AGENT_USAGE_DB=off.", file=sys.stderr)
        return 1
    db = connect(path)
    try:
        if args.command == "price":
            date.fromisoformat(args.valid_from)
            with db:
                db.execute(
                    "INSERT OR REPLACE INTO prices VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (args.provider, args.model, args.valid_from, str(args.input), str(args.output),
                     None if args.cache_read is None else str(args.cache_read), args.source),
                )
            print(f"Цена {args.provider}/{args.model} с {args.valid_from} записана.")
            return 0
        month = getattr(args, "month", date.today().strftime("%Y-%m"))
        by = getattr(args, "by", "topic")
        _print_report(report(db, month, by), month, by)
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
