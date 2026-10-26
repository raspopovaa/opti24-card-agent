"""Демо-режим: синтетический договор работает без сети и без ключа Opti24."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from agent.config import Mode
from agent.dashboard import collect
from agent.demo import DEMO_CONTRACT_NUMBER, DemoWriteError, SyntheticOpti24, demo_client
from agent.safety import ApprovalGate, BillingBudget
from agent.tools import build_tools

TODAY = date(2026, 10, 25)


def tools_for(tmp_path: Path, answer: bool = True) -> tuple[dict[str, Any], SyntheticOpti24]:
    async def ask(_: str) -> bool:
        return answer

    fake = SyntheticOpti24(today=TODAY)
    gate = ApprovalGate(mode=Mode.CONFIRM, ask=ask, audit_log=tmp_path / "audit.jsonl")
    tools = {t.name: t for t in build_tools(fake, BillingBudget(limit=50), gate)}  # type: ignore[arg-type]
    return tools, fake


def call(tools: dict[str, Any], name: str, args: dict[str, object]) -> dict[str, Any]:
    return json.loads(asyncio.run(tools[name].call(args)))


def test_demo_dashboard_has_contract_and_no_notes() -> None:
    board = asyncio.run(collect(SyntheticOpti24(today=TODAY), BillingBudget(limit=10), TODAY))  # type: ignore[arg-type]
    assert board.notes == []
    assert board.contract["Договор"] == DEMO_CONTRACT_NUMBER
    assert board.month["Топливо"]  # есть продукты за месяц
    assert board.cards["Карт всего"] == "10"


def test_demo_is_deterministic() -> None:
    first = SyntheticOpti24(today=TODAY).demo_store.read(date(2026, 9, 1), TODAY)
    second = SyntheticOpti24(today=TODAY).demo_store.read(date(2026, 9, 1), TODAY)
    assert first and [t["id"] for t in first] == [t["id"] for t in second]
    assert sum(float(t["sum"]) for t in first) == sum(float(t["sum"]) for t in second)


def test_demo_reads_tools_without_real_api(tmp_path: Path) -> None:
    tools, _ = tools_for(tmp_path)
    cards = call(tools, "list_cards", {})
    assert cards["total"] == 10
    profile = call(tools, "get_card_profile", {"card_id": "c-01"})
    assert profile["drivers"][0]["phone"] == "…-01"  # телефон маскируется и здесь
    spending = call(tools, "spending_report", {"date_from": "2026-09-01", "date_to": "2026-09-30"})
    assert "error" not in spending and spending["operations"] > 0
    suspicious = call(tools, "find_suspicious_fills", {"date_from": "2026-09-01", "date_to": "2026-09-30"})
    assert suspicious.get("suspicious_total", 0) > 0


def test_demo_change_is_confirmed_but_not_sent(tmp_path: Path) -> None:
    tools, fake = tools_for(tmp_path)
    out = call(tools, "block_cards", {"card_ids": ["c-07"], "reason": "проверка демо"})
    assert out["status"] == "demo_not_sent"
    assert "не отправлено" in out["message"] and "7826010000100007" in out["message"]
    assert fake.cards_state["c-07"]["status"] == "Active"
    comment = call(tools, "set_card_comment", {"card_id": "c-07", "comment": "Новый текст"})
    assert comment["status"] == "demo_not_sent"
    assert fake.cards_state["c-07"]["comment"].startswith("Газель")
    events = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [e["status"] for e in events if e["event"] == "outcome"] == ["demo_not_sent", "demo_not_sent"]


def test_demo_refused_change_does_not_touch_state(tmp_path: Path) -> None:
    tools, fake = tools_for(tmp_path, answer=False)
    out = call(tools, "block_cards", {"card_ids": ["c-07"], "reason": "x"})
    assert out["status"] == "declined"
    assert fake.cards_state["c-07"]["status"] == "Active"


def test_demo_client_writes_are_disabled() -> None:
    fake = SyntheticOpti24(today=TODAY)
    with pytest.raises(DemoWriteError):
        asyncio.run(fake.cards.block_card(card_ids=["c-07"], block=True))


def test_demo_reads_only_its_own_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work_db, demo_db = tmp_path / "agent.db", tmp_path / "demo.db"
    monkeypatch.setenv("AGENT_DB_PATH", str(work_db))
    monkeypatch.setenv("AGENT_DEMO_DB_PATH", str(demo_db))
    demo_db.write_text("старый файл, не база", encoding="utf-8")  # пересоздаётся при запуске

    async def ask(_: str) -> bool:
        return True

    async def run() -> tuple[dict[str, Any], dict[str, Any]]:
        async with demo_client() as client:
            gate = ApprovalGate(mode=Mode.CONFIRM, ask=ask, audit_log=tmp_path / "audit.jsonl")
            tools = {t.name: t for t in build_tools(client, BillingBudget(limit=50), gate)}
            today = date.today()
            period = {"date_from": today.replace(day=1).isoformat(), "date_to": today.isoformat()}
            spending = json.loads(await tools["spending_report"].call(period))
            first = (await client.transactions.get_transactions_v2(**period)).data.result[0]
            detail = json.loads(await tools["get_transaction_detail"].call({"transaction_id": str(first.id)}))
            return spending, detail

    spending, detail = asyncio.run(run())
    assert spending["data_source"] == "локальная база" and spending["operations"] > 0
    assert "error" not in detail
    assert not work_db.exists()  # рабочая база агента в демо не открывается
    with sqlite3.connect(demo_db) as db:
        assert db.execute("SELECT COUNT(*) FROM operations").fetchone()[0] > 0


def test_demo_client_context_is_usable_without_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_DEMO_DB_PATH", str(tmp_path / "demo.db"))

    async def run() -> str:
        async with demo_client() as client:
            info = (await client.auth.get_info()).data
            return f"{info.client_info.Queries}"

    assert asyncio.run(run()) == "500"
