"""Демо-режим: синтетический договор работает без сети и без ключа Opti24."""

from __future__ import annotations

import asyncio
import json
from datetime import date
from pathlib import Path
from typing import Any

from agent.config import Mode
from agent.dashboard import collect
from agent.demo import DEMO_CONTRACT_NUMBER, SyntheticOpti24, demo_client
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
    first = SyntheticOpti24(today=TODAY)
    second = SyntheticOpti24(today=TODAY)
    assert [t.id for t in first.operations] == [t.id for t in second.operations]
    assert sum(float(t.sum) for t in first.operations) == sum(float(t.sum) for t in second.operations)


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


def test_demo_block_is_confirmed_and_rechecked(tmp_path: Path) -> None:
    tools, fake = tools_for(tmp_path)
    out = call(tools, "block_cards", {"card_ids": ["c-07"], "reason": "проверка демо"})
    assert out["status"] == "blocked" and out["verified"]["status"] == "match"
    assert fake.cards_state["c-07"]["status"] == "Locked(Client)"


def test_demo_refused_change_does_not_touch_state(tmp_path: Path) -> None:
    tools, fake = tools_for(tmp_path, answer=False)
    call(tools, "block_cards", {"card_ids": ["c-07"], "reason": "x"})
    assert fake.cards_state["c-07"]["status"] == "Active"


def test_demo_client_context_is_usable_without_network() -> None:
    async def run() -> str:
        async with demo_client() as client:
            info = (await client.auth.get_info()).data
            return f"{info.client_info.Queries}"

    assert asyncio.run(run()) == "500"
