"""Значения по умолчанию: подставляются, только если пользователь не назвал своё, и видны в ответе инструмента."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from agent.config import Mode
from agent.safety import ApprovalGate, BillingBudget
from agent.tools import build_tools
from agent.tools.common import mask_email
from tests.test_agent import FakeOpti24, make_tools
from tests.test_price_analytics import _with_stations, fill


def call(tools: dict[str, object], name: str, args: dict[str, object]) -> dict[str, object]:
    return json.loads(asyncio.run(tools[name].call(args)))  # type: ignore[attr-defined]


def test_report_goes_to_the_profile_email_shown_masked_to_the_model(tmp_path: Path) -> None:
    fake = FakeOpti24([])
    asked: list[str] = []

    async def ask(summary: str) -> bool:
        asked.append(summary)
        return True

    gate = ApprovalGate(mode=Mode.CONFIRM, ask=ask, audit_log=tmp_path / "audit.jsonl")
    tools = {t.name: t for t in build_tools(fake, BillingBudget(3), gate, reports_dir=tmp_path)}  # type: ignore[arg-type]
    out = call(tools, "order_fuel_report", {"date_from": "2026-09-01", "date_to": "2026-09-30"})
    assert fake.ordered_reports[0]["email"] == "user@example.ru"
    assert "user@example.ru" in asked[0]  # человек видит полный адрес в подтверждении
    assert out["email"] == "us***@example.ru" and out["email_source"].startswith("по умолчанию")
    assert "user@example.ru" not in (tmp_path / "audit.jsonl").read_text(encoding="utf-8")

    named = call(tools, "order_fuel_report", {"date_from": "2026-09-01", "date_to": "2026-09-30", "email": "a@b.ru"})
    assert fake.ordered_reports[1]["email"] == "a@b.ru" and named["email_source"] == "указан пользователем"


def test_mask_email() -> None:
    assert mask_email("ivan.petrov@example.ru") == "iv***@example.ru"
    assert mask_email(None) is None and mask_email("без собаки") is None


def test_default_product_filters_price_analytics_only_when_not_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPTI24_DEFAULT_PRODUCT", "ДТ")
    monkeypatch.setenv("OPTI24_SERVICE_FEE_PERCENT", "1")
    txs = [fill(1, "azs-msk", "ДТ", 100, "6000", "6100"), fill(2, "azs-msk", "АИ-95", 10, "650", "650")]
    fake, _, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path, txs=txs)
    _with_stations(fake, [])
    period = {"date_from": "2026-09-01", "date_to": "2026-09-30"}
    default = call(tools, "fuel_price_analytics", period)
    assert [p["product"] for p in default["by_product"]] == ["ДТ"]
    assert default["defaults"] == {"product": "ДТ (OPTI24_DEFAULT_PRODUCT)"}
    named = call(tools, "fuel_price_analytics", {**period, "product": "АИ-95"})
    assert [p["product"] for p in named["by_product"]] == ["АИ-95"] and "defaults" not in named


def test_default_product_fills_the_receipt_for_check_purchase(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPTI24_DEFAULT_PRODUCT", "ДТ")
    _, _, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path, budget_limit=5)
    out = call(tools, "diagnose_fueling", {"card_id": "c1"})
    assert out["receipt"]["product_assumed"].startswith("ДТ (OPTI24_DEFAULT_PRODUCT)")


def test_default_group_narrows_lists_but_not_searches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPTI24_DEFAULT_GROUP_ID", "g7")
    fake, _, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path, budget_limit=5)
    seen: list[dict[str, object]] = []
    cards, limits = fake.cards.get_cards_v2, fake.limits.get_limits

    async def cards_spy(**kwargs: object) -> NS:
        seen.append({"cards": kwargs.get("group_id")})
        return await cards(**kwargs)

    async def limits_spy(**kwargs: object) -> NS:
        seen.append({"limits": kwargs.get("group_id")})
        return await limits(**kwargs)

    fake.cards.get_cards_v2, fake.limits.get_limits = cards_spy, limits_spy
    listed = call(tools, "list_cards", {})
    call(tools, "list_cards", {"query": "Иванов"})  # поиск по имени — по всем картам
    rules = call(tools, "get_limits", {})
    call(tools, "get_limits", {"card_id": "c1"})
    assert seen == [{"cards": "g7"}, {"cards": None}, {"limits": "g7"}, {"limits": None}]
    assert listed["defaults"] == {"group_id": "g7 (OPTI24_DEFAULT_GROUP_ID)"} and "defaults" in rules


def test_without_settings_nothing_is_substituted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OPTI24_DEFAULT_PRODUCT", "OPTI24_DEFAULT_GROUP_ID"):
        monkeypatch.delenv(name, raising=False)
    _, _, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path, budget_limit=5)
    assert "defaults" not in call(tools, "list_cards", {})
    assert "product_assumed" not in call(tools, "diagnose_fueling", {"card_id": "c1"})["receipt"]
