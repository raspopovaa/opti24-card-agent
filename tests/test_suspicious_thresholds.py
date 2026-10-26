"""Пороги проверки подозрительных заправок: 650 л и 4 заправки за день, настройка и параметры инструмента."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import pytest

from agent.analysis import (
    DEFAULT_MAX_FILLS_PER_DAY,
    DEFAULT_MAX_LITERS,
    AnomalyRules,
    anomaly_rules_from_env,
    find_anomalies,
)
from agent.config import Mode
from agent.demo import SyntheticOpti24
from agent.safety import ApprovalGate, BillingBudget
from agent.tools import build_tools

TODAY = __import__("datetime").date(2026, 10, 25)


def fill(tx_id: int, when: datetime, liters: float, card: str = "c-1") -> NS:
    return NS(id=str(tx_id), timestamp=when, card_id=card, poi_id="poi-1", type="P", product_name="ДТ",
              qty=liters, price="70.00", sum=str(Decimal("70.00") * Decimal(str(liters))), is_storno=False)


def reasons_of(result: dict[str, Any]) -> dict[str, str]:
    return {f["transaction_id"]: " ".join(f["reasons"]) for f in result["shown"]}


def test_defaults_are_650_liters_and_4_fills() -> None:
    assert DEFAULT_MAX_LITERS == Decimal("650") and DEFAULT_MAX_FILLS_PER_DAY == 4
    rules = AnomalyRules()
    assert rules.max_liters_per_fill == Decimal("650") and rules.max_fills_per_day == 4


def test_liters_threshold_is_strictly_above_650() -> None:
    items = [fill(1, datetime(2026, 10, 1, 12, 0), 650), fill(2, datetime(2026, 10, 2, 12, 0), 650.5)]
    reasons = reasons_of(find_anomalies(items))
    assert "1" not in reasons and "объём" in reasons["2"]


def test_fourth_fill_in_a_day_is_flagged_and_third_is_not() -> None:
    day = datetime(2026, 10, 3)
    three = [fill(i, day.replace(hour=8 + i), 100) for i in range(3)]
    assert find_anomalies(three)["suspicious_total"] == 0
    four = three + [fill(9, day.replace(hour=15), 100)]
    result = find_anomalies(four)
    assert result["suspicious_total"] == 4
    assert "4 заправки по карте за день (порог 4)" in reasons_of(result)["9"]


def test_rules_are_reported_in_the_answer() -> None:
    rules = find_anomalies([])["rules"]
    assert rules["max_liters_per_fill"] == "650" and rules["max_fills_per_day"] == 4


def test_env_overrides_thresholds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_SUSPICIOUS_MAX_LITERS", "500,5")
    monkeypatch.setenv("AGENT_SUSPICIOUS_MAX_FILLS_PER_DAY", "3")
    rules = anomaly_rules_from_env()
    assert rules.max_liters_per_fill == Decimal("500.5") and rules.max_fills_per_day == 3


@pytest.mark.parametrize(("name", "value"), [
    ("AGENT_SUSPICIOUS_MAX_LITERS", "много"),
    ("AGENT_SUSPICIOUS_MAX_LITERS", "0"),
    ("AGENT_SUSPICIOUS_MAX_FILLS_PER_DAY", "0"),
    ("AGENT_SUSPICIOUS_MAX_FILLS_PER_DAY", "2.5"),
])
def test_bad_env_values_fail_with_a_clear_message(monkeypatch: pytest.MonkeyPatch, name: str, value: str) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        anomaly_rules_from_env()


def tools_for(tmp_path: Path) -> dict[str, Any]:
    async def ask(_: str) -> bool:
        return True

    gate = ApprovalGate(mode=Mode.CONFIRM, ask=ask, audit_log=tmp_path / "audit.jsonl")
    return {t.name: t for t in build_tools(SyntheticOpti24(today=TODAY), BillingBudget(limit=50), gate)}  # type: ignore[arg-type]


def call(tools: dict[str, Any], args: dict[str, object]) -> dict[str, Any]:
    return json.loads(asyncio.run(tools["find_suspicious_fills"].call(args)))


PERIOD = {"date_from": "2026-09-01", "date_to": "2026-09-30"}


def test_tool_uses_defaults_without_arguments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_DEMO_DB_PATH", str(tmp_path / "demo.db"))
    out = call(tools_for(tmp_path), PERIOD)
    assert out["rules"]["max_liters_per_fill"] == "650" and out["rules"]["max_fills_per_day"] == 4


def test_tool_request_overrides_threshold_for_that_call_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_DEMO_DB_PATH", str(tmp_path / "demo.db"))
    tools = tools_for(tmp_path)
    strict = call(tools, {**PERIOD, "max_liters": 300})
    default = call(tools, PERIOD)
    assert strict["rules"]["max_liters_per_fill"] == "300"
    assert strict["suspicious_total"] > default["suspicious_total"]
    assert default["rules"]["max_liters_per_fill"] == "650"


def test_tool_rejects_bad_request_thresholds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_DEMO_DB_PATH", str(tmp_path / "demo.db"))
    tools = tools_for(tmp_path)
    assert call(tools, {**PERIOD, "max_liters": 0})["error"] == "bad_threshold"
    assert call(tools, {**PERIOD, "max_fills_per_day": 0})["error"] == "bad_threshold"
