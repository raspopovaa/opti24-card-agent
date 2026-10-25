"""Изменения: что видит человек в подтверждении, исход в журнале и неизвестный результат."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

from apisdkopti24.errors import APIError, APIResponseTimeoutError, ServerError

from agent.config import Mode
from agent.safety import ApprovalGate, BillingBudget
from agent.tools import build_tools
from tests.test_agent import FakeOpti24


def setup(tmp_path: Path, answer: bool = True, budget: int = 5) -> tuple[FakeOpti24, dict[str, Any], list[str]]:
    fake = FakeOpti24([])
    asked: list[str] = []

    async def ask(text: str) -> bool:
        asked.append(text)
        return answer

    gate = ApprovalGate(mode=Mode.CONFIRM, ask=ask, audit_log=tmp_path / "audit.jsonl")
    tools = {t.name: t for t in build_tools(fake, BillingBudget(limit=budget), gate)}  # type: ignore[arg-type]
    return fake, tools, asked


def call(tools: dict[str, Any], name: str, args: dict[str, object]) -> dict[str, Any]:
    return json.loads(asyncio.run(tools[name].call(args)))


def audit(tmp_path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()]


def test_block_shows_card_from_api_and_labels_model_reason(tmp_path: Path) -> None:
    _, tools, asked = setup(tmp_path)
    call(tools, "block_cards", {"card_ids": ["c1"], "reason": "водитель уволился"})
    assert "7826010000000001 (ID c1)" in asked[0] and "«Газель, Иванов»" in asked[0]
    assert "сейчас: Активна" in asked[0] and "Причина со слов агента: водитель уволился" in asked[0]


def test_unknown_card_is_not_offered_for_confirmation(tmp_path: Path) -> None:
    fake, tools, asked = setup(tmp_path)
    out = call(tools, "block_cards", {"card_ids": ["c1", "zz"], "reason": "x"})
    assert out["error"] == "not_found" and "zz" in out["message"]
    assert not asked and fake.blocked == []


def test_audit_has_decision_and_outcome_with_same_id(tmp_path: Path) -> None:
    _, tools, _ = setup(tmp_path)
    call(tools, "block_cards", {"card_ids": ["c1"], "reason": "секретная причина"})
    decision, outcome = audit(tmp_path)
    assert decision["event"] == "decision" and outcome["event"] == "outcome"
    assert decision["id"] == outcome["id"] and decision["decided_by"] == "terminal"
    assert decision["operator"]  # кто из людей подтвердил: AGENT_OPERATOR или пользователь ОС
    assert decision["target"] == {"card_ids": ["c1"], "cards": ["…0001"]} and decision["args_sha256"]
    assert outcome["status"] == "done" and outcome["done"] == ["c1"]
    assert "секретная" not in (tmp_path / "audit.jsonl").read_text(encoding="utf-8")


def test_timeout_after_confirmation_reports_unknown_outcome(tmp_path: Path) -> None:
    fake, tools, _ = setup(tmp_path)

    async def hang(**_: object) -> NS:
        raise APIResponseTimeoutError("timeout")

    fake.cards.block_card = hang
    out = call(tools, "block_cards", {"card_ids": ["c1"], "reason": "x"})
    assert out["status"] == "unknown" and "НЕ вызывай инструмент повторно" in out["message"]
    assert audit(tmp_path)[-1]["status"] == "unknown"


def test_server_error_is_unknown_but_client_error_is_not(tmp_path: Path) -> None:
    fake, tools, _ = setup(tmp_path)

    async def server_error(**_: object) -> NS:
        raise ServerError(status_code=502, message="bad gateway", http_status_code=502)

    async def rejected(**_: object) -> NS:
        raise APIError(status_code=400, message="bad", http_status_code=400)

    fake.region_limits.set_region_limit = server_error
    args = {"card_id": "c1", "region": "MOS", "limit_type": 2}
    assert call(tools, "set_region_limit", args)["status"] == "unknown"
    fake.region_limits.set_region_limit = rejected
    assert call(tools, "set_region_limit", args)["error"] == "APIError"
    assert [r["status"] for r in audit(tmp_path) if r["event"] == "outcome"] == ["unknown", "error"]


def test_set_limit_shows_old_and_new_limit(tmp_path: Path) -> None:
    _, tools, asked = setup(tmp_path)
    out = call(
        tools, "set_limit", {"card_id": "c1", "rubles": 5000, "period_number": 1, "period_type": 3, "limit_id": "l1"}
    )
    assert out["status"] == "saved"
    assert "Было: лимит l1: 9000.0 ₽ (израсходовано 231.46)" in asked[0]
    assert "Станет: 5000 ₽ на сутки" in asked[0] and "карта 7826010000000001" in asked[0]


def test_set_limit_with_unknown_limit_id_is_refused_without_asking(tmp_path: Path) -> None:
    _, tools, asked = setup(tmp_path)
    out = call(tools, "set_limit", {"card_id": "c1", "liters": 10, "period_type": 3, "limit_id": "nope"})
    assert out["error"] == "not_found" and not asked


def test_remove_rule_needs_access_rules_first(tmp_path: Path) -> None:
    _, tools, asked = setup(tmp_path)
    out = call(tools, "remove_region_limit", {"region_limit_id": "r1"})
    assert out["error"] == "unknown_rule" and not asked
    call(tools, "get_access_rules", {"card_id": "c1"})
    out = call(tools, "remove_region_limit", {"region_limit_id": "r1"})
    assert out["status"] == "removed"
    assert "региональный лимит r1: RUS/MOS запрещён для карты c1" in asked[0] and "необратимо" in asked[0]


def test_comment_change_shows_old_comment(tmp_path: Path) -> None:
    _, tools, asked = setup(tmp_path)
    call(tools, "set_card_comment", {"card_id": "c1", "comment": "Газель, Петров"})
    assert "было «Газель, Иванов», станет «Газель, Петров»" in asked[0]


def test_human_is_not_asked_when_budget_is_spent(tmp_path: Path) -> None:
    fake, tools, asked = setup(tmp_path, budget=0)
    out = call(tools, "block_cards", {"card_ids": ["c1"], "reason": "x"})
    assert out["error"] == "BudgetExceededError" and not asked and fake.blocked == []


def test_card_lookup_reads_next_pages_until_card_is_found(tmp_path: Path) -> None:
    fake, tools, asked = setup(tmp_path)
    pages: list[object] = []
    far = NS(id="far", number="7826010000009999", status="Active", status_name="Активна",
             group_name=None, comment=None)

    async def paged(**kwargs: object) -> NS:
        pages.append(kwargs.get("page"))
        batch = [far] if kwargs.get("page") == 2 else [fake.more_cards[0]]
        return NS(data=NS(result=batch, total_count=2))

    fake.cards.get_cards_v2 = paged
    call(tools, "block_cards", {"card_ids": ["far"], "reason": "x"})
    # Страницы 1–2 до подтверждения и ещё раз после блокировки: сверка статуса.
    assert pages == [1, 2, 1, 2] and "7826010000009999 (ID far)" in asked[0]


def test_operator_name_comes_from_env(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("AGENT_OPERATOR", "Логист Петров")
    _, tools, _ = setup(tmp_path)
    call(tools, "block_cards", {"card_ids": ["c1"], "reason": "x"})
    assert audit(tmp_path)[0]["operator"] == "Логист Петров"


def test_block_is_rechecked_in_card_list(tmp_path: Path) -> None:
    fake, tools, _ = setup(tmp_path)
    out = call(tools, "block_cards", {"card_ids": ["c1"], "reason": "x"})
    assert out["verified"]["status"] == "match"
    assert audit(tmp_path)[-1]["verified"] == "match"

    async def accepted_but_ignored(*, card_ids: list[str], block: bool) -> NS:
        return NS(data=list(card_ids))  # оператор ответил «принято», статус не изменился

    fake.cards.block_card = accepted_but_ignored
    out = call(tools, "block_cards", {"card_ids": ["a"], "reason": "x"})
    assert out["verified"]["status"] == "mismatch" and "не повторяй" in out["verified"]["message"]


def test_comment_is_rechecked(tmp_path: Path) -> None:
    _, tools, _ = setup(tmp_path)
    out = call(tools, "set_card_comment", {"card_id": "c1", "comment": "Камаз, Петров"})
    assert out["verified"]["status"] == "match"


def test_set_limit_is_rechecked_by_free_get_limits(tmp_path: Path) -> None:
    fake, tools, _ = setup(tmp_path)
    # В get_limits у l1: 9000 ₽ на период (1, 5); тот же лимит после изменения совпадает.
    same = {"card_id": "c1", "rubles": 9000, "period_type": 5, "limit_id": "l1"}
    assert call(tools, "set_limit", same)["verified"]["status"] == "match"
    other = {**same, "rubles": 5000}
    out = call(tools, "set_limit", other)
    assert out["verified"]["status"] == "mismatch" and fake.saved_limits


def test_remove_limit_is_rechecked_and_rules_say_not_checked(tmp_path: Path) -> None:
    fake, tools, _ = setup(tmp_path, budget=10)
    out = call(tools, "remove_limit", {"limit_id": "l1", "card_id": "c1"})
    assert out["verified"]["status"] == "mismatch"  # в фейке лимит после удаления остаётся

    async def empty(**_: object) -> NS:
        return NS(data=NS(result=[], total_count=0))

    call(tools, "get_access_rules", {"card_id": "c1"})
    fake.limits.get_limits = empty
    out = call(tools, "set_region_limit", {"card_id": "c1", "region": "SPE", "limit_type": 2})
    assert out["verified"]["status"] == "not_checked" and "get_access_rules" in out["verified"]["message"]
