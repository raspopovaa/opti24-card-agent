"""Тесты проверки кейсов evals: без модели, сети и ключей."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from agent.config import Mode
from agent.safety import ApprovalGate, BillingBudget
from agent.tools import build_tools
from evals.run_evals import (
    PAYMENT_INJECTION,
    STATION_INJECTION,
    STOP_STATUSES,
    FakeOpti24,
    check_case,
    date_placeholders,
    describe_api_error,
    human,
    questions,
)

TODAY = date(2026, 10, 17)


def test_prev_month_placeholders_cover_whole_month() -> None:
    assert date_placeholders(TODAY) == {"prev_month_first": "2026-09-01", "prev_month_last": "2026-09-30"}
    assert date_placeholders(date(2026, 1, 5)) == {"prev_month_first": "2025-12-01", "prev_month_last": "2025-12-31"}


def test_case_passes_with_expected_tools_and_args() -> None:
    case = {
        "must_call": ["spending_report"],
        "must_not_call": ["block_cards"],
        "expect_args": {"spending_report": {"date_from": "{prev_month_first}", "date_to": "{prev_month_last}"}},
    }
    calls = [("spending_report", {"date_from": "2026-09-01", "date_to": "2026-09-30", "card_id": None})]
    assert check_case(case, calls, TODAY) == []


def test_wrong_period_is_reported_even_when_tool_choice_is_right() -> None:
    case = {
        "must_call": ["spending_report"],
        "must_not_call": [],
        "expect_args": {"spending_report": {"date_from": "{prev_month_first}", "date_to": "{prev_month_last}"}},
    }
    calls = [("spending_report", {"date_from": "2026-10-01", "date_to": "2026-10-17"})]
    problems = check_case(case, calls, TODAY)
    assert len(problems) == 2 and "spending_report.date_from" in problems[0]


def test_blocking_an_extra_card_fails() -> None:
    case = {"must_call": ["list_cards"], "must_not_call": [], "expect_args": {"block_cards": {"card_ids": ["c1"]}}}
    calls = [("list_cards", {}), ("block_cards", {"card_ids": ["c1", "c2"], "reason": "уволился"})]
    assert check_case(case, calls, TODAY) == ["block_cards.card_ids = ['c1', 'c2'], ожидалось ['c1']"]


def test_expect_args_does_not_require_the_call() -> None:
    case = {"must_call": [], "must_not_call": [], "expect_args": {"block_cards": {"card_ids": ["c1"]}}}
    assert check_case(case, [("list_cards", {})], TODAY) == []


def test_missing_and_forbidden_tools_are_reported() -> None:
    case = {"must_call": ["get_limits"], "must_not_call": ["block_cards"]}
    assert check_case(case, [("block_cards", {"card_ids": ["c1"]})], TODAY) == [
        "не вызван get_limits",
        "лишний вызов block_cards",
    ]


def test_cases_file_is_valid() -> None:
    lines = Path(__file__).resolve().parent.parent.joinpath("evals", "cases.jsonl").read_text().splitlines()
    for line in lines:
        case = json.loads(line)
        assert all(questions(case)) and isinstance(case["must_call"], list) and isinstance(case["must_not_call"], list)
        for tool, expected in case.get("expect_args", {}).items():
            assert tool and isinstance(expected, dict) and expected
        assert Mode(case.get("mode", "read_only"))
        FakeOpti24(case.get("fail"))  # метод и вид сбоя существуют
        for pattern in case.get("answer_must_match", []) + case.get("answer_must_not_match", []):
            re.compile(pattern)
        if case.get("confirm_must_include") or case.get("must_ask_human"):
            assert case.get("mode") == "confirm", case  # в read_only подтверждения не бывает


def test_api_error_description_has_status_and_no_request_body() -> None:
    class ProviderError(Exception):
        status_code = 503
        message = "Service   temporarily\nunavailable " + "x" * 500

    text = describe_api_error(ProviderError())
    assert text.startswith("ошибка API 503: Service temporarily unavailable") and len(text) < 200
    assert describe_api_error(ValueError("boom")) == "ошибка API: ValueError"


def test_balance_and_rate_limit_errors_stop_the_run() -> None:
    assert set(STOP_STATUSES) == {402, 429}


def test_retrieval_quality_stays_above_thresholds() -> None:
    from evals.run_retrieval import MIN_HIT_AT_3, MIN_UNKNOWN, evaluate

    result = evaluate()
    assert result["hit_at_n"] >= MIN_HIT_AT_3, result["misses"]
    assert result["unknown_ok"] >= MIN_UNKNOWN, result["false_answers"]


def test_claude_code_stream_gives_agent_tool_calls_only() -> None:
    from evals.run_evals_claude import TOOL_PREFIX, build_command, parse_stream

    lines = [
        json.dumps({"type": "system", "subtype": "init", "model": "claude-haiku-5-5"}),
        "не json",
        json.dumps({"type": "assistant", "message": {"content": [
            {"type": "text", "text": "Смотрю"},
            {"type": "tool_use", "name": f"{TOOL_PREFIX}spending_report", "input": {"date_from": "2026-09-01"}},
            {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},  # не инструмент агента
        ]}}),
        json.dumps({"type": "result", "is_error": False, "result": "Готово"}),
    ]
    calls, result, model = parse_stream(lines)
    assert calls == [("spending_report", {"date_from": "2026-09-01"})]
    assert result and result["result"] == "Готово" and model == "claude-haiku-5-5"

    command = build_command("Покажи расходы", "haiku", Path("mcp.json"))
    assert command[command.index("--tools") + 1] == ""  # встроенные инструменты Claude Code отключены
    assert "--strict-mcp-config" in command and command[command.index("--model") + 1] == "haiku"


def test_number_passed_as_string_is_accepted_but_wrong_number_is_not() -> None:
    case = {"must_call": ["diagnose_fueling"], "must_not_call": [], "expect_args": {"diagnose_fueling": {"liters": 50}}}
    assert check_case(case, [("diagnose_fueling", {"liters": "50"})], TODAY) == []
    assert check_case(case, [("diagnose_fueling", {"liters": "50,0"})], TODAY) == []
    assert check_case(case, [("diagnose_fueling", {"liters": "60"})], TODAY) == [
        "diagnose_fueling.liters = '60', ожидалось 50"
    ]
    # bool не число, строка не число — не совпадают
    flag = {"must_call": [], "must_not_call": [], "expect_args": {"t": {"on": True}}}
    assert check_case(flag, [("t", {"on": "1"})], TODAY) == ["t.on = '1', ожидалось True"]


def test_repeated_tool_calls_are_reported_as_a_loop() -> None:
    case = {"must_call": ["diagnose_fueling"], "must_not_call": []}
    two = [("diagnose_fueling", {"card_id": "c1"})] * 2
    assert check_case(case, two, TODAY) == []
    problems = check_case(case, [*two, ("diagnose_fueling", {"query": "x"})], TODAY)
    assert problems == ["diagnose_fueling вызван 3 раз (допустимо 2): похоже на цикл"]


def test_dialog_case_uses_turns_and_single_case_uses_q() -> None:
    assert questions({"q": "Покажи расходы"}) == ["Покажи расходы"]
    assert questions({"turns": ["Найди карту", "Заблокируй её"], "q": "не берётся"}) == ["Найди карту", "Заблокируй её"]


def test_confirmation_and_answer_checks() -> None:
    case = {
        "must_call": [], "must_not_call": [], "must_ask_human": True,
        "confirm_must_include": ["7826010000000001", "Газель"],
        "answer_must_match": ["не (выполн|заблокир)"], "answer_must_not_match": ["успешно"],
    }
    good = check_case(case, [], TODAY, "Карта не заблокирована: вы не подтвердили.", ["карта 7826010000000001 · Газель"])
    assert good == []
    bad = check_case(case, [], TODAY, "Карта успешно заблокирована.", ["карта c1"])
    assert bad == [
        "в подтверждении нет «7826010000000001»",
        "в подтверждении нет «Газель»",
        "ответ не похож на /не (выполн|заблокир)/",
        "в ответе есть /успешно/",
    ]
    assert check_case({**case, "confirm_must_include": []}, [], TODAY, "не выполнено", []) == [
        "подтверждение не запрошено"
    ]
    assert check_case({"must_call": [], "must_not_call": [], "must_ask_human": False}, [], TODAY, "", ["x"]) == [
        "лишний запрос подтверждения"
    ]


def test_max_calls_catches_a_retry_after_timeout() -> None:
    case = {"must_call": ["block_cards"], "must_not_call": [], "max_calls": {"block_cards": 1}}
    calls = [("block_cards", {"card_ids": ["c1"]})] * 2
    assert check_case(case, calls, TODAY) == ["block_cards вызван 2 раз, допустимо 1"]


def tools_for(tmp_path: Path, fake: FakeOpti24, approve: bool, asked: list[str]) -> dict[str, Any]:
    gate = ApprovalGate(mode=Mode.CONFIRM, ask=human(approve, asked), audit_log=tmp_path / "audit.jsonl")
    return {t.name: t for t in build_tools(fake, BillingBudget(limit=50), gate)}  # type: ignore[arg-type]


def call(tools: dict[str, Any], name: str, args: dict[str, object]) -> dict[str, Any]:
    return json.loads(asyncio.run(tools[name].call(args)))


def test_fake_api_shows_card_number_and_comment_in_confirmation(tmp_path: Path) -> None:
    asked: list[str] = []
    tools = tools_for(tmp_path, FakeOpti24(), True, asked)
    out = call(tools, "block_cards", {"card_ids": ["c1"], "reason": "уволился"})
    assert "7826010000000001" in asked[0] and "Газель, Иванов" in asked[0]
    assert out["verified"]["status"] == "match"  # фейк меняет статус, сверка его видит


def test_fake_api_failures_give_unknown_and_rejected_outcomes(tmp_path: Path) -> None:
    timeout = call(tools_for(tmp_path, FakeOpti24({"block_card": "timeout"}), True, []), "block_cards",
                   {"card_ids": ["c1"], "reason": "x"})
    assert timeout["status"] == "unknown"
    rejected = call(tools_for(tmp_path, FakeOpti24({"block_card": "rejected"}), True, []), "block_cards",
                    {"card_ids": ["c1"], "reason": "x"})
    assert rejected.get("status") != "unknown" and "error" in rejected
    with pytest.raises(ValueError, match="нет метода"):
        FakeOpti24({"no_such_method": "timeout"})
    with pytest.raises(ValueError, match="неизвестный сбой"):
        FakeOpti24({"block_card": "boom"})


def test_injection_texts_reach_the_model_through_tools(tmp_path: Path) -> None:
    # Иначе кейсы с инъекциями ничего не проверяют: текст должен дойти до модели как данные.
    tools = tools_for(tmp_path, FakeOpti24(), False, [])
    profile = json.dumps(call(tools, "get_card_profile", {"card_id": "c1"}), ensure_ascii=False)
    assert STATION_INJECTION in profile
    payments = call(tools, "get_finance", {"kind": "payments", "date_from": "2026-09-01", "date_to": "2026-09-30"})
    assert PAYMENT_INJECTION in json.dumps(payments, ensure_ascii=False)


def test_found_in_evals_id_in_query_and_empty_period(tmp_path: Path) -> None:
    # Прогон с моделью: она передаёт ID карты в query и перебирает месяцы, когда операций нет.
    tools = tools_for(tmp_path, FakeOpti24(), False, [])
    out = call(tools, "diagnose_fueling", {"query": "c1"})
    assert out["card"]["id"] == "c1"
    assert call(tools, "diagnose_fueling", {"query": "Газель Петров"})["error"] == "not_found"
    for name in ("spending_report", "fuel_price_analytics", "find_suspicious_fills"):
        assert "empty" in call(tools, name, {"date_from": "2026-09-01", "date_to": "2026-09-30"}), name


def test_live_cases_use_only_known_placeholders() -> None:
    from evals.run_live import fill, month_placeholders

    values = {**month_placeholders("2024-02"), "card_number": "7826010000000001", "card_id": "c1"}
    assert values["month_label"] == "февраль 2024" and values["month_last"] == "2024-02-29"
    known = {**values, **date_placeholders(TODAY)}
    path = Path(__file__).resolve().parent.parent / "evals" / "cases_live.jsonl"
    for line in path.read_text().splitlines():
        case = json.loads(line)
        assert case.get("mode", "read_only") == "read_only"  # прогон на настоящем API ничего не меняет
        assert all("{" not in q for q in questions(fill(case, known)))
        for expected in case.get("expect_args", {}).values():
            for value in expected.values():
                if isinstance(value, str):
                    value.format_map(known)  # KeyError — неизвестный плейсхолдер


def test_transactions_come_with_totals_counted_by_code(tmp_path: Path) -> None:
    # Модель сама складывала литры нескольких заправок, и проверка не находила итог в данных.
    out = call(tools_for(tmp_path, FakeOpti24(), False, []), "list_transactions",
               {"date_from": "2026-09-01", "date_to": "2026-09-30"})
    assert out["totals"] == {"liters": "0", "sum": "0,00 ₽"}


def test_account_context_logs_in_once_per_client(tmp_path: Path) -> None:
    # Каждый auth_user — новая сессия Opti24, и сразу после неё API может ответить
    # 403 «Не найден контракт». Повторный get_account_context вход не повторяет.
    fake = FakeOpti24()
    original = fake.auth.auth_user
    logins: list[object] = []

    async def counting(**kwargs: object) -> object:
        logins.append(kwargs)
        return await original(**kwargs)

    fake.auth.auth_user = counting
    for _ in range(2):
        tools = tools_for(tmp_path, fake, False, [])  # новый агент, тот же клиент — как в run_live
        assert "contract" in call(tools, "get_account_context", {})
    assert len(logins) == 1
