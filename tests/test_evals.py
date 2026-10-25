"""Тесты проверки кейсов evals: без модели, сети и ключей."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from evals.run_evals import STOP_STATUSES, check_case, date_placeholders, describe_api_error

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
        assert case["q"] and isinstance(case["must_call"], list) and isinstance(case["must_not_call"], list)
        for tool, expected in case.get("expect_args", {}).items():
            assert tool and isinstance(expected, dict) and expected


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
