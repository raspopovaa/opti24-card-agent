"""Проверка ответа кодом: числа из данных сессии и обязательные элементы темы."""

from __future__ import annotations

import asyncio
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from agent.classifier import ANSWER_TITLES, classify, hint, load_matrix
from agent.validation import ANSWER_CHECKS, Evidence, check_answer, required_elements
from evals.run_evals_claude import answer_problems, tool_results

TOOL_OUTPUT = json.dumps(
    {
        "period": "2026-04-01..2026-04-30",
        "total": "11 600,00 ₽",
        "liters": 240.0,
        "client_price": "48.50",
        "fee_rate": 0.0375,
        "operations": 8,
        "card": "****2323",
    },
    ensure_ascii=False,
)


def evidence(*payloads: object) -> Evidence:
    found = Evidence()
    for payload in payloads:
        found.add(payload)
    return found


def test_numbers_from_tool_data_pass_in_any_russian_format() -> None:
    answer = (
        "Расходы за апрель (01.04–30.04.2026): списано 11 600,00 ₽, около 11,6 тыс. ₽, 240 л, "
        "цена 48,50 ₽/л, сбор 3,75 %. Карта ****2323, 8 операций."
    )
    assert check_answer(answer, evidence(TOOL_OUTPUT), ["spending_report"]).ok


def test_number_missing_from_data_is_reported_with_its_unit() -> None:
    check = check_answer("Списано 12 345 ₽ по цене 51,20 ₽/л за апрель.", evidence(TOOL_OUTPUT), ["spending_report"])
    assert check.unverified == ("12 345 ₽", "51,20 ₽")
    assert "12 345 ₽" in check.note() and "без нужного вызова инструмента" in check.note()
    assert check.problems() == ["число не из данных: 12 345 ₽", "число не из данных: 51,20 ₽"]


def test_digit_groups_with_thin_and_narrow_spaces_are_one_number() -> None:
    # Регрессия: «8\u2009768,54 ₽» читалось как «768,54 ₽» и помечалось ложно.
    data = evidence('{"sum_left": "8768.54"}')
    for space in (" ", "\u00a0", "\u202f", "\u2009", "\u2007"):
        assert check_answer(f"Остаток 8{space}768,54 ₽.", data, []).ok, repr(space)
    assert not check_answer("Остаток 1\u2009200\u2009000 ₽.", data, []).ok  # крупная сумма — не номер


def test_rounding_must_match_the_precision_written_in_the_answer() -> None:
    data = evidence('{"total": 11643.4}')
    assert check_answer("Итого 11 643 ₽.", data, []).ok
    assert not check_answer("Итого 11 600 ₽.", data, []).ok  # не округление 11 643
    assert check_answer("Итого около 11 600 ₽.", data, []).ok  # «около» допускает 1 %


def test_dates_ids_years_and_small_counts_are_not_values() -> None:
    answer = "За 2026 год по карте 7826010000000001 (ID 123456) было 3 заправки, последняя 14.09 в 23:40."
    assert check_answer(answer, Evidence(), []).ok


def test_instruction_and_question_numbers_count_as_data() -> None:
    from agent.prompt import SYSTEM_PROMPT

    data = evidence(SYSTEM_PROMPT, "[Сегодня 2026-10-19] Хватит ли 500 запросов?")
    assert check_answer("На тарифе Free в пакете 500 запросов, сбор 3,75 %.", data, []).ok


def test_required_elements_follow_the_first_tool_of_the_topic() -> None:
    assert required_elements(["spending_report"]) == ["period"]
    assert required_elements(["diagnose_fueling"]) == ["probable"]
    assert set(required_elements(["lookup_knowledge"])) == {"clause", "typical"}
    assert required_elements(["list_cards", "get_contract_summary"]) == []


@pytest.mark.parametrize(
    ("answer", "tools", "missing"),
    [
        ("Списано 11 600,00 ₽.", ["spending_report"], ("период",)),
        ("За апрель списано 11 600,00 ₽.", ["spending_report"], ()),
        ("Карта заблокирована.", ["diagnose_fueling"], ("оговорка, что причина вероятная",)),
        ("Вероятная причина: карта заблокирована.", ["diagnose_fueling"], ()),
        ("Какую карту проверить?", ["diagnose_fueling"], ()),  # уточняющий вопрос
        ("Не удалось получить данные карты.", ["diagnose_fueling"], ()),  # сообщение о неудаче
        ("Оплата до 10 числа.", ["lookup_knowledge"], ("пункт или раздел договора", "оговорка о типовой редакции договора")),
        ("По п. 4.2 типового договора оплата до 10 числа.", ["lookup_knowledge"], ()),
    ],
)
def test_required_elements(answer: str, tools: list[str], missing: tuple[str, ...]) -> None:
    assert check_answer(answer, Evidence(), tools).missing == missing


def test_matrix_requirements_are_known_and_attached_to_tools() -> None:
    assert ANSWER_TITLES.keys() == ANSWER_CHECKS.keys()
    for topic in load_matrix():
        assert set(topic.answer_must) <= ANSWER_TITLES.keys(), topic.id
        assert not topic.answer_must or topic.tools, topic.id


def test_hint_tells_the_model_what_the_answer_must_name() -> None:
    assert "в ответе обязательно: оговорка, что причина вероятная" in hint(classify("Почему карта 7826010000000001 не работает?"))


def test_agent_appends_the_check_to_a_wrong_answer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from agent.config import Settings
    from agent.core import FuelCardAgent
    from tests.test_agent import FakeOpti24

    monkeypatch.setenv("AGENT_PROVIDER", "custom")
    monkeypatch.setenv("AGENT_TOOL_ROUTING", "off")
    monkeypatch.setenv("AGENT_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    monkeypatch.delenv("AGENT_TOPIC_LOG", raising=False)

    def reply(content: str | None, calls: list[NS] | None) -> NS:
        message = NS(content=content, tool_calls=calls)
        message.model_dump = lambda **_: {"role": "assistant", "content": content}  # type: ignore[attr-defined]
        return NS(choices=[NS(message=message)], usage=None)

    summary = NS(id="t1", function=NS(name="get_contract_summary", arguments="{}"))
    responses = iter(
        [
            reply(None, [summary]),
            reply("Доступно 123 139,19 ₽.", None),
            reply("Доступно 999 999,00 ₽.", None),  # второй вопрос: цифра не из данных
        ]
    )

    async def create(**_: object) -> NS:
        return next(responses)

    async def never(_: str) -> bool:
        return False

    agent = FuelCardAgent(settings=Settings.from_env(), opti24=FakeOpti24([]), ask_human=never)  # type: ignore[arg-type]
    agent.llm = NS(chat=NS(completions=NS(create=create)))

    first = asyncio.run(agent.ask("Сколько денег осталось на договоре?"))
    assert first == "Доступно 123 139,19 ₽." and agent.last_check.ok

    second = asyncio.run(agent.ask("А сколько было?"))
    assert second.startswith("Доступно 999 999,00 ₽.\n\n")
    assert "Проверка ответа:" in second  # вопрос «А сколько было?» не относится к теме: инструмент не назван
    assert agent.last_check.unverified == ("999 999,00 ₽",)
    assert agent.history[-1]["content"] == second  # человек и история видят одно и то же


def test_claude_code_stream_answer_is_checked_against_tool_results() -> None:
    lines = [
        json.dumps({"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t", "content": [{"type": "text", "text": TOOL_OUTPUT}]},
        ]}}),
        "не json",
    ]
    assert tool_results(lines) == [TOOL_OUTPUT]
    calls = [("spending_report", {"date_from": "2026-04-01"})]
    assert answer_problems("Расходы за апрель?", lines, calls, "За апрель списано 11 600,00 ₽.") == []
    assert answer_problems("Расходы за апрель?", lines, calls, "Списано 5 000 ₽.") == [
        "число не из данных: 5 000 ₽",
        "в ответе нет: период",
    ]


def test_tools_give_computed_values_so_the_model_does_not_derive_them(tmp_path: Path) -> None:
    """Остатки лимитов и доли пакета считает код: прогон evals показал, что иначе их считает модель."""
    from agent.config import Mode
    from agent.safety import ApprovalGate, BillingBudget
    from agent.tools import build_tools
    from evals.run_evals import FakeOpti24, never

    tools = {
        t.name: t
        for t in build_tools(FakeOpti24(), BillingBudget(10), ApprovalGate(Mode.READ_ONLY, never, tmp_path / "a.jsonl"))  # type: ignore[arg-type]
    }
    usage = json.loads(asyncio.run(tools["get_api_usage"].call({})))
    assert (usage["used_percent"], usage["left_percent"]) == ("2.4", "97.6")  # 12 и 488 из 500
    diagnosis = json.loads(asyncio.run(tools["diagnose_fueling"].call({"card_id": "c1", "decline_code": "51"})))
    assert diagnosis["checks"]["limits"]["items"][0]["sum_left"] == "8768.54"
    assert check_answer("Использовано 2,4 % пакета, осталось 97,6 %.", evidence(json.dumps(usage)), []).ok


def test_openai_loop_records_only_executed_calls_with_fixed_arguments(tmp_path: Path) -> None:
    from agent.config import Mode
    from agent.core_types import Usage
    from agent.llm_openai import run_openai_loop
    from tests.test_agent import make_tools

    _, budget, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path, budget_limit=5)

    def reply(content: str | None, calls: list[NS] | None) -> NS:
        message = NS(content=content, tool_calls=calls)
        message.model_dump = lambda **_: {"role": "assistant", "content": content}  # type: ignore[attr-defined]
        return NS(choices=[NS(message=message)], usage=None)

    calls = [
        NS(id="a", function=NS(name="spending_report", arguments='{"from": "2026-08-01", "to": "2026-08-31"}')),
        NS(id="b", function=NS(name="no_such_tool", arguments="{}")),
        NS(id="c", function=NS(name="spending_report", arguments='{"day": "2026-08-01"}')),
    ]
    responses = iter([reply(None, calls), reply("Готово", None)])

    async def create(**_: object) -> NS:
        return next(responses)

    llm = NS(chat=NS(completions=NS(create=create)))
    _, called = asyncio.run(
        run_openai_loop(llm, "m", "sys", [{"role": "user", "content": "q"}], list(tools.values()), 5, Usage())
    )
    assert called == [("spending_report", {"date_from": "2026-08-01", "date_to": "2026-08-31"})]


def test_bold_markdown_does_not_hide_the_number() -> None:
    # Регрессия из прогона Cloud.ru: «**123 139,19 ₽**» проверялось как «139,19 ₽» и помечалось ложно.
    data = evidence('{"balance": "123 139,19 ₽", "fee": 75}')
    assert check_answer("На договоре **123 139,19 ₽** доступно.", data, []).ok
    assert check_answer("Остаток **8 768,54 ₽**, сбор __75 %__.", evidence('{"left": "8768.54", "fee": 75}'), []).ok


def test_masked_card_numbers_are_not_values() -> None:
    assert check_answer("Карта ****2323 заблокирована, см. **c1**.", Evidence(), []).ok


def test_no_data_answers_are_not_held_to_content_requirements() -> None:
    # Регрессия из прогона: «операций не было» — требование «требует проверки» здесь не применяется.
    answer = "За август операций не было. Подозрительных заправок не обнаружено."
    assert check_answer(answer, Evidence(), ["find_suspicious_fills"]).ok
    assert check_answer("По ДТ за сентябрь 0 заправок.", Evidence(), ["fuel_price_analytics"]).ok
    assert not check_answer("Найдено 3 подозрительные заправки.", Evidence(), ["find_suspicious_fills"]).ok


MAY_2024 = json.dumps(
    {
        "product_prices": [
            {
                "product": "Аи-95",
                "liters": 200.23,
                "amount": 9351.63,
                "price_per_liter": 46.71,
                "gas_station_price": 48.64,
                "discount_per_liter": 1.93,
                "discount_total": -386.71,
                "price_with_fee": 48.46,
                "service_fee": 350.87,
            }
        ],
        "service_fee_percent": 3.75,
    }
)


def test_numbers_derived_from_api_fields_are_accepted() -> None:
    # Регрессия из диалога: «1,93 = 48,64 − 46,71» и «386,71 ≈ 1,93 × 200,23» помечались ложно.
    answer = (
        "Май: 200,23 л, 46,71 ₽/л, АЗС 48,64 ₽/л, скидка 1,93 ₽/л (386,71 ₽ всего), "
        "со сбором 48,46 ₽/л, сбор 350,87 ₽, ставка 3,75 %."
    )
    assert check_answer(answer, evidence(MAY_2024), ["fuel_price_analytics"]).ok


def test_arithmetic_does_not_accept_arbitrary_numbers() -> None:
    # Погрешность округления крупных величин большая: произвольная сумма не должна «совпасть».
    data = evidence(MAY_2024)
    for fabricated in ("5 000 ₽", "12 345 ₽", "999 999 ₽", "70 000 ₽"):
        assert check_answer(f"Итого {fabricated}.", data, []).unverified == (fabricated,), fabricated


def test_instruction_forbids_invented_numbers_and_requires_the_formula() -> None:
    from agent.prompt import SYSTEM_PROMPT

    assert "Цифры берутся только из ответов инструментов" in SYSTEM_PROMPT
    assert "напиши в ответе, как оно получено" in SYSTEM_PROMPT
    assert "Сырой JSON ответа инструмента в чат не выводи" in SYSTEM_PROMPT


def _agent_with(monkeypatch, tmp_path, verify: str, replies):  # type: ignore[no-untyped-def]
    from agent.config import Settings
    from agent.core import FuelCardAgent
    from tests.test_agent import FakeOpti24

    monkeypatch.setenv("AGENT_PROVIDER", "custom")
    monkeypatch.setenv("AGENT_TOOL_ROUTING", "off")
    monkeypatch.setenv("AGENT_VERIFY_ANSWER", verify)
    monkeypatch.setenv("AGENT_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    monkeypatch.delenv("AGENT_TOPIC_LOG", raising=False)
    fake = FakeOpti24([])

    async def never(_: str) -> bool:
        return False

    agent = FuelCardAgent(settings=Settings.from_env(), opti24=fake, ask_human=never)  # type: ignore[arg-type]
    queue = iter(replies)

    async def create(**_: object) -> NS:
        return next(queue)

    agent.llm = NS(chat=NS(completions=NS(create=create)))
    return agent, fake


def _text(content: str) -> NS:
    return NS(choices=[NS(message=NS(content=content, tool_calls=None))], usage=None)


def test_verification_calls_the_topic_tool_and_confirms_the_number(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    # Модель назвала число без вызова инструмента: проверка сама вызывает get_contract_summary и сверяет.
    agent, _ = _agent_with(monkeypatch, tmp_path, "on", [_text("Доступно 123 139,19 ₽.")])
    answer = asyncio.run(agent.ask("Сколько денег осталось на договоре?"))
    assert "сверены повторным вызовом get_contract_summary" in answer
    # Вызов проверки — действие системы, а не выбор модели: в tool_calls (оценка выбора) его нет.
    assert agent.last_check.ok and agent.tool_calls == []


def test_verification_reports_numbers_that_still_do_not_match(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    agent, _ = _agent_with(monkeypatch, tmp_path, "on", [_text("Доступно 999 999,00 ₽.")])
    answer = asyncio.run(agent.ask("Сколько денег осталось на договоре?"))
    assert "часть чисел не подтвердилась" in answer and "999 999,00 ₽" in agent.last_check.unverified[0]


def test_verification_can_be_switched_off(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    agent, _ = _agent_with(monkeypatch, tmp_path, "off", [_text("Доступно 123 139,19 ₽.")])
    answer = asyncio.run(agent.ask("Сколько денег осталось на договоре?"))
    assert agent.tool_calls == [] and "Проверка" not in answer.replace("Проверка ответа", "")  # только примечание


def test_plan_repeats_question_calls_but_never_write_tools() -> None:
    from agent.verification import plan_verification

    tools = {"get_limits": object(), "block_cards": object()}
    plan = plan_verification(
        [("get_limits", {"card_id": "c1"}), ("block_cards", {"card_ids": ["c1"]}), ("get_limits", {"card_id": "c1"})],
        None, date(2026, 10, 22), tools,  # type: ignore[arg-type]
    )
    assert plan.calls == [("get_limits", {"card_id": "c1"})] and plan.skipped == []


def test_plan_without_calls_uses_topic_tool_with_previous_month(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from agent.classifier import classify
    from agent.verification import plan_verification

    class Tool:
        def to_dict(self) -> dict:
            return {"input_schema": {"required": ["date_from", "date_to"]}}

    topic = classify("Покажи расходы").primary
    plan = plan_verification([], topic, date(2026, 10, 22), {"spending_report": Tool()})  # type: ignore[dict-item]
    assert plan.calls == [("spending_report", {"date_from": "2026-09-01", "date_to": "2026-09-30"})]


def test_plan_does_not_invent_a_card(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from agent.classifier import classify
    from agent.verification import plan_verification

    class Tool:
        def to_dict(self) -> dict:
            return {"input_schema": {"required": ["card_id"]}}

    topic = classify("Почему карта не работает?").primary
    plan = plan_verification([], topic, date(2026, 10, 22), {"diagnose_fueling": Tool()})  # type: ignore[dict-item]
    assert plan.calls == [] and "card_id" in plan.skipped[0]
