"""Выбор инструментов по матрице запросов: реестр групп, маршрутизатор, цикл и подключение к агенту."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from agent.classifier import classify, load_matrix
from agent.config import Mode, Settings
from agent.core import FuelCardAgent
from agent.core_types import Usage
from agent.llm_openai import run_openai_loop
from agent.tools.catalog import CORE, WRITE_TOOLS
from agent.tools.router import ToolRouter
from tests.test_agent import FakeOpti24, make_tools


def _router(tmp_path: Path, mode: Mode = Mode.CONFIRM) -> ToolRouter:
    _, _, tools = make_tools(mode, answer=False, tmp=tmp_path)
    return ToolRouter(list(tools.values()), mode)


def _size(tools: list) -> int:  # type: ignore[type-arg]
    return sum(len(json.dumps(t.to_dict(), ensure_ascii=False)) for t in tools)


def test_every_tool_is_reachable_through_the_core_or_a_topic(tmp_path: Path) -> None:
    _, _, tools = make_tools(Mode.CONFIRM, answer=False, tmp=tmp_path)
    reachable = {*CORE, *(name for topic in load_matrix() for name in topic.tools)}
    assert reachable == set(tools)
    assert WRITE_TOOLS <= set(tools)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Покажи платежи за сентябрь", {"get_finance"}),
        ("Какие лимиты на карте Газели Иванова?", {"get_limits", "list_cards"}),
        ("Почему карта c1 не работает?", {"diagnose_fueling"}),
        ("Какие отчёты уже заказаны?", {"get_report_status"}),
    ],
)
def test_matrix_opens_the_tools_of_the_question(tmp_path: Path, question: str, expected: set[str]) -> None:
    router = _router(tmp_path)
    router.preload(question)
    assert expected <= {t.name for t in router.visible()}


def test_unknown_question_opens_only_the_core(tmp_path: Path) -> None:
    router = _router(tmp_path)
    assert router.preload("Какая погода в Москве?") == set()
    assert {t.name for t in router.visible()} == {*CORE, "open_toolbox"}


def test_confident_eval_cases_get_their_tools_without_asking_the_model(tmp_path: Path) -> None:
    cases = [json.loads(line) for line in (Path(__file__).parent.parent / "evals" / "cases.jsonl").open()]
    confident = 0
    for case in cases:
        router = _router(tmp_path)
        if router.preload(case["q"]) or classify(case["q"]).confident:
            confident += 1
            visible = {t.name for t in router.visible()}
            assert set(case["must_call"]) <= visible, case["q"]
    # Остальные кейсы модель разбирает сама через каталог тем и open_toolbox.
    assert confident >= len(cases) * 0.8


def test_routing_makes_the_request_much_smaller(tmp_path: Path) -> None:
    router = _router(tmp_path, Mode.READ_ONLY)
    _, _, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path)
    full = _size(list(tools.values()))
    router.preload("Покажи расходы")
    assert _size(router.visible()) < full * 0.4
    assert _size([router.open_toolbox]) < 1500  # сам открыватель групп короткий


def test_read_only_hides_write_tools_and_confirm_shows_them(tmp_path: Path) -> None:
    for mode, shown in ((Mode.READ_ONLY, False), (Mode.CONFIRM, True)):
        router = _router(tmp_path, mode)
        router.preload("Заблокируй карту c1, водитель уволился")
        router.preload("Поставь лимит 300 литров в месяц на карту c1")
        names = {t.name for t in router.visible()}
        assert ("block_cards" in names) is shown and ("set_limit" in names) is shown


def test_open_toolbox_opens_group_and_keeps_it_open(tmp_path: Path) -> None:
    router = _router(tmp_path)
    assert "get_limits" not in {t.name for t in router.visible()}
    out = json.loads(asyncio.run(router.open_toolbox.call({"topic": " Limits "})))
    assert out["opened"] == "limits" and out["title"] == "Лимиты и правила" and "get_limits" in out["tools"]
    assert "get_limits" in {t.name for t in router.visible()}
    assert router.preload("а за август?") == set()  # уточняющий вопрос без слов темы
    assert "get_limits" in {t.name for t in router.visible()}  # группа осталась открытой


def test_open_toolbox_rejects_unknown_group(tmp_path: Path) -> None:
    out = json.loads(asyncio.run(_router(tmp_path).open_toolbox.call({"topic": "nope"})))
    assert out["error"] == "unknown_topic" and "spending" in out["sections"]


def test_loop_shows_newly_opened_tools_on_the_next_step(tmp_path: Path) -> None:
    router = _router(tmp_path)
    seen: list[set[str]] = []
    responses = iter(
        [
            _reply(None, [NS(id="c1", function=NS(name="open_toolbox", arguments='{"topic": "limits"}'))]),
            _reply("Готово", None),
        ]
    )

    async def create(**kwargs: object) -> NS:
        seen.append({t["function"]["name"] for t in kwargs["tools"]})  # type: ignore[index]
        return next(responses)

    llm = NS(chat=NS(completions=NS(create=create)))
    text, called = asyncio.run(
        run_openai_loop(llm, "m", "sys", [{"role": "user", "content": "q"}], router.visible, 5, Usage())
    )
    assert text == "Готово" and called == [("open_toolbox", {"topic": "limits"})]
    assert "get_limits" not in seen[0] and "get_limits" in seen[1]
    assert {"get_account_context", "open_toolbox"} <= seen[0]


def test_calling_a_hidden_tool_points_to_open_toolbox(tmp_path: Path) -> None:
    router = _router(tmp_path)
    responses = iter(
        [_reply(None, [NS(id="c1", function=NS(name="get_limits", arguments="{}"))]), _reply("ок", None)]
    )
    sent: list[list[dict]] = []  # type: ignore[type-arg]

    async def create(**kwargs: object) -> NS:
        sent.append(kwargs["messages"])  # type: ignore[arg-type]
        return next(responses)

    llm = NS(chat=NS(completions=NS(create=create)))
    asyncio.run(run_openai_loop(llm, "m", "s", [{"role": "user", "content": "q"}], router.visible, 5, Usage()))
    assert "open_toolbox" in sent[1][-1]["content"]


def _reply(content: str | None, calls: list[NS] | None) -> NS:
    message = NS(content=content, tool_calls=calls)
    message.model_dump = lambda **_: {"role": "assistant", "content": content}  # type: ignore[attr-defined]
    return NS(choices=[NS(message=message)], usage=NS(prompt_tokens=10, completion_tokens=5))


@pytest.mark.parametrize(
    ("provider", "routing", "expected"),
    [("custom", "auto", True), ("custom", "off", False), ("claude", "auto", False), ("claude", "on", False)],
)
def test_router_is_used_only_for_openai_compatible_providers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, provider: str, routing: str, expected: bool
) -> None:
    monkeypatch.setenv("AGENT_PROVIDER", provider)
    monkeypatch.setenv("AGENT_TOOL_ROUTING", routing)
    monkeypatch.setenv("AGENT_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    settings = Settings.from_env()

    async def never(_: str) -> bool:
        return False

    agent = FuelCardAgent(settings=settings, opti24=FakeOpti24([]), ask_human=never)  # type: ignore[arg-type]
    assert (agent.router is not None) is expected


def test_invalid_routing_value_gives_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_TOOL_ROUTING", "maybe")
    with pytest.raises(ValueError, match="AGENT_TOOL_ROUTING"):
        Settings.from_env()


def test_unknown_argument_names_are_refused_before_the_tool_runs(tmp_path: Path) -> None:
    from agent.llm_openai import _run_tool

    _, budget, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path, budget_limit=5)
    by_name = {t.name: t for t in tools.values()}
    raw = asyncio.run(_run_tool(by_name, "spending_report", '{"day_from": "2026-08-01", "day_to": "2026-08-31"}'))
    out = json.loads(raw)
    assert out["error"] == "unknown_argument" and "date_from" in out["message"]
    assert budget.spent == 0  # ни одного тарифицируемого запроса на угадывание


def test_non_object_json_arguments_do_not_crash_the_loop(tmp_path: Path) -> None:
    from agent.llm_openai import _run_tool

    _, budget, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path, budget_limit=5)
    by_name = {t.name: t for t in tools.values()}
    for raw_args in ('["2026-08-01"]', "null", "не json"):
        out = json.loads(asyncio.run(_run_tool(by_name, "spending_report", raw_args)))
        assert out["error"] == "bad_arguments"
    assert budget.spent == 0


def test_synonyms_are_mapped_and_standard_names_work(tmp_path: Path) -> None:
    from agent.llm_openai import _run_tool

    _, budget, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path, budget_limit=5)
    by_name = {t.name: t for t in tools.values()}
    mapped = json.loads(asyncio.run(_run_tool(by_name, "spending_report", '{"period_from": "2026-08-01", "period_to": "2026-08-31"}')))
    assert "error" not in mapped and budget.spent == 1  # однозначный синоним принят
    plain = json.loads(asyncio.run(_run_tool(by_name, "spending_report", '{"date_from": "2026-08-01", "date_to": "2026-08-31"}')))
    assert "error" not in plain and budget.spent == 1  # тот же период — операции из кеша, без оплаты  # теперь каждый вызов тратит запрос


def test_refusal_questions_open_the_refusal_topic_without_a_card(tmp_path: Path) -> None:
    for question in ("Почему карта c1 не работает?", "Водителю не дали заправиться по карте c1"):
        found = classify(question)
        assert found.primary is not None and found.primary.id == "refusal", question
