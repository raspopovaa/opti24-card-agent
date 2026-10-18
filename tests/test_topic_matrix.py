"""Матрица тем: классификация, связь с инструментами и SDK, подсказка, журнал, документ."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from apisdkopti24.endpoints import OPERATION_METADATA_BY_NAME

from agent.classifier import DETECTORS, PARAM_TITLES, classify, has_param, hint, load_matrix, log_question
from agent.config import Mode, Settings
from agent.core import FuelCardAgent
from agent.tools.reference import DICTIONARIES
from scripts.generate_topic_matrix import main as generate_doc
from scripts.generate_topic_matrix import response_fields, tool_operations
from tests.test_agent import FakeOpti24, make_tools

TOPICS = load_matrix()


def test_topic_ids_are_unique_and_titled() -> None:
    ids = [t.id for t in TOPICS]
    assert len(ids) == len(set(ids))
    assert all(t.title and t.goal and t.examples and t.keywords for t in TOPICS)


@pytest.mark.parametrize(("topic_id", "example"), [(t.id, e) for t in TOPICS for e in t.examples])
def test_every_example_is_classified_into_its_topic(topic_id: str, example: str) -> None:
    found = classify(example).primary
    assert found is not None and found.id == topic_id


def test_topics_point_to_real_tools(tmp_path: Path) -> None:
    _, _, tools = make_tools(Mode.CONFIRM, answer=False, tmp=tmp_path)
    for topic in TOPICS:
        assert set(topic.tools) <= set(tools), topic.id
        if topic.supported:
            assert topic.tools or topic.id == "contract_terms", topic.id
        else:
            assert not topic.tools and topic.reply, topic.id


@pytest.mark.parametrize("ref", sorted({ref for t in TOPICS for ref in t.answer_fields}))
def test_answer_fields_exist_in_sdk_response_models(ref: str) -> None:
    operation, path = ref.split(":", 1)
    assert operation in OPERATION_METADATA_BY_NAME
    assert path in response_fields(operation), ref


def test_tools_call_only_real_sdk_operations() -> None:
    operations = tool_operations()
    assert {"get_contract_summary", "diagnose_fueling", "lookup_dictionary"} <= operations.keys()
    for tool, ops in operations.items():
        assert set(ops) <= OPERATION_METADATA_BY_NAME.keys(), tool
    assert "get_dictionary" in operations["get_contract_summary"]  # схема скидки и условия оплаты


def test_dictionaries_and_params_are_known() -> None:
    for topic in TOPICS:
        assert set(topic.dictionaries) <= DICTIONARIES.keys(), topic.id
        for param in topic.params:
            assert param.kind in DETECTORS and param.kind in PARAM_TITLES, (topic.id, param.kind)
            if param.required:
                assert param.default or param.ask, (topic.id, param.kind)


@pytest.mark.parametrize(
    ("kind", "question", "present"),
    [
        ("period", "Сколько потратили в сентябре 2026?", True),
        ("period", "Сколько потратили в прошлом месяце?", True),
        ("period", "Покажи расходы", False),
        ("card", "Заблокируй карту 7826010000000001", True),
        ("card", "Какие лимиты на карте Газели Иванова?", True),
        ("card", "Почему карта не работает?", False),
        ("decline_code", "в чеке код ответа 51", True),
        ("station", "отказали на АЗС 137", True),
        ("liters", "ДТ 50 литров", True),
        ("product", "ДТ 50 литров", True),
        ("transaction_id", "Покажи детали операции 123", True),
        ("reason", "водитель уволился", True),
    ],
)
def test_parameter_detectors(kind: str, question: str, present: bool) -> None:
    assert has_param(kind, question) is present


def test_hint_asks_for_missing_card_and_defaults_the_period() -> None:
    refusal = hint(classify("Почему карта не работает?"))
    assert "тема «Отказ на АЗС»" in refusal and "сначала спроси" in refusal and "diagnose_fueling" in refusal
    assert "сначала спроси" not in hint(classify("Почему карта 7826010000000001 не работает?"))
    spending = hint(classify("Покажи расходы"))
    assert "возьми текущий месяц" in spending and "сначала спроси" not in spending


def test_hint_for_unsupported_and_unknown_questions() -> None:
    assert "вне возможностей агента" in hint(classify("Переведи 5000 рублей на карту"))
    assert hint(classify("Расскажи анекдот")) == ""


def test_question_log_is_opt_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_TOPIC_LOG", raising=False)
    log_question("Покажи расходы", classify("Покажи расходы"))
    assert not list(tmp_path.iterdir())
    target = tmp_path / "topics.jsonl"
    monkeypatch.setenv("AGENT_TOPIC_LOG", str(target))
    log_question("Расскажи анекдот", classify("Расскажи анекдот"))
    record = json.loads(target.read_text(encoding="utf-8"))
    assert record["question"] == "Расскажи анекдот" and record["topics"] == []  # повод пополнить матрицу


def test_agent_sends_hint_with_the_question_but_keeps_history_clean(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AGENT_PROVIDER", "custom")
    monkeypatch.setenv("AGENT_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    monkeypatch.delenv("AGENT_TOPIC_LOG", raising=False)
    sent: list[list[dict[str, str]]] = []

    async def create(**kwargs: object) -> NS:
        sent.append(kwargs["messages"])  # type: ignore[arg-type]
        message = NS(content="Ответ", tool_calls=None)
        return NS(choices=[NS(message=message)], usage=None)

    async def never(_: str) -> bool:
        return False

    agent = FuelCardAgent(settings=Settings.from_env(), opti24=FakeOpti24([]), ask_human=never)  # type: ignore[arg-type]
    agent.llm = NS(chat=NS(completions=NS(create=create)))
    import asyncio

    asyncio.run(agent.ask("Покажи расходы"))
    assert "[Подсказка: тема «Расходы»" in sent[0][-1]["content"]
    assert all("[Подсказка" not in m["content"] for m in agent.history)
    assert {"spending_report", "open_toolbox"} <= {t.name for t in agent.router.visible()}  # type: ignore[union-attr]


def test_topic_matrix_document_is_up_to_date() -> None:
    assert generate_doc(["--check"]) == 0


def test_paraphrases_never_get_a_confident_wrong_topic() -> None:
    """Неверная подсказка хуже никакой: на перефразировках тема либо верна, либо не уверена.

    Доля «верно» здесь не проверяется: часть разговорных слов матрицы подобрана по этим же
    вопросам, поэтому честную долю покажет только журнал реальных вопросов (AGENT_TOPIC_LOG).
    """
    rows = [json.loads(line) for line in (Path(__file__).parent.parent / "evals" / "paraphrases.jsonl").open()]
    assert len(rows) >= 20 and {row["topic"] for row in rows} <= {t.id for t in TOPICS}
    for row in rows:
        found = classify(row["q"])
        if found.confident:
            assert found.primary is not None and found.primary.id == row["topic"], row["q"]


def test_weak_or_tied_matches_give_no_hint_and_open_nothing(tmp_path: Path) -> None:
    from agent.tools.router import ToolRouter

    # «заблокируют» (блокировка карт) и «запросы» (пакет API) дают ничью: это не уверенность.
    tied = classify("Нас не заблокируют за частые запросы?")
    assert not tied.confident and hint(tied) == ""
    _, _, tools = make_tools(Mode.CONFIRM, answer=False, tmp=tmp_path)
    router = ToolRouter(list(tools.values()), Mode.CONFIRM)
    assert router.preload("Нас не заблокируют за частые запросы?") == set()
    assert "block_cards" not in {t.name for t in router.visible()}


def test_open_toolbox_opens_a_whole_section_and_tells_what_to_ask(tmp_path: Path) -> None:
    import asyncio

    from agent.tools.router import ToolRouter

    _, _, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path)
    router = ToolRouter(list(tools.values()), Mode.READ_ONLY)
    section = json.loads(asyncio.run(router.open_toolbox.call({"topic": "spending"})))
    assert section["title"] == "Расходы и цены" and "fuel_price_analytics" in section["tools"]
    assert {"fuel_price", "suspicious"} <= {t["id"] for t in section["topics"]}
    refusal = json.loads(asyncio.run(router.open_toolbox.call({"topic": "refusal"})))
    assert refusal["tools"] == ["diagnose_fueling"]
    assert refusal["ask_if_missing"][0]["param"] == "карта"
    outside = json.loads(asyncio.run(router.open_toolbox.call({"topic": "money_transfer"})))
    assert outside["tools"] == [] and "личном кабинете" in outside["reply"]


def test_prompt_contains_the_topic_catalog() -> None:
    from agent.prompt import SYSTEM_PROMPT

    assert "{catalog}" not in SYSTEM_PROMPT
    for topic in TOPICS:
        assert f"{topic.id} — {topic.title}" in SYSTEM_PROMPT
