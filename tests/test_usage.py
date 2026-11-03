"""Учёт расхода по вопросам: что агент пишет в data/usage.db и как это читает отчёт."""

from __future__ import annotations

import asyncio
import sqlite3
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import pytest

from agent import usage
from agent.config import Settings
from agent.core import FuelCardAgent
from agent.core_types import Usage
from agent.db.store import contract_key_for
from tests.test_agent import FakeOpti24


def _reply(content: str | None, calls: list[NS] | None, prompt: int, completion: int, cached: int = 0) -> NS:
    message = NS(content=content, tool_calls=calls)
    message.model_dump = lambda **_: {"role": "assistant", "content": content}  # type: ignore[attr-defined]
    stats = NS(prompt_tokens=prompt, completion_tokens=completion, prompt_cache_hit_tokens=cached)
    return NS(choices=[NS(message=message, finish_reason="stop" if content else "tool_calls")], usage=stats)


def _agent(responses: list[NS]) -> FuelCardAgent:
    replies = iter(responses)

    async def create(**_: object) -> NS:
        return next(replies)

    async def never(_: str) -> bool:
        return False

    fake: Any = FakeOpti24([])
    agent = FuelCardAgent(settings=Settings.from_env(), opti24=fake, ask_human=never)
    agent.llm = NS(chat=NS(completions=NS(create=create)))
    return agent


def _rows(path: Path, sql: str) -> list[tuple[Any, ...]]:
    with sqlite3.connect(path) as db:
        return db.execute(sql).fetchall()


def test_question_writes_turn_model_steps_and_tool_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_TOOL_ROUTING", "off")  # модель видит все инструменты, как Claude
    summary = NS(id="t1", function=NS(name="get_contract_summary", arguments="{}"))
    first = _reply(None, [summary], 1000, 40, cached=200)
    agent = _agent([first, _reply("Доступно 123 139,19 ₽.", None, 1500, 60)])
    asyncio.run(agent.ask("Сколько денег осталось на договоре?"))

    path = tmp_path / "usage.db"  # AGENT_USAGE_DB из conftest
    (turn,) = _rows(
        path,
        "SELECT topic, steps, tool_count, billable_calls, input_tokens, output_tokens, cache_read_tokens, "
        "question_chars, outcome, unverified_numbers FROM turns",
    )
    assert turn == ("balance", 2, 1, 1, 2500, 100, 200, len("Сколько денег осталось на договоре?"), "answered", 0)
    calls = _rows(path, "SELECT step, purpose, provider, input_tokens, finish_reason, tools_visible FROM llm_calls")
    assert [c[:5] for c in calls] == [
        (1, "answer", "cloudru", 1000, "tool_calls"),
        (2, "answer", "cloudru", 1500, "stop"),
    ]
    assert all(c[5] and c[5] > 0 for c in calls)
    assert _rows(path, "SELECT tool, billable_calls, error_reason FROM tool_calls") == [
        ("get_contract_summary", 1, None)
    ]
    (session,) = _rows(path, "SELECT agent_id, interface, mode, contract_key FROM sessions")
    assert session == ("opti24-card-agent", "cli", "read_only", usage_contract_key())


def usage_contract_key() -> str:
    return contract_key_for("c1")


def test_no_question_text_is_stored(tmp_path: Path) -> None:
    agent = _agent([_reply("Готово.", None, 10, 5)])
    asyncio.run(agent.ask("Карта 7826010000000001 водителя Иванова"))
    with sqlite3.connect(tmp_path / "usage.db") as db:
        dump = "\n".join(db.iterdump())
    assert "7826010000000001" not in dump and "Иванова" not in dump


def test_limit_and_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    call = NS(id="t", function=NS(name="get_api_usage", arguments="{}"))
    agent = _agent([_reply(None, [call], 10, 5) for _ in range(8)])
    answer = asyncio.run(agent.ask("Сколько запросов осталось в пакете?"))
    assert answer.startswith("Достигнут предел шагов")
    assert _rows(tmp_path / "usage.db", "SELECT outcome, steps FROM turns") == [("limit", 8)]

    monkeypatch.setenv("AGENT_USAGE_DB", "off")
    assert Settings.from_env().usage_db is None
    assert _agent([]).usage.ledger is None


def test_write_failure_does_not_break_answer(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    agent = _agent([_reply("Готово.", None, 10, 5), _reply("Ещё.", None, 10, 5)])
    blocker = tmp_path / "file"
    blocker.write_text("")
    agent.usage.ledger.path = blocker / "usage.db"  # каталог — это файл: записать нельзя
    assert asyncio.run(agent.ask("Пакет запросов?")) == "Готово."
    assert asyncio.run(agent.ask("Ещё?")) == "Ещё."
    assert capsys.readouterr().err.count("Учёт расхода не записан") == 1


def test_claude_input_is_normalized_to_include_cache(tmp_path: Path) -> None:
    ledger = usage.UsageLedger(tmp_path / "u.db", provider="claude", mode="read_only", interface="cli", prompt="p")
    sink = Usage(ledger=ledger)
    ledger.begin_turn(topic=None, section=None, confident=False, question="q")
    stats = NS(input_tokens=100, output_tokens=10, cache_read_input_tokens=900, cache_creation_input_tokens=0)
    sink.add_anthropic(stats)
    ledger.end_turn(answer="a", outcome="answered")
    assert _rows(tmp_path / "u.db", "SELECT input_tokens, cache_read_tokens FROM llm_calls") == [(1000, 900)]
    assert sink.input_tokens == 100  # итог сессии в терминале остаётся как был


def test_embeddings_go_to_current_question(tmp_path: Path) -> None:
    ledger = usage.UsageLedger(tmp_path / "u.db", provider="cloudru", mode="read_only", interface="cli", prompt="p")
    usage.record_embedding("bge-m3", 7, 3)  # вне вопроса — никуда
    ledger.begin_turn(topic="contract_terms", section="reference", confident=True, question="q")
    usage.record_embedding("BAAI/bge-m3", 12, 5)
    ledger.end_turn(answer="a", outcome="answered")
    assert _rows(tmp_path / "u.db", "SELECT purpose, model, input_tokens FROM llm_calls") == [
        ("embedding", "BAAI/bge-m3", 12)
    ]
    assert _rows(tmp_path / "u.db", "SELECT steps FROM turns") == [(0,)]


def test_report_groups_and_prices_added_later(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "u.db"
    ledger = usage.UsageLedger(path, provider="cloudru", mode="read_only", interface="cli", prompt="p")
    for topic in ("balance", "balance", "spending"):
        ledger.begin_turn(topic=topic, section="money", confident=True, question="q")
        ledger.llm_call(model="m", input_tokens=1_000_000, output_tokens=100_000, cache_read_tokens=0)
        ledger.end_turn(answer="a", outcome="answered")
    month = ledger.started_at[:7]

    assert usage.main(["--db", str(path), "report", "--month", month]) == 0
    assert "нет цены" in capsys.readouterr().out  # без цены рубли не выдумываются

    assert usage.main([
        "--db", str(path), "price", "--provider", "cloudru", "--model", "m", "--from", "2000-01-01",
        "--input", "10", "--output", "40",
    ]) == 0
    with sqlite3.connect(path) as db:
        rows = usage.report(db, month)
    assert [(r["group"], r["questions"], r["input_tokens"], r["cost_rub"]) for r in rows] == [
        ("balance", 2, 2_000_000, "28.00"),  # 2 × (1 млн × 10 ₽ + 0,1 млн × 40 ₽) / 1 млн
        ("spending", 1, 1_000_000, "14.00"),
    ]


def test_call_cost_uses_cache_price() -> None:
    call = usage.LlmCall(1, "answer", "m", 1_000_000, 0, 800_000, 0, 0, None, None, None)
    price = usage.Price(Decimal("10"), Decimal("40"), Decimal("1"))
    assert usage.call_cost(call, price) == Decimal("2.8000")  # 0,2 млн × 10 + 0,8 млн × 1
    assert usage.call_cost(call, None) is None
