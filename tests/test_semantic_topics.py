"""Тема по смыслу: ближайший пример темы, порог сходства и отрыва, запасной путь без модели."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from agent.classifier import classify, load_matrix
from agent.config import Mode
from agent.core import FuelCardAgent
from agent.semantic_topics import SemanticTopics, semantic_topics_from_env
from tests.test_tool_routing import _router

TOPICS = load_matrix()
IDS = [t.id for t in TOPICS]


def axis(topic_id: str, scale: float = 1.0) -> list[float]:
    return [scale if i == topic_id else 0.0 for i in IDS]


class FakeEmbedder:
    """Тексты тем — оси своих тем; вопросы — заданные векторы."""

    def __init__(self, questions: dict[str, list[float]]) -> None:
        self.questions = questions
        self.calls = 0
        owner = {}
        for topic in TOPICS:
            for text in (f"{topic.title}. {topic.goal}", *topic.examples):
                owner[text] = topic.id
        self.owner = owner

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return [self.questions.get(t) or axis(self.owner[t]) for t in texts]


def mix(a: str, b: str, wa: float, wb: float) -> list[float]:
    return [x + y for x, y in zip(axis(a, wa), axis(b, wb), strict=True)]


def run(coro):
    return asyncio.run(coro)


def test_nearest_topic_is_confident_with_margin() -> None:
    q = "у меня спёрли карту из кабины"
    semantic = SemanticTopics(FakeEmbedder({q: mix("block_card", "refusal", 0.9, 0.2)}),
                              min_similarity=0.55, min_margin=0.05)
    found = run(semantic.classify(q))
    assert found is not None and found.primary.id == "block_card"
    assert found.confident and found.source == "semantic"
    assert set(found.tools) == set(next(t for t in TOPICS if t.id == "block_card").tools)


def test_close_runner_up_gives_no_hint() -> None:
    q = "что-то про карту и отказ"
    semantic = SemanticTopics(FakeEmbedder({q: mix("block_card", "refusal", 0.7, 0.68)}),
                              min_similarity=0.55, min_margin=0.05)
    match = run(semantic.match(q))
    assert match.margin < 0.05
    assert run(semantic.classify(q)) is None


def test_low_similarity_gives_no_hint() -> None:
    q = "какая погода в Москве"
    vector = [0.1] * len(IDS)
    vector[IDS.index("balance")] = 0.3
    semantic = SemanticTopics(FakeEmbedder({q: vector}), min_similarity=0.55, min_margin=0.05)
    assert run(semantic.match(q)).similarity < 0.55
    assert run(semantic.classify(q)) is None


def test_topic_texts_are_embedded_once() -> None:
    embedder = FakeEmbedder({"a": axis("balance"), "b": axis("refusal")})
    semantic = SemanticTopics(embedder, min_similarity=0.5, min_margin=0.05)
    run(semantic.classify("a"))
    run(semantic.classify("b"))
    assert embedder.calls == 3  # темы один раз и по одному вызову на вопрос


def test_thresholds_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_TOPIC_MIN_SIMILARITY", "0.7")
    monkeypatch.setenv("AGENT_TOPIC_MIN_MARGIN", "bad")
    semantic = SemanticTopics(FakeEmbedder({}))
    assert semantic.min_similarity == 0.7 and semantic.min_margin == 0.05


def test_off_by_default_and_on_with_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_EMBEDDING_MODEL", "hash")
    monkeypatch.delenv("AGENT_SEMANTIC_TOPICS", raising=False)
    assert semantic_topics_from_env() is None
    monkeypatch.setenv("AGENT_SEMANTIC_TOPICS", "1")
    assert isinstance(semantic_topics_from_env(), SemanticTopics)
    monkeypatch.setenv("AGENT_EMBEDDING_MODEL", "off")
    assert semantic_topics_from_env() is None


def test_agent_uses_meaning_only_when_keywords_are_unsure() -> None:
    sure, unsure = "Какой баланс договора?", "у меня спёрли карту из кабины"
    assert classify(sure).confident and not classify(unsure).confident
    embedder = FakeEmbedder({unsure: axis("block_card")})
    agent = NS(semantic=SemanticTopics(embedder, min_similarity=0.5, min_margin=0.05))
    assert run(FuelCardAgent._classify(agent, sure)).source == "keywords"
    assert embedder.calls == 0
    assert run(FuelCardAgent._classify(agent, unsure)).primary.id == "block_card"


def test_agent_falls_back_when_embeddings_fail() -> None:
    class Broken:
        async def embed(self, texts: Sequence[str]) -> list[list[float]]:
            raise RuntimeError("модель недоступна")

    question = "у меня спёрли карту из кабины"
    agent = NS(semantic=SemanticTopics(Broken()))
    found = run(FuelCardAgent._classify(agent, question))
    assert found == classify(question)


def test_router_preloads_semantic_topic(tmp_path: Path) -> None:
    q = "у меня спёрли карту из кабины"
    found = run(SemanticTopics(FakeEmbedder({q: axis("block_card")}), min_similarity=0.5, min_margin=0.05).classify(q))
    router = _router(tmp_path, Mode.CONFIRM)
    assert router.preload(q) == set()  # ключевые слова не уверены
    assert "block_cards" in router.preload(q, found)
