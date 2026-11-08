"""Тема вопроса по смыслу: второй путь классификатора, когда ключевые слова не уверены.

Вопрос и примеры тем (examples, название и цель из agent/topics.toml) переводятся в векторы той же
моделью эмбеддингов, что и база знаний (AGENT_EMBEDDING_MODEL). Тема — та, чей пример ближе всего
к вопросу. Подсказка даётся, только если сходство не ниже порога и лучшая тема заметно опережает
следующую: неверная уверенная подсказка хуже никакой. Сходство — не вероятность правильности,
поэтому пороги подбираются на размеченных вопросах (scripts/check_topic_routing.py).

Включается переменной AGENT_SEMANTIC_TOPICS=1. Вопрос уходит только к провайдеру эмбеддингов,
которого уже использует база знаний; запросов к API Opti24 нет. Если модель недоступна, остаётся
обычный путь: тему выбирает сама модель через open_toolbox.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass

from .classifier import Classification, Topic, has_param, load_matrix
from .rag.embeddings import Embedder, embedder_from_env

MIN_SIMILARITY = 0.55
MIN_MARGIN = 0.05


@dataclass(frozen=True, slots=True)
class Match:
    topic: Topic
    similarity: float
    margin: float  # насколько лучшая тема ближе следующей


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


class SemanticTopics:
    def __init__(
        self,
        embedder: Embedder,
        topics: tuple[Topic, ...] | None = None,
        *,
        min_similarity: float | None = None,
        min_margin: float | None = None,
    ) -> None:
        self.embedder = embedder
        self.topics = topics or load_matrix()
        self.min_similarity = min_similarity if min_similarity is not None else _env_float(
            "AGENT_TOPIC_MIN_SIMILARITY", MIN_SIMILARITY
        )
        self.min_margin = min_margin if min_margin is not None else _env_float("AGENT_TOPIC_MIN_MARGIN", MIN_MARGIN)
        self._owners: list[Topic] = []
        self._vectors: list[list[float]] = []

    async def _ensure(self) -> None:
        if self._vectors:
            return
        texts, owners = [], []
        for topic in self.topics:
            for text in (f"{topic.title}. {topic.goal}", *topic.examples):
                texts.append(text)
                owners.append(topic)
        self._vectors = await self.embedder.embed(texts)  # один раз за процесс
        self._owners = owners

    async def match(self, question: str) -> Match | None:
        """Ближайшая тема со сходством и отрывом от следующей; None — тем нет."""
        await self._ensure()
        (vector,) = await self.embedder.embed([question])
        best: dict[str, tuple[float, Topic]] = {}
        for owner, example in zip(self._owners, self._vectors, strict=True):
            similarity = _cosine(vector, example)
            if owner.id not in best or similarity > best[owner.id][0]:
                best[owner.id] = (similarity, owner)
        ranked = sorted(best.values(), key=lambda item: item[0], reverse=True)
        if not ranked:
            return None
        runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
        return Match(ranked[0][1], ranked[0][0], ranked[0][0] - runner_up)

    def confident(self, found: Match) -> bool:
        return found.similarity >= self.min_similarity and found.margin >= self.min_margin

    async def classify(self, question: str) -> Classification | None:
        """Уверенная тема по смыслу или None: тогда подсказки нет, как и раньше."""
        found = await self.match(question)
        if found is None or not self.confident(found):
            return None
        missing = tuple(p for p in found.topic.params if p.required and not has_param(p.kind, question))
        return Classification((found.topic,), missing, confident=True, source="semantic")


def semantic_topics_from_env() -> SemanticTopics | None:
    """Классификатор по смыслу, если он включён и модель эмбеддингов настроена."""
    if os.environ.get("AGENT_SEMANTIC_TOPICS", "0").strip().lower() not in {"1", "true", "on", "yes"}:
        return None
    embedder = embedder_from_env()
    return SemanticTopics(embedder) if embedder is not None else None
