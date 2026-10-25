"""Векторный поиск по базе знаний: находит по смыслу, хранит векторы, без модели — поиск по словам."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from agent.config import Mode
from agent.knowledge import Section, load_sections
from agent.rag import HashEmbedder, KnowledgeIndex, embedder_from_env
from tests.test_agent import make_tools
from tests.test_knowledge import QUESTIONS

# Подставная модель: вектор — какие понятия есть в тексте. Синонимы, которых нет в
# основах слов («отключить», «посеял»), модель знает, поиск по словам — нет.
CONCEPTS = {
    "блокировка": ("блокир", "отключ", "посеял", "потерял", "украл"),
    "сбор": ("сбор", "сверху", "комисси", "обслуживан"),
    "погода": ("погод",),
}


class ConceptEmbedder:
    model = "syn-concepts"

    def __init__(self) -> None:
        self.texts: list[str] = []

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.texts += texts
        return [
            [float(any(word in text.casefold() for word in words)) for words in CONCEPTS.values()]
            for text in texts
        ]


class BrokenEmbedder:
    model = "syn-broken"

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise ConnectionError("нет сети")


SECTIONS = [
    Section("cards.md", "Карты", "Как заблокировать карту", "2025", "Заблокировать карту можно в личном кабинете."),
    Section("fee.md", "Цены", "Сервисный сбор", "2025", "Сбор — процент от стоимости покупки."),
]


def _search(index: KnowledgeIndex, question: str) -> tuple[list[str], str]:
    found, method = asyncio.run(index.search(question))
    return [item["section"] for item in found], method


def test_synonym_finds_section_that_words_miss(tmp_path: Path) -> None:
    question = "водитель посеял пластик, отключите"
    without_vectors = KnowledgeIndex(SECTIONS, None, tmp_path / "k.db")
    with_vectors = KnowledgeIndex(SECTIONS, ConceptEmbedder(), tmp_path / "k.db")

    assert _search(without_vectors, question) == ([], "words")
    assert _search(with_vectors, question) == (["Как заблокировать карту"], "hybrid")


def test_question_off_topic_finds_nothing(tmp_path: Path) -> None:
    index = KnowledgeIndex(SECTIONS, ConceptEmbedder(), tmp_path / "k.db")

    assert _search(index, "какая погода в Москве") == ([], "hybrid")


def test_section_vectors_are_computed_once_and_kept_on_disk(tmp_path: Path) -> None:
    first, second = ConceptEmbedder(), ConceptEmbedder()
    _search(KnowledgeIndex(SECTIONS, first, tmp_path / "k.db"), "сколько берёте сверху")
    changed = [SECTIONS[0], Section("fee.md", "Цены", "Сервисный сбор", "2026", "Сбор 3,75 %.")]

    _search(KnowledgeIndex(changed, second, tmp_path / "k.db"), "сколько берёте сверху")

    assert len(first.texts) == 3  # два раздела и вопрос
    assert second.texts == ["Цены. Сервисный сбор\nСбор 3,75 %.", "сколько берёте сверху"]


def test_model_failure_falls_back_to_words(tmp_path: Path) -> None:
    index = KnowledgeIndex(SECTIONS, BrokenEmbedder(), tmp_path / "k.db")

    assert _search(index, "как заблокировать карту") == (["Как заблокировать карту"], "words")


@pytest.mark.parametrize(("question", "section"), QUESTIONS.items())
def test_word_answers_stay_in_results_with_vectors(question: str, section: str) -> None:
    # Даже со слабой моделью раздел, найденный словами, из ответа не пропадает.
    index = KnowledgeIndex(load_sections(), HashEmbedder(), None)

    assert section in _search(index, question)[0]


def test_rrf_puts_section_found_by_words_and_meaning_first(tmp_path: Path) -> None:
    # Векторы ставят блокировку и сбор вровень, слова находят только сбор: он и первый.
    index = KnowledgeIndex(SECTIONS, ConceptEmbedder(), tmp_path / "k.db")

    found, _ = asyncio.run(index.search("сервисный сбор, посеял", fusion="rrf"))

    assert found[0]["section"] == "Сервисный сбор"


def test_threshold_comes_from_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_RAG_MIN_SIMILARITY", "0.99")
    index = KnowledgeIndex(SECTIONS, ConceptEmbedder(), tmp_path / "k.db")

    # «сбор, посеял» похож на оба раздела на 0,71: выше порога 0,99 нет ничего, отвечают слова.
    assert _search(index, "сервисный сбор, посеял") == (["Сервисный сбор"], "hybrid")


def test_vectors_first_keeps_model_order(tmp_path: Path) -> None:
    index = KnowledgeIndex(SECTIONS, ConceptEmbedder(), tmp_path / "k.db")

    found, _ = asyncio.run(index.search("сбор, потерял карту", fusion="vectors_first"))

    assert [item["section"] for item in found] == ["Как заблокировать карту", "Сервисный сбор"]


def test_embedder_is_off_unless_model_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("AGENT_API_KEY", "syn-key")
    assert embedder_from_env() is None

    monkeypatch.setenv("AGENT_EMBEDDING_MODEL", "hash")
    assert isinstance(embedder_from_env(), HashEmbedder)

    monkeypatch.setenv("AGENT_EMBEDDING_MODEL", "syn-embedding-model")
    embedder = embedder_from_env()
    assert embedder is not None and embedder.model == "syn-embedding-model"


def test_lookup_knowledge_says_how_it_searched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_EMBEDDING_MODEL", "hash")
    _, budget, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path)

    found = json.loads(asyncio.run(tools["lookup_knowledge"].call({"question": "какой сервисный сбор"})))

    assert found["search"] == "hybrid" and found["sections"][0]["section"] == "Сервисный сбор"
    assert budget.spent == 0
