"""Сравнить способы поиска по базе знаний на модели эмбеддингов.

    AGENT_EMBEDDING_MODEL=<модель> python scripts/check_knowledge_search.py [--verbose]

Вопросы: evals/retrieval_cases.jsonl (25 с ответом и 5 не по теме) и
evals/colloquial_cases.jsonl (15 разговорных). Для каждого способа — сколько
вопросов нашли верный раздел первым и сколько вопросов не по теме верно остались
без ответа. В конце — таблица порога сходства (AGENT_RAG_MIN_SIMILARITY): сколько
верных ответов и молчаний не по теме дал бы каждый порог. Запросов к API Opti24
нет; к API модели — один запрос на 32 раздела при первом запуске (векторы
сохраняются в data/knowledge.db) и по одному на вопрос.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.knowledge import Section, load_sections, rank, search  # noqa: E402
from agent.rag import KnowledgeIndex, assess, embedder_from_env, index_path  # noqa: E402
from agent.rag.index import MIN_SIMILARITY  # noqa: E402

CASE_FILES = ("retrieval_cases.jsonl", "colloquial_cases.jsonl")
METHODS = ("слова", "смысл", "rrf", "векторы первыми")
THRESHOLDS = (0.40, 0.45, 0.50, 0.52, 0.55, 0.60, 0.65, 0.70)
GAPS = (0.0, 0.01, 0.02, 0.03, 0.05)  # CLOSE_GAP; LOW_MARGIN берётся таким же


def load_cases() -> list[dict[str, object]]:
    cases = []
    for name in CASE_FILES:
        with (ROOT / "evals" / name).open(encoding="utf-8") as lines:
            cases += [json.loads(line) for line in lines if line.strip()]
    return cases


Scored = list[tuple[float, Section]]


async def main() -> int:
    embedder = embedder_from_env()
    if embedder is None:
        print("Задайте AGENT_EMBEDDING_MODEL (и AGENT_BASE_URL, AGENT_API_KEY для модели в API).")
        return 2
    sections = load_sections()
    index = KnowledgeIndex(sections, embedder, index_path())
    cases = load_cases()
    hits = dict.fromkeys(METHODS, 0)
    silent = dict.fromkeys(METHODS, 0)
    # Для таблицы порога: разделы по косинусу, первый раздел по словам и что считается верным.
    scored_cases: list[tuple[list[tuple[float, str]], str | None, set[object] | None]] = []
    # Для таблицы уточнений: сходства с разделами, первый раздел по словам, верен ли ответ.
    doubt_cases: list[tuple[Scored, Section | None, bool]] = []

    for case in cases:
        question = str(case["q"])
        expected = case["section"]
        accepted = set(expected) if isinstance(expected, list) else {expected}
        found = await search_all(index, sections, question)
        scored = await index.similarities(question) or []
        word_top = found["слова"][0]["section"] if found["слова"] else None
        scored_cases.append(([(sim, sec.heading) for sim, sec in scored], word_top, accepted if expected else None))
        if expected:
            first = found["векторы первыми"]
            ranked_words = rank(sections, question)
            first_ok = (first[0]["section"] if first else None) in accepted
            doubt_cases.append((scored, ranked_words[0][1] if ranked_words else None, first_ok))
        tops = {method: (items[0]["section"] if items else None) for method, items in found.items()}
        correct = {method: (top in accepted if expected else top is None) for method, top in tops.items()}
        for method, ok in correct.items():
            (hits if expected else silent)[method] += ok
        if "--verbose" in sys.argv or not all(correct.values()):
            print(f"{question} — ждали {expected or 'ничего'}")
            for method, top in tops.items():
                print(f"  {method}: {top or '—'}")

    known = sum(1 for case in cases if case["section"])
    unknown = len(cases) - known
    print(f"\nМодель {embedder.model}, {known} вопросов с ответом и {unknown} не по теме:")
    for method in METHODS:
        print(f"  {method}: верно {hits[method]}/{known}, не по теме без ответа {silent[method]}/{unknown}")
    print_thresholds(scored_cases, known, unknown)
    print_gaps(doubt_cases)
    return 0


async def search_all(index: KnowledgeIndex, sections: list[Section], question: str) -> dict[str, list]:
    """Разделы, найденные каждым способом из METHODS."""
    rrf, _ = await index.search(question, fusion="rrf")
    first, _ = await index.search(question, fusion="vectors_first")
    return {
        "слова": search(sections, question),
        "смысл": await index.by_meaning(question) or [],
        "rrf": rrf,
        "векторы первыми": first,
    }


def print_thresholds(
    scored_cases: list[tuple[list[tuple[float, str]], str | None, set[object] | None]], known: int, unknown: int
) -> None:
    """Сколько верных ответов и молчаний не по теме дал бы каждый порог из THRESHOLDS."""
    print("\nПорог сходства для «векторы первыми» (сейчас AGENT_RAG_MIN_SIMILARITY или 0.52):")
    for threshold in THRESHOLDS:
        right = quiet = 0
        for scored, word_top, accepted in scored_cases:
            above = [heading for sim, heading in scored if sim >= threshold]
            top = above[0] if above else word_top
            if accepted is None:
                quiet += top is None
            else:
                right += top in accepted
        print(f"  {threshold:.2f}: верно {right}/{known}, не по теме без ответа {quiet}/{unknown}")


def print_gaps(doubt_cases: list[tuple[Scored, Section | None, bool]]) -> None:
    """Сколько ошибок и верных ответов пометил бы сомнительными каждый зазор из GAPS."""
    wrong_total = sum(not ok for _, _, ok in doubt_cases)
    print(f"\nУточнение при сомнении (порог {MIN_SIMILARITY}; ошибок «векторы первыми» {wrong_total}):")
    for gap in GAPS:
        caught = asked = 0
        for scored, word_top, ok in doubt_cases:
            if assess(scored, word_top, MIN_SIMILARITY, close_gap=gap, low_margin=gap):
                caught += not ok
                asked += ok
        print(f"  зазор {gap:.2f}: ошибок переспрошено {caught}/{wrong_total}, верных переспрошено зря {asked}")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
