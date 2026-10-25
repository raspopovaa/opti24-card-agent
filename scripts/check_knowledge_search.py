"""Сравнить способы поиска по базе знаний на модели эмбеддингов.

    AGENT_EMBEDDING_MODEL=<модель> python scripts/check_knowledge_search.py [--verbose]

Вопросы: evals/retrieval_cases.jsonl (25 с ответом и 5 не по теме) и
evals/colloquial_cases.jsonl (15 разговорных). Для каждого способа — сколько
вопросов нашли верный раздел первым и сколько вопросов не по теме верно остались
без ответа. Запросов к API Opti24 нет; к API модели — один запрос на 32 раздела
при первом запуске (векторы сохраняются в data/knowledge.db) и по одному на вопрос.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.knowledge import load_sections, search  # noqa: E402
from agent.rag import KnowledgeIndex, embedder_from_env, index_path  # noqa: E402

CASE_FILES = ("retrieval_cases.jsonl", "colloquial_cases.jsonl")
METHODS = ("слова", "смысл", "rrf", "векторы первыми")


def load_cases() -> list[dict[str, object]]:
    cases = []
    for name in CASE_FILES:
        with (ROOT / "evals" / name).open(encoding="utf-8") as lines:
            cases += [json.loads(line) for line in lines if line.strip()]
    return cases


async def main() -> int:
    embedder = embedder_from_env()
    if embedder is None:
        print("Задайте AGENT_EMBEDDING_MODEL (и AGENT_BASE_URL, AGENT_API_KEY для модели в API).")
        return 2
    verbose = "--verbose" in sys.argv
    sections = load_sections()
    index = KnowledgeIndex(sections, embedder, index_path())
    cases = load_cases()
    known = [case for case in cases if case["section"]]
    hits = dict.fromkeys(METHODS, 0)
    silent = dict.fromkeys(METHODS, 0)

    for case in cases:
        question = str(case["q"])
        expected = case["section"]
        accepted = set(expected) if isinstance(expected, list) else {expected}
        rrf, _ = await index.search(question, fusion="rrf")
        first, _ = await index.search(question, fusion="vectors_first")
        found = {
            "слова": search(sections, question),
            "смысл": await index.by_meaning(question) or [],
            "rrf": rrf,
            "векторы первыми": first,
        }
        tops = {method: (items[0]["section"] if items else None) for method, items in found.items()}
        right = {method: (top in accepted if expected else top is None) for method, top in tops.items()}
        for method, ok in right.items():
            if expected:
                hits[method] += ok
            else:
                silent[method] += ok
        if verbose or not all(right.values()):
            print(f"{question} — ждали {expected or 'ничего'}")
            for method, top in tops.items():
                print(f"  {method}: {top or '—'}")

    unknown = len(cases) - len(known)
    print(f"\nМодель {embedder.model}, {len(known)} вопросов с ответом и {unknown} не по теме:")
    for method in METHODS:
        print(f"  {method}: верно {hits[method]}/{len(known)}, не по теме без ответа {silent[method]}/{unknown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
