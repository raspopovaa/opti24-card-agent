"""Замер классификатора тем: ключевые слова и ключевые слова + поиск по смыслу.

    AGENT_EMBEDDING_MODEL=<модель> python scripts/check_topic_routing.py [--verbose]

Вопросы: evals/topic_cases.jsonl (перефразировки и разговорные, у каждого — допустимые темы).
Для каждого порога сходства и отрыва — сколько вопросов получили подсказку, сколько из них верно
и сколько неверно. Неверная подсказка уводит модель не к тем инструментам, поэтому порог для
AGENT_TOPIC_MIN_SIMILARITY и AGENT_TOPIC_MIN_MARGIN выбирается так, чтобы смысл не добавил неверных.
Запросов к API Opti24 нет; к модели эмбеддингов — один запрос на темы и по одному на вопрос.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.classifier import classify  # noqa: E402
from agent.rag import embedder_from_env  # noqa: E402
from agent.semantic_topics import Match, SemanticTopics  # noqa: E402

CASES = ROOT / "evals" / "topic_cases.jsonl"
SIMILARITIES = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
MARGINS = (0.0, 0.02, 0.05, 0.08, 0.12)


def load_cases(path: Path = CASES) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tally(rows: list[tuple[dict, str | None, Match | None]], similarity: float | None, margin: float | None) -> tuple[int, int]:
    """(верных подсказок, неверных); similarity=None — только ключевые слова."""
    right = wrong = 0
    for case, keyword_topic, match in rows:
        topic = keyword_topic
        if topic is None and similarity is not None and match is not None:
            if match.similarity >= similarity and match.margin >= (margin or 0.0):
                topic = match.topic.id
        if topic is not None:
            right += topic in case["topics"]
            wrong += topic not in case["topics"]
    return right, wrong


async def collect(semantic: SemanticTopics, cases: list[dict]) -> list[tuple[dict, str | None, Match | None]]:
    rows = []
    for case in cases:
        found = classify(case["q"])
        keyword_topic = found.primary.id if found.confident and found.primary else None
        match = None if keyword_topic else await semantic.match(case["q"])
        rows.append((case, keyword_topic, match))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--verbose", action="store_true", help="показать каждый вопрос")
    args = parser.parse_args(argv)
    embedder = embedder_from_env()
    if embedder is None:
        print("Задайте AGENT_EMBEDDING_MODEL (и ключ AGENT_API_KEY): без модели сравнивать не с чем.", file=sys.stderr)
        return 1
    cases = load_cases()
    rows = asyncio.run(collect(SemanticTopics(embedder), cases))
    right, wrong = tally(rows, None, None)
    print(f"Вопросов: {len(cases)}. Только ключевые слова: подсказка {right + wrong}, верно {right}, неверно {wrong}")
    print("\nСходство  Отрыв  Подсказка  Верно  Неверно")
    best: tuple[int, float, float] | None = None
    for similarity in SIMILARITIES:
        for margin in MARGINS:
            r, w = tally(rows, similarity, margin)
            print(f"{similarity:>8.2f}  {margin:>5.2f}  {r + w:>9}  {r:>5}  {w:>7}")
            if w <= wrong and (best is None or r > best[0]):  # смысл не добавил неверных подсказок
                best = (r, similarity, margin)
    if best:
        print(f"\nБольше всего верных без новых неверных: {best[0]} при сходстве {best[1]:.2f} и отрыве {best[2]:.2f}")
    if args.verbose:
        print()
        for case, keyword_topic, match in rows:
            how = f"слова → {keyword_topic}" if keyword_topic else (
                f"смысл → {match.topic.id} ({match.similarity:.3f}, отрыв {match.margin:.3f})" if match else "—"
            )
            print(f"{'/'.join(case['topics']):28} {how:48} {case['q']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
