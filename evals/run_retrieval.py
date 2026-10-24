"""Оценка поиска по базе знаний без модели, сети и ключей.

    python evals/run_retrieval.py            # печатает промахи и итоговые метрики
    python evals/run_retrieval.py --top 3

Метрики: доля вопросов, где нужный раздел в первых N результатах (hit@N), и доля
вопросов вне базы, на которые поиск честно ничего не вернул. Код выхода 1, если
метрики ниже порогов: так плохое изменение поиска не пройдёт CI.
Набор вопросов: evals/retrieval_cases.jsonl (`section: null` — ответа в базе нет).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.knowledge import load_sections, search  # noqa: E402

MIN_HIT_AT_3 = 0.9
MIN_UNKNOWN = 1.0


def evaluate(top: int = 3) -> dict[str, object]:
    cases = [json.loads(line) for line in Path(__file__).with_name("retrieval_cases.jsonl").open(encoding="utf-8")]
    sections = load_sections(Path(__file__).resolve().parent.parent / "docs" / "knowledge")
    known = [c for c in cases if c["section"]]
    unknown = [c for c in cases if not c["section"]]
    hit1 = hit_n = 0
    misses: list[str] = []
    for case in known:
        found = [r["section"] for r in search(sections, case["q"], top=top)]
        hit1 += found[:1] == [case["section"]]
        if case["section"] in found:
            hit_n += 1
        else:
            misses.append(f"{case['q']!r}: ждали «{case['section']}», получили {found or 'ничего'}")
    wrong = [c["q"] for c in unknown if search(sections, c["q"], top=top)]
    return {
        "known": len(known),
        "hit_at_1": hit1 / len(known),
        "hit_at_n": hit_n / len(known),
        "unknown": len(unknown),
        "unknown_ok": 1 - len(wrong) / len(unknown),
        "misses": misses,
        "false_answers": wrong,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args()
    result = evaluate(args.top)
    for line in result["misses"]:  # type: ignore[union-attr]
        print("ПРОМАХ", line)
    for question in result["false_answers"]:  # type: ignore[union-attr]
        print("ЛИШНИЙ ОТВЕТ на вопрос вне базы:", question)
    print(
        f"hit@1 {result['hit_at_1']:.0%}, hit@{args.top} {result['hit_at_n']:.0%} "
        f"({result['known']} вопросов); «ответа нет» верно в {result['unknown_ok']:.0%} из {result['unknown']}"
    )
    ok = result["hit_at_n"] >= MIN_HIT_AT_3 and result["unknown_ok"] >= MIN_UNKNOWN  # type: ignore[operator]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
