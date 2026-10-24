"""Сравнить поиск по базе знаний: только основы слов против гибридного с моделью эмбеддингов.

    AGENT_EMBEDDING_MODEL=<модель> AGENT_BASE_URL=... AGENT_API_KEY=... \\
        python scripts/check_knowledge_search.py

Без сети, если AGENT_EMBEDDING_MODEL=hash. Запросов к API Opti24 нет; к API модели —
один запрос на 32 раздела при первом запуске (векторы сохраняются в data/knowledge.db)
и по одному на каждый вопрос.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.knowledge import load_sections, search  # noqa: E402
from agent.rag import KnowledgeIndex, embedder_from_env, index_path  # noqa: E402

# Вопрос и раздел, который должен быть первым. Вторая половина — разговорные слова,
# которых нет в тексте договора: их и должен находить векторный поиск.
CASES = {
    "как заблокировать потерянную карту": "Как заблокировать или разблокировать карту",
    "какой сервисный сбор": "Сервисный сбор",
    "до какого числа вернуть документы": "Закрывающие документы",
    "водитель посеял карту, отключите её": "Как заблокировать или разблокировать карту",
    "сколько вы берёте сверху за обслуживание": "Сервисный сбор",
    "когда пришлёте бумаги за месяц": "Закрывающие документы",
    "почему на заправке не дали солярку": "Почему по карте могут не отпустить топливо",
    "как закинуть деньги на договор": "Пополнение счёта договора",
}


async def main() -> int:
    embedder = embedder_from_env()
    if embedder is None:
        print("Задайте AGENT_EMBEDDING_MODEL (и AGENT_BASE_URL, AGENT_API_KEY для модели в API).")
        return 2
    sections = load_sections()
    index = KnowledgeIndex(sections, embedder, index_path())
    hits = {"words": 0, "hybrid": 0}
    for question, expected in CASES.items():
        words = search(sections, question)
        hybrid, method = await index.search(question)
        top_words = words[0]["section"] if words else "—"
        top_hybrid = hybrid[0]["section"] if hybrid else "—"
        hits["words"] += top_words == expected
        hits["hybrid"] += top_hybrid == expected
        print(f"{question}\n  слова: {top_words}\n  {method}: {top_hybrid}")
    print(f"\nМодель {embedder.model}: верно по словам {hits['words']}/{len(CASES)}, "
          f"гибридно {hits['hybrid']}/{len(CASES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
