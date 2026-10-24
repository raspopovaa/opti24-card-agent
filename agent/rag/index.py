"""Векторный индекс выжимок договора в SQLite и гибридный поиск.

Разделов сотни, поэтому отдельная векторная БД не нужна: векторы лежат в SQLite,
поиск — полный перебор по косинусу. Вектор раздела считается один раз и хранится
по хешу текста и модели: изменённый раздел пересчитывается, остальные нет.

Итог — слияние двух списков (reciprocal rank fusion): векторный поиск находит
раздел по смыслу («отключи карту» → блокировка), поиск по основам слов — по
точным терминам («УПД», «претензия»). Если модель эмбеддингов недоступна,
остаётся поиск по основам слов, как раньше.

    lookup_knowledge ──► KnowledgeIndex.search ──► векторы (SQLite) + основы слов
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from array import array
from pathlib import Path
from typing import Any

from ..knowledge import Section, as_result, rank
from .embeddings import Embedder, cosine

DEFAULT_PATH = Path("data/knowledge.db")
MIN_SIMILARITY = 0.45  # ниже — раздел не по теме вопроса
_RRF_K = 60

_SCHEMA = """
CREATE TABLE IF NOT EXISTS vectors (
    model TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (model, text_hash)
);
"""


class KnowledgeIndex:
    def __init__(
        self,
        sections: list[Section],
        embedder: Embedder | None,
        path: Path | None = None,
        min_similarity: float = MIN_SIMILARITY,
    ) -> None:
        self._sections = sections
        self._embedder = embedder
        self._path = path
        self._min_similarity = min_similarity
        self._vectors: list[list[float]] | None = None

    async def search(self, query: str, top: int = 3) -> tuple[list[dict[str, str]], str]:
        """Лучшие разделы и способ поиска: "hybrid" или "words" (без векторов)."""
        by_words = [section for _, section in rank(self._sections, query)]
        by_meaning = await self._by_meaning(query)
        if by_meaning is None:
            return [as_result(section) for section in by_words[:top]], "words"

        scores: dict[int, float] = {}
        for ranked in (by_words, by_meaning):
            for place, section in enumerate(ranked):
                key = id(section)
                scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + place + 1)
        merged = {id(section): section for section in by_words + by_meaning}
        best = sorted(merged, key=lambda key: -scores[key])[:top]
        return [as_result(merged[key]) for key in best], "hybrid"

    async def _by_meaning(self, query: str) -> list[Section] | None:
        if self._embedder is None or not query.strip():
            return None
        try:
            vectors = await self._section_vectors(self._embedder)
            (wanted,) = await self._embedder.embed([query])
        except Exception:  # noqa: BLE001 — сеть или модель недоступны: остаётся поиск по словам
            return None
        similar = [
            (similarity, section)
            for section, vector in zip(self._sections, vectors, strict=True)
            if (similarity := cosine(wanted, vector)) >= self._min_similarity
        ]
        similar.sort(key=lambda pair: -pair[0])
        return [section for _, section in similar]

    async def _section_vectors(self, embedder: Embedder) -> list[list[float]]:
        if self._vectors is not None:
            return self._vectors
        texts = [f"{section.title}. {section.heading}\n{section.text}" for section in self._sections]
        hashes = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
        db = _connect(self._path)
        try:
            known = _load(db, embedder.model, hashes)
            missing = [i for i, text_hash in enumerate(hashes) if text_hash not in known]
            if missing:
                fresh = await embedder.embed([texts[i] for i in missing])
                for i, vector in zip(missing, fresh, strict=True):
                    known[hashes[i]] = vector
                with db:
                    db.executemany(
                        "INSERT OR REPLACE INTO vectors VALUES (?, ?, ?)",
                        [(embedder.model, hashes[i], array("f", known[hashes[i]]).tobytes()) for i in missing],
                    )
        finally:
            db.close()
        self._vectors = [known[text_hash] for text_hash in hashes]
        return self._vectors


def index_path() -> Path | None:
    """Файл векторов из AGENT_KNOWLEDGE_DB (по умолчанию data/knowledge.db); off — в памяти."""
    setting = os.environ.get("AGENT_KNOWLEDGE_DB", str(DEFAULT_PATH)).strip()
    return None if setting.lower() in {"", "off", "0", "false"} else Path(setting).expanduser()


def _connect(path: Path | None) -> sqlite3.Connection:
    if path is None:
        db = sqlite3.connect(":memory:")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(path)
    db.executescript(_SCHEMA)
    return db


def _load(db: sqlite3.Connection, model: str, hashes: list[str]) -> dict[str, list[float]]:
    known: dict[str, list[float]] = {}
    rows: list[Any] = db.execute("SELECT text_hash, vector FROM vectors WHERE model = ?", (model,)).fetchall()
    wanted = set(hashes)
    for text_hash, blob in rows:
        if text_hash in wanted:
            known[text_hash] = array("f", blob).tolist()
    return known
