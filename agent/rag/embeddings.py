"""Эмбеддинги для поиска по базе знаний: OpenAI-совместимый API или локальный хеш.

    AGENT_EMBEDDING_MODEL      модель эмбеддингов; не задана — векторного поиска нет
    AGENT_EMBEDDING_BASE_URL   адрес API, по умолчанию AGENT_BASE_URL, а без него адрес Cloud.ru
                               у провайдера по умолчанию
    AGENT_EMBEDDING_API_KEY    ключ, по умолчанию AGENT_API_KEY

AGENT_EMBEDDING_MODEL=hash включает локальный хеш символьных триграмм: без сети и
модели, ловит общие куски слов («блокир» в «заблокировать»), но не синонимы.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import time
from collections.abc import Sequence
from typing import Any, Protocol
from urllib.parse import urlsplit

from ..providers import DEFAULT_PROVIDER, PROVIDERS
from ..usage import record_embedding

HASH_MODEL = "hash"
HASH_DIM = 512
_WORD = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)


class Embedder(Protocol):
    model: str

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Детерминированный вектор из триграмм символов: для офлайна и тестов."""

    model = HASH_MODEL

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [_hash_vector(text) for text in texts]


class OpenAIEmbedder:
    """POST /embeddings OpenAI-совместимого API (Cloud.ru Foundation Models, OpenAI, vLLM)."""

    batch = 32

    def __init__(self, client: Any, model: str, provider: str | None = None) -> None:
        self._client = client
        self.model = model
        self.provider = provider

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch):
            started = time.monotonic()
            response = await self._client.embeddings.create(
                model=self.model, input=list(texts[start : start + self.batch])
            )
            usage = getattr(response, "usage", None)
            record_embedding(
                self.model, getattr(usage, "prompt_tokens", 0) or 0, int((time.monotonic() - started) * 1000),
                self.provider,
            )
            vectors += [list(item.embedding) for item in sorted(response.data, key=lambda d: d.index)]
        return vectors


def embedder_from_env() -> Embedder | None:
    """Эмбеддер по переменным окружения; None — векторный поиск выключен."""
    model = os.environ.get("AGENT_EMBEDDING_MODEL", "").strip()
    if not model or model.lower() in {"off", "0", "false"}:
        return None
    if model == HASH_MODEL:
        return HashEmbedder()
    base_url = (
        os.environ.get("AGENT_EMBEDDING_BASE_URL") or os.environ.get("AGENT_BASE_URL") or _provider_base_url()
    )
    api_key = os.environ.get("AGENT_EMBEDDING_API_KEY") or os.environ.get("AGENT_API_KEY")
    if not base_url or not api_key:
        return None
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key.strip(), base_url=base_url)
    return OpenAIEmbedder(client, model, provider_for_url(base_url))


def provider_for_url(base_url: str) -> str:
    """Имя провайдера для учёта расхода: из списка провайдеров по адресу, иначе хост адреса."""
    url = base_url.rstrip("/")
    for name, provider in PROVIDERS.items():
        if provider.base_url and provider.base_url.rstrip("/") == url:
            return name
    return urlsplit(url).hostname or url


def _provider_base_url() -> str:
    """Адрес провайдера чата, если его ключ тоже лежит в AGENT_API_KEY (Cloud.ru)."""
    provider = PROVIDERS.get(os.environ.get("AGENT_PROVIDER", DEFAULT_PROVIDER))
    if provider is None or provider.key_env != "AGENT_API_KEY":
        return ""
    return provider.base_url


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def _hash_vector(text: str) -> list[float]:
    vector = [0.0] * HASH_DIM
    for word in _WORD.findall(text.casefold()):
        padded = f" {word} "
        for i in range(len(padded) - 2):
            digest = hashlib.blake2b(padded[i : i + 3].encode(), digest_size=4).digest()
            vector[int.from_bytes(digest, "little") % HASH_DIM] += 1.0
    return vector
