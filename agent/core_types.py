"""Общие типы ядра (вынесены отдельно, чтобы модули не импортировали друг друга по кругу)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class UsageSink(Protocol):
    def add_tokens(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> None: ...


@dataclass(slots=True)
class Usage:
    """Учёт токенов за сессию — чтобы экономию было видно, а не предполагать."""

    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0

    def add_tokens(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> None:
        self.requests += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cache_read_tokens += cache_read_tokens
        self.cache_write_tokens += cache_write_tokens

    def add_anthropic(self, usage: Any) -> None:
        self.add_tokens(
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        )
