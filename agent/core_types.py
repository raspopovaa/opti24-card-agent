"""Общие типы ядра (вынесены отдельно, чтобы модули не импортировали друг друга по кругу)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

# Ответ цикла, когда модель не уложилась в AGENT_MAX_ITERATIONS: по нему учёт видит исход «limit».
STEP_LIMIT_TEXT = "Достигнут предел шагов агента"


class UsageSink(Protocol):
    def add_tokens(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        model: str = "",
        reasoning_tokens: int = 0,
        tools_visible: int | None = None,
        duration_ms: int | None = None,
        finish_reason: str | None = None,
    ) -> None: ...


@dataclass(slots=True)
class Usage:
    """Учёт токенов за сессию — чтобы экономию было видно, а не предполагать."""

    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0
    # Учёт по вопросам в базе (agent.usage.UsageLedger); без него — только итог сессии.
    ledger: Any = None

    def add_tokens(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        model: str = "",
        reasoning_tokens: int = 0,
        tools_visible: int | None = None,
        duration_ms: int | None = None,
        finish_reason: str | None = None,
        input_includes_cache: bool = True,
    ) -> None:
        """input_includes_cache: у OpenAI-совместимых вход уже содержит токены из кеша, у Claude — нет."""
        self.requests += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cache_read_tokens += cache_read_tokens
        self.cache_write_tokens += cache_write_tokens
        if self.ledger is not None:
            prompt = input_tokens if input_includes_cache else input_tokens + cache_read_tokens + cache_write_tokens
            self.ledger.llm_call(
                model=model,
                input_tokens=prompt,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
                reasoning_tokens=reasoning_tokens,
                tools_visible=tools_visible,
                duration_ms=duration_ms,
                finish_reason=finish_reason,
            )

    def add_anthropic(
        self, usage: Any, *, model: str = "", tools_visible: int | None = None, duration_ms: int | None = None,
        finish_reason: str | None = None,
    ) -> None:
        self.add_tokens(
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            model=model,
            tools_visible=tools_visible,
            duration_ms=duration_ms,
            finish_reason=finish_reason,
            input_includes_cache=False,
        )
