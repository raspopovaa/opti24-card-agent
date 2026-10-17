"""Цикл агента для OpenAI-совместимых API (DeepSeek, Qwen, Kimi, GLM, Ollama).

У Claude цикл «модель → инструмент → модель» пишет Tool Runner из SDK Anthropic.
Для остальных провайдеров тот же цикл занимает полсотни строк: это и есть суть
агента, и здесь он виден целиком.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any

from .core_types import UsageSink

MAX_TOOL_OUTPUT = 6000  # символов: слишком длинный ответ инструмента обрезаем


def to_openai_tools(tools: Sequence[Any]) -> list[dict[str, Any]]:
    result = []
    for tool in tools:
        spec = tool.to_dict()
        result.append(
            {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": spec["input_schema"],
                },
            }
        )
    return result


def parse_tool_args(raw_args: str | None) -> dict[str, Any]:
    """Аргументы вызова из JSON модели; битый JSON или не-объект — пустой словарь."""
    try:
        parsed = json.loads(raw_args or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def _run_tool(tools: dict[str, Any], name: str, raw_args: str) -> str:
    tool = tools.get(name)
    if tool is None:
        return json.dumps({"error": "unknown_tool", "message": f"Нет инструмента {name}"})
    try:
        args = json.loads(raw_args or "{}")
        result = await tool.call(args)
    except Exception as exc:  # noqa: BLE001 — ошибку инструмента отдаём модели, а не роняем цикл
        return json.dumps({"error": type(exc).__name__, "message": str(exc)[:300]}, ensure_ascii=False)
    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    return text[:MAX_TOOL_OUTPUT]


async def run_openai_loop(
    client: Any,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    tools: Sequence[Any],
    max_iterations: int,
    usage: UsageSink,
) -> tuple[str, list[tuple[str, dict[str, Any]]]]:
    """Возвращает итоговый текст и вызванные инструменты: (имя, аргументы)."""
    by_name = {t.name: t for t in tools}
    schemas = to_openai_tools(tools)
    conversation: list[dict[str, Any]] = [{"role": "system", "content": system}, *messages]
    called: list[tuple[str, dict[str, Any]]] = []
    text = ""

    for _ in range(max_iterations):
        response = await client.chat.completions.create(
            model=model, messages=conversation, tools=schemas, max_tokens=4096
        )
        u = response.usage
        if u is not None:
            usage.add_tokens(
                input_tokens=u.prompt_tokens or 0,
                output_tokens=u.completion_tokens or 0,
                # DeepSeek отдаёт попадания в кеш отдельным полем
                cache_read_tokens=getattr(u, "prompt_cache_hit_tokens", 0) or 0,
            )
        message = response.choices[0].message
        text = message.content or text
        calls = message.tool_calls or []
        if not calls:
            return text, called

        conversation.append(message.model_dump(exclude_none=True))
        results = await asyncio.gather(
            *(_run_tool(by_name, c.function.name, c.function.arguments) for c in calls)
        )
        for call, output in zip(calls, results, strict=True):
            called.append((call.function.name, parse_tool_args(call.function.arguments)))
            conversation.append({"role": "tool", "tool_call_id": call.id, "content": output})

    return text or "Достигнут предел шагов агента: уточните вопрос.", called
