"""Цикл агента для OpenAI-совместимых API (OpenAI, DeepSeek, Qwen, YandexGPT, GigaChat, Ollama и др.).

У Claude цикл «модель → инструмент → модель» пишет Tool Runner из SDK Anthropic.
Для остальных провайдеров тот же цикл занимает полсотни строк: это и есть суть
агента, и здесь он виден целиком.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
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


# Синонимы, которые модели чаще всего путают с правильными именами аргументов.
_ALIASES = {"period_from": "date_from", "period_to": "date_to", "start_date": "date_from",
            "end_date": "date_to", "from": "date_from", "to": "date_to", "date_start": "date_from",
            "date_end": "date_to", "card": "card_id", "contract": "contract_id"}


def _schema_names(tool: Any) -> set[str]:
    return set(tool.to_dict()["input_schema"].get("properties", {}))


def _check_arguments(tool: Any, args: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """Нормализует синонимы; возвращает неизвестные имена и аргументы, которые можно передать."""
    known = _schema_names(tool)
    fixed: dict[str, Any] = {}
    unknown: list[str] = []
    for key, value in args.items():
        target = key if key in known else _ALIASES.get(key)
        if target in known and target not in fixed:
            fixed[target] = value
        else:
            unknown.append(key)
    return unknown, fixed


def _hint(tool: Any) -> str:
    expected = sorted(_schema_names(tool))
    return f"Допустимые аргументы: {', '.join(expected)}." if expected else "У инструмента нет аргументов."


async def _run_tool(
    tools: dict[str, Any], name: str, raw_args: str, executed: list[tuple[str, dict[str, Any]]] | None = None
) -> str:
    """Выполняет вызов модели; в ``executed`` попадают только выполненные вызовы с исправленными аргументами."""
    tool = tools.get(name)
    if tool is None:
        return json.dumps(
            {
                "error": "unknown_tool",
                "message": f"Нет инструмента {name}. Если он нужен, открой его группу вызовом open_toolbox.",
            },
            ensure_ascii=False,
        )
    try:
        args = json.loads(raw_args or "{}")
    except json.JSONDecodeError:
        args = None
    if not isinstance(args, dict):
        return json.dumps({"error": "bad_arguments", "message": "Аргументы не JSON-объект; повтори вызов."}, ensure_ascii=False)
    # Неизвестное имя аргумента не должно уйти в инструмент: он не расходует запрос, а пропуск
    # обязательной даты превратился бы в запрос за весь период. Отвечаем именем правильного аргумента.
    unknown, fixed = _check_arguments(tool, args)
    if unknown:
        return json.dumps(
            {"error": "unknown_argument", "message": f"Неизвестные аргументы: {', '.join(unknown)}. "
             + _hint(tool), "expected": sorted(_schema_names(tool))},
            ensure_ascii=False,
        )
    if executed is not None:
        executed.append((name, fixed))  # до await: порядок вызовов сохраняется и при gather
    try:
        result = await tool.call(fixed)
    except Exception as exc:  # noqa: BLE001 — ошибку инструмента отдаём модели, а не роняем цикл
        return json.dumps({"error": type(exc).__name__, "message": str(exc)[:300]}, ensure_ascii=False)
    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    return text[:MAX_TOOL_OUTPUT]


async def run_openai_loop(
    client: Any,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    tools: Sequence[Any] | Callable[[], Sequence[Any]],
    max_iterations: int,
    usage: UsageSink,
) -> tuple[str, list[tuple[str, dict[str, Any]]]]:
    """Возвращает итоговый текст и вызванные инструменты: (имя, аргументы).

    ``tools`` — список или функция, возвращающая список: функция вызывается на каждом шаге,
    поэтому набор инструментов может меняться по ходу вопроса (двухуровневый выбор).
    """
    conversation: list[dict[str, Any]] = [{"role": "system", "content": system}, *messages]
    called: list[tuple[str, dict[str, Any]]] = []
    text = ""

    for _ in range(max_iterations):
        current = tools() if callable(tools) else tools
        by_name = {t.name: t for t in current}
        schemas = to_openai_tools(current)
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
        # В called — только выполненные вызовы с аргументами, которые получил инструмент: отклонённая
        # попытка (скрытый инструмент, неизвестный аргумент) ничего не сделала и не тратила запросы.
        results = await asyncio.gather(
            *(_run_tool(by_name, c.function.name, c.function.arguments, called) for c in calls)
        )
        for call, output in zip(calls, results, strict=True):
            conversation.append({"role": "tool", "tool_call_id": call.id, "content": output})

    return text or "Достигнут предел шагов агента: уточните вопрос.", called
