"""Проверка ответа повторным вызовом инструмента.

Если в ответе есть числа, которых нет в данных этого вопроса, проверка сама вызывает нужный
инструмент и сравнивает числа с его свежим ответом. Вызовы берутся из того же вопроса; если
модель ничего не вызывала, — первый инструмент темы вопроса с датами прошлого месяца. Изменяющие
инструменты проверка не вызывает никогда.

Пункт временный, для этапа тестирования: выключается AGENT_VERIFY_ANSWER=off. Каждая повторная
проверка тратит тарифицируемые запросы (операции за период кешируются), поэтому расход виден в ответе.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from .classifier import Topic
from .prompt import previous_month
from .tools.catalog import WRITE_TOOLS


@dataclass(frozen=True, slots=True)
class Plan:
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # почему проверка не может вызвать инструмент


def plan_verification(
    question_calls: list[tuple[str, dict[str, Any]]],
    topic: Topic | None,
    today: date,
    tools: dict[str, Any],
) -> Plan:
    """Какие инструменты вызвать для сверки; изменяющие и неизвестные инструменты пропускаются."""
    if question_calls:
        seen: list[tuple[str, dict[str, Any]]] = []
        for name, call_args in question_calls:
            if name in WRITE_TOOLS or name not in tools:
                continue
            if (name, call_args) not in seen:
                seen.append((name, call_args))
        return Plan(calls=seen)
    if topic is None or not topic.tools:
        return Plan(skipped=["для вопроса нет инструмента, по которому можно проверить ответ"])
    name = topic.tools[0]
    tool = tools.get(name)
    if tool is None or name in WRITE_TOOLS:
        return Plan(skipped=[f"инструмент {name} недоступен для проверки"])
    start, end = previous_month(today)
    args: dict[str, Any] = {}
    for key in tool.to_dict()["input_schema"].get("required", []):
        if key == "date_from":
            args[key] = start.isoformat()
        elif key == "date_to":
            args[key] = end.isoformat()
        else:  # например, карта: её не угадать, и проверка не должна выдумывать
            return Plan(skipped=[f"для {name} нужен параметр {key}: проверка его не выдумывает"])
    return Plan(calls=[(name, args)])
