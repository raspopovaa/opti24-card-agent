"""Предохранители: подтверждение изменений, бюджет запросов API и журнал аудита.

Модель может ошибиться в выборе карты или суммы лимита, поэтому решение о
любом изменении принимает человек, а не модель. Проверка стоит в коде
инструмента: модель не может её обойти, как бы ни был сформулирован запрос.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .config import Mode

AskHuman = Callable[[str], Awaitable[bool]]


class BudgetExceededError(RuntimeError):
    pass


@dataclass(slots=True)
class BillingBudget:
    """Сколько тарифицируемых вызовов Opti24 агент ещё может сделать в этой сессии."""

    limit: int
    spent: int = 0

    def spend(self, calls: int = 1) -> None:
        if self.spent + calls > self.limit:
            raise BudgetExceededError(
                f"Исчерпан бюджет тарифицируемых запросов на сессию ({self.limit}). "
                "Сообщи пользователю и предложи продолжить в новой сессии."
            )
        self.spent += calls


@dataclass(slots=True)
class ApprovalGate:
    mode: Mode
    ask: AskHuman
    audit_log: Path
    decisions: list[dict[str, object]] = field(default_factory=list)

    async def approve(self, action: str, summary: str, objects: int) -> bool:
        if self.mode is Mode.READ_ONLY:
            approved = False
        else:
            approved = await self.ask(f"{summary}\nВыполнить? [да/нет]: ")
        self._audit(action, objects, approved, asked=True)
        return approved

    def record_without_asking(self, action: str, objects: int) -> None:
        """Обратимое действие выполнено без вопроса: пишем в журнал, что вопроса не было."""
        self._audit(action, objects, approved=True, asked=False)

    def _audit(self, action: str, objects: int, approved: bool, asked: bool = True) -> None:
        # В журнал не пишем номера карт и ID: только что, сколько и чем закончилось.
        record = {
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "action": action,
            "objects": objects,
            "approved": approved,
            "asked": asked,
        }
        self.decisions.append(record)
        with self.audit_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
