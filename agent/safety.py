"""Предохранители: подтверждение изменений, бюджет запросов API и журнал аудита.

Модель может ошибиться в выборе карты или суммы лимита, поэтому решение о
любом изменении принимает человек, а не модель. Проверка стоит в коде
инструмента: модель не может её обойти, как бы ни был сформулирован запрос.

Журнал аудита на каждое изменение пишет две записи с общим ``id``: решение
человека до запроса к API и исход после него. По журналу видно, кто из людей
подтвердил, что, над какой картой или правилом и с каким результатом было сделано.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import uuid
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .config import Mode

AskHuman = Callable[[str], Awaitable[bool]]


class BudgetExceededError(RuntimeError):
    pass


# Счётчик тарифицируемых вызовов текущего инструмента. Инструменты одного шага идут через
# asyncio.gather, у каждой задачи свой контекст, поэтому вызовы не смешиваются.
_tool_spend: ContextVar[list[int] | None] = ContextVar("tool_spend", default=None)


@contextmanager
def count_tool_spend() -> Iterator[list[int]]:
    """Считает тарифицируемые вызовы одного инструмента в текущей задаче: counter[0]."""
    counter = [0]
    token = _tool_spend.set(counter)
    try:
        yield counter
    finally:
        _tool_spend.reset(token)


@dataclass(slots=True)
class BillingBudget:
    """Считает тарифицируемые вызовы Opti24 в сессии; предел необязателен.

    По умолчанию предела нет (``limit=None``). Он появляется только в двух случаях: его задали явно
    (``AGENT_MAX_BILLABLE_CALLS``) или API сообщил остаток месячного пакета: тогда агент не тратит
    больше остатка, чтобы не заблокировать договор для всех интеграций до конца месяца.
    """

    limit: int | None = None
    spent: int = 0

    @property
    def left(self) -> int | None:
        """Сколько тарифицируемых вызовов ещё можно сделать; None — предела нет."""
        return None if self.limit is None else max(0, self.limit - self.spent)

    def spend(self, calls: int = 1) -> None:
        if self.limit is not None and self.spent + calls > self.limit:
            raise BudgetExceededError(
                f"Исчерпан предел тарифицируемых запросов ({self.limit}). "
                "Сообщи пользователю и предложи продолжить в новой сессии."
            )
        self.spent += calls
        counter = _tool_spend.get()
        if counter is not None:
            counter[0] += calls


def mask_card_number(number: object) -> str | None:
    """Последние 4 цифры номера карты для журнала: «…0001»."""
    text = str(number or "").strip()
    return f"…{text[-4:]}" if len(text) >= 4 else None


def _digest(args: Mapping[str, object] | None) -> str | None:
    """Отпечаток аргументов изменения: по нему видно, что выполнено ровно то, что подтвердили."""
    if not args:
        return None
    raw = json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def operator_name() -> str:
    """Кто отвечает на вопросы о подтверждении: AGENT_OPERATOR, иначе пользователь ОС."""
    name = os.environ.get("AGENT_OPERATOR", "").strip()
    if name:
        return name[:64]
    try:
        return getpass.getuser()
    except (KeyError, OSError):
        return "unknown"


@dataclass(frozen=True, slots=True)
class Approval:
    """Решение по одному изменению; истинно, только если человек сказал «да»."""

    id: str
    action: str
    approved: bool

    def __bool__(self) -> bool:
        return self.approved


@dataclass(slots=True)
class ApprovalGate:
    mode: Mode
    ask: AskHuman
    audit_log: Path
    # Где спрашивали человека: terminal, mcp_elicitation, web. Пишется в журнал.
    channel: str = "terminal"
    # Кто из людей отвечает на вопросы: после спорной блокировки по журналу видно, кто подтвердил.
    operator: str = field(default_factory=operator_name)
    decisions: list[dict[str, object]] = field(default_factory=list)

    async def approve(
        self,
        action: str,
        summary: str,
        objects: int,
        *,
        target: Mapping[str, object] | None = None,
        args: Mapping[str, object] | None = None,
    ) -> Approval:
        """Спрашивает человека; в режиме только чтения отказывает без вопроса."""
        if self.mode is Mode.READ_ONLY:
            return self.deny(action, objects, target=target, args=args)
        approved = await self.ask(f"{summary}\nВыполнить? [да/нет]: ")
        return self._decision(action, objects, target, args, (approved, self.channel))

    def deny(
        self,
        action: str,
        objects: int,
        *,
        target: Mapping[str, object] | None = None,
        args: Mapping[str, object] | None = None,
    ) -> Approval:
        """Отказ без вопроса человеку: режим только чтения."""
        return self._decision(action, objects, target, args, (False, "read_only"))

    def outcome(self, approval: Approval, status: str, detail: Mapping[str, object] | None = None) -> None:
        """Исход подтверждённого изменения: done, partial, not_done, error, unknown или demo_not_sent."""
        self._write(
            {"id": approval.id, "event": "outcome", "action": approval.action, "status": status, **(detail or {})}
        )

    def _decision(
        self,
        action: str,
        objects: int,
        target: Mapping[str, object] | None,
        args: Mapping[str, object] | None,
        verdict: tuple[bool, str],
    ) -> Approval:
        """Пишет решение: verdict — (одобрено ли, кто решил)."""
        approved, decided_by = verdict
        digest = _digest(args)
        approval = Approval(id=uuid.uuid4().hex[:12], action=action, approved=approved)
        # Свободный текст (причину от модели, текст комментария) в журнал не пишем: только ID,
        # маски номеров карт и отпечаток аргументов.
        self._write(
            {
                "id": approval.id,
                "event": "decision",
                "action": action,
                "objects": objects,
                **({"target": dict(target)} if target else {}),
                **({"args_sha256": digest} if digest else {}),
                "approved": approved,
                "decided_by": decided_by,
                **({"operator": self.operator} if decided_by != "read_only" else {}),
            }
        )
        return approval

    def _write(self, record: dict[str, object]) -> None:
        record = {"at": datetime.now(UTC).isoformat(timespec="seconds"), **record}
        self.decisions.append(record)
        with self.audit_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
