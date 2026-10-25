"""Предохранители: подтверждение изменений, бюджет запросов API и журнал аудита.

Модель может ошибиться в выборе карты или суммы лимита, поэтому решение о
любом изменении принимает человек, а не модель. Проверка стоит в коде
инструмента: модель не может её обойти, как бы ни был сформулирован запрос.

Журнал аудита на каждое изменение пишет две записи с общим ``id``: решение
человека до запроса к API и исход после него. По журналу видно, что, над какой
картой или правилом и с каким результатом было сделано.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Awaitable, Callable, Mapping
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

    @property
    def left(self) -> int:
        """Сколько тарифицируемых вызовов ещё можно сделать."""
        return max(0, self.limit - self.spent)

    def spend(self, calls: int = 1) -> None:
        if self.spent + calls > self.limit:
            raise BudgetExceededError(
                f"Исчерпан бюджет тарифицируемых запросов на сессию ({self.limit}). "
                "Сообщи пользователю и предложи продолжить в новой сессии."
            )
        self.spent += calls


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
        """Исход подтверждённого изменения: done, partial, not_done, error или unknown."""
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
            }
        )
        return approval

    def _write(self, record: dict[str, object]) -> None:
        record = {"at": datetime.now(UTC).isoformat(timespec="seconds"), **record}
        self.decisions.append(record)
        with self.audit_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
