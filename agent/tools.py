"""Инструменты агента — тонкая обёртка над apisdkopti24.

Правила, по которым написаны инструменты:

* Инструмент отвечает на один понятный вопрос и возвращает короткий JSON.
  Сырые ответы API (десятки полей) в контекст модели не попадают.
* Список ограничен по размеру (``limit``), а лишнее отсекается кодом.
* Тарифицируемые методы Opti24 списывают бюджет сессии (``BillingBudget``).
* Изменяющий инструмент спрашивает человека через ``ApprovalGate``.
* Ошибка не роняет агента: инструмент возвращает понятный текст ошибки,
  и модель может объяснить её пользователю или попробовать другой путь.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from anthropic import beta_async_tool
from apisdkopti24 import APIClient, APIError
from apisdkopti24.errors import RequestValidationError

from .analysis import AnomalyRules, find_anomalies, summarize
from .money import rub, to_decimal
from .config import Mode
from .safety import ApprovalGate, BillingBudget, BudgetExceededError

MAX_LIST = 25


def _json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


class TransactionsUnavailableError(Exception):
    """Стенд не отдаёт операции (на DEMO get_transactions_v2 отвечает HTTP 405)."""


def _error(exc: Exception) -> str:
    # Текст исключений SDK очищен от секретов, но всё равно ограничиваем длину.
    return _json({"error": type(exc).__name__, "message": str(exc)[:300]})


def _check_dates(date_from: str, date_to: str) -> str | None:
    try:
        start, end = date.fromisoformat(date_from), date.fromisoformat(date_to)
    except ValueError:
        return "Даты нужны в формате ГГГГ-ММ-ДД."
    if end < start:
        return "Дата окончания раньше даты начала."
    if (end - start).days > 31:
        return "Период не больше месяца: разбей запрос на части."
    return None


async def load_transactions(
    client: APIClient,
    budget: BillingBudget,
    date_from: str,
    date_to: str,
    card_id: str | None = None,
    max_pages: int | None = None,
) -> tuple[list[Any], int]:
    """Операции за период и их общее число; страница из 100 операций — один платный запрос.

    При ``max_pages`` загрузка останавливается раньше, и тогда длина списка
    меньше общего числа: вызывающий код видит, что данные неполные.
    """
    items: list[Any] = []
    offset = 0
    pages = 0
    total = 0
    while True:
        if max_pages is not None and pages >= max_pages:
            return items, total
        pages += 1
        budget.spend()
        try:
            if card_id:
                page = await client.transactions.get_card_transactions_v2(
                    card_id=card_id,
                    date_from=date_from,
                    date_to=date_to,
                    page_limit=100,
                    page_offset=offset,
                )
            else:
                page = await client.transactions.get_transactions_v2(
                    date_from=date_from, date_to=date_to, page_limit=100, page_offset=offset
                )
        except APIError as exc:
            if exc.http_status_code == 405:
                budget.spent -= 1  # отклонённый запрос не считаем платным
                raise TransactionsUnavailableError(
                    "Сервер ответил HTTP 405: на DEMO операции не отдаются (так отвечает "
                    "сам стенд, это видно в проверках opti24-live-check). Операции доступны "
                    "на рабочем договоре."
                ) from exc
            raise
        batch = page.data.result or []
        items.extend(batch)
        offset += len(batch)
        total = page.data.total_count
        if not batch or offset >= total:
            return items, total


def build_tools(client: APIClient, budget: BillingBudget, gate: ApprovalGate) -> list[Any]:
    """Собирает инструменты, привязанные к клиенту, бюджету и подтверждению."""

    @beta_async_tool
    async def get_contract_summary() -> str:
        """Баланс, кредитный лимит и статус договора. Платный запрос API (1 шт.)."""
        try:
            budget.spend()
            data = (await client.contracts.get_contract_data()).data.contractData
        except (APIError, BudgetExceededError) as exc:
            return _error(exc)
        return _json(
            {
                "contract_number": data.contract_number,
                "status": data.contract_status_name,
                "available": rub(to_decimal(data.balance_amount_limiter)),
                "spent_total": rub(to_decimal(data.current_amount_limiter)),
                "credit_limit": rub(to_decimal(data.credit_limit)),
            }
        )

    @beta_async_tool
    async def list_cards(
        status: str | None = None, group_id: str | None = None, query: str | None = None
    ) -> str:
        """Список топливных карт договора. Бесплатный запрос API.

        Args:
            status: Фильтр по статусу, например "Active" (активна) или "Blocked".
            group_id: Показать только карты этой группы (ID из list_card_groups).
            query: Поиск по номеру карты или комментарию.
        """
        try:
            response = await client.cards.get_cards_v2(
                status=status, group_id=group_id, q=query, onpage=MAX_LIST
            )
        except (APIError, RequestValidationError) as exc:
            return _error(exc)
        cards = response.data.result or []
        return _json(
            {
                "total": response.data.total_count,
                "shown": len(cards),
                "cards": [
                    {
                        "id": c.id,
                        "number": c.number,
                        "status": c.status_name or c.status,
                        "group": c.group_name,
                        "comment": c.comment,
                    }
                    for c in cards
                ],
            }
        )

    @beta_async_tool
    async def list_card_groups() -> str:
        """Группы карт (например, по подразделениям). Бесплатный запрос API."""
        try:
            response = await client.card_groups.get_card_groups()
        except APIError as exc:
            return _error(exc)
        return _json(
            [
                {"id": g.id, "name": g.name, "cards": g.cards_count}
                for g in response.data.result or []
            ]
        )

    @beta_async_tool
    async def get_limits(card_id: str | None = None, group_id: str | None = None) -> str:
        """Продуктовые лимиты по карте или группе карт. Бесплатный запрос API.

        Args:
            card_id: ID карты.
            group_id: ID группы карт. Нужно указать что-то одно из двух.
        """
        try:
            response = await client.limits.get_limits(card_id=card_id, group_id=group_id)
        except (APIError, RequestValidationError) as exc:
            return _error(exc)
        return _json(
            [
                {
                    "id": item.id,
                    "card_id": item.card_id,
                    "group_id": item.group_id,
                    "liters": item.amount.model_dump() if item.amount else None,
                    "rubles": item.sum.model_dump() if item.sum else None,
                    "period_type": item.time.type,
                }
                for item in (response.data.result or [])[:MAX_LIST]
            ]
        )

    @beta_async_tool
    async def spending_report(date_from: str, date_to: str, card_id: str | None = None) -> str:
        """Сводка расходов за период: сумма, литры, цена литра, топ карт.

        Платный запрос API: 1 запрос на каждую страницу из 100 операций.
        Период не больше месяца.

        Args:
            date_from: Начало периода, ГГГГ-ММ-ДД.
            date_to: Конец периода, ГГГГ-ММ-ДД.
            card_id: Ограничить одной картой (необязательно).
        """
        if problem := _check_dates(date_from, date_to):
            return _json({"error": "bad_dates", "message": problem})
        try:
            items, _ = await load_transactions(client, budget, date_from, date_to, card_id)
        except (APIError, BudgetExceededError, RequestValidationError, TransactionsUnavailableError) as exc:
            return _error(exc)
        return _json({"period": f"{date_from}…{date_to}", **summarize(items)})

    @beta_async_tool
    async def find_suspicious_fills(date_from: str, date_to: str, card_id: str | None = None) -> str:
        """Подозрительные заправки за период: дорогие, ночные, крупные, частые.

        Платный запрос API: 1 запрос на каждую страницу из 100 операций.
        Правила проверки заданы в коде, а не придуманы моделью.

        Args:
            date_from: Начало периода, ГГГГ-ММ-ДД.
            date_to: Конец периода, ГГГГ-ММ-ДД.
            card_id: Ограничить одной картой (необязательно).
        """
        if problem := _check_dates(date_from, date_to):
            return _json({"error": "bad_dates", "message": problem})
        try:
            items, _ = await load_transactions(client, budget, date_from, date_to, card_id)
        except (APIError, BudgetExceededError, RequestValidationError, TransactionsUnavailableError) as exc:
            return _error(exc)
        return _json(find_anomalies(items, AnomalyRules()))

    @beta_async_tool
    async def get_api_usage() -> str:
        """Сколько запросов к API уже израсходовано и сколько в пакете. Бесплатно."""
        try:
            info = (await client.auth.get_info()).data
        except APIError as exc:
            return _error(exc)
        return _json(
            {
                "package_queries": info.client_info.Queries,
                "calls_total": info.methods.all,
                "agent_billable_calls_this_session": budget.spent,
                "agent_billable_limit": budget.limit,
            }
        )

    @beta_async_tool
    async def block_cards(card_ids: list[str], reason: str) -> str:
        """Заблокировать топливные карты. ИЗМЕНЯЕТ ДАННЫЕ, нужно подтверждение человека.

        Вызывай только после того, как пользователь явно попросил заблокировать
        конкретные карты. Разблокировка делается отдельным решением человека.

        Args:
            card_ids: ID карт (из list_cards), не больше 10.
            reason: Причина блокировки одной фразой, для журнала.
        """
        if not card_ids or len(card_ids) > 10:
            return _json({"error": "bad_request", "message": "Нужно от 1 до 10 карт."})
        if gate.mode is Mode.READ_ONLY:
            gate._audit("block_cards", len(card_ids), False)
            return _json(
                {
                    "status": "not_allowed",
                    "message": "Агент запущен в режиме только чтения (AGENT_MODE=read_only): "
                    "блокировка отключена настройкой, карты НЕ тронуты, оператор запрос не получал. "
                    "Так и скажи пользователю; включить блокировку можно запуском с AGENT_MODE=confirm.",
                }
            )
        approved = await gate.approve(
            "block_cards",
            f"Заблокировать карт: {len(card_ids)}. Причина: {reason}",
            len(card_ids),
        )
        if not approved:
            return _json({"status": "declined", "message": "Пользователь не подтвердил блокировку: карты не тронуты."})
        try:
            budget.spend()
            await client.cards.block_card(card_ids=card_ids, block=True)
        except (APIError, BudgetExceededError, RequestValidationError) as exc:
            return _error(exc)
        return _json({"status": "blocked", "cards": len(card_ids)})

    read_tools = [
        get_contract_summary,
        list_cards,
        list_card_groups,
        get_limits,
        spending_report,
        find_suspicious_fills,
        get_api_usage,
    ]
    # Изменяющий инструмент всегда в списке: в режиме read_only гейт его отклонит,
    # а модель получит понятный ответ и объяснит пользователю, что режим только чтение.
    return [*read_tools, block_cards]
