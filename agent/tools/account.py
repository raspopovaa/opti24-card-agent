"""Инструменты агента: Договор, пакет запросов API, финансы и база знаний."""

from __future__ import annotations

from typing import Any

from anthropic import beta_async_tool
from apisdkopti24 import APIError
from apisdkopti24.errors import ContractSelectionError, RequestValidationError

from ..config import Mode
from ..knowledge import search as search_knowledge
from ..money import rub, to_decimal
from ..pricing import TERMS_EDITION, service_fee_for_volume
from ..safety import BudgetExceededError
from .common import _check_dates, _doc_date, _error, _iso, _json, _rub_optional
from .context import ToolContext


def build(ctx: ToolContext) -> dict[str, Any]:
    """Инструменты темы по имени; порядок в списке агента задаёт agent/tools/__init__.py."""
    client = ctx.client
    budget = ctx.budget
    knowledge = ctx.knowledge
    quota_payload = ctx.quota_payload
    gate = ctx.gate
    dictionary_label = ctx.dictionary_label

    @beta_async_tool
    async def get_account_context() -> str:
        """Договор, роль, разрешены ли изменения, пакет и период статистики. Бесплатно.

        Вызывай в начале разговора, чтобы понять, какой договор выбран и можно ли
        что-то менять (changes_enabled). Если договоров несколько, покажи их номера.
        """
        try:
            contract_id = getattr(client, "contract_id", None)
            auth = await client.auth.auth_user(contract_id=contract_id)
            info = (await client.auth.get_info()).data
        except (APIError, RequestValidationError, ContractSelectionError, BudgetExceededError) as exc:
            return _error(exc)
        data = auth.data
        contracts = [
            {"id": c.id, "number": c.number, "cards": c.cards_count}
            for c in data.contracts
        ]
        current = next((c for c in contracts if c["id"] == contract_id), None)
        return _json(
            {
                "contract": current,
                "contracts": contracts,
                "org_name": data.org_name,
                "role": data.role_name,
                "read_only": data.read_only,
                # Режим агента (AGENT_MODE) отдельно от прав аккаунта: изменения возможны, только если
                # агент запущен в confirm и у аккаунта есть право записи.
                "agent_mode": gate.mode.value,
                "changes_enabled": gate.mode is Mode.CONFIRM and not data.read_only,
                "period": {"from": _iso(info.from_), "to": _iso(info.to)},
                **quota_payload(info),
            }
        )

    @beta_async_tool
    async def get_contract_summary() -> str:
        """Баланс договора, расход месяца и карты. Тарифный запрос API (1 шт.).

        «Доступно» — это balanceData.available_amount; лимитные поля договора
        показываются отдельно и могут быть null. Не путай их с остатком денег.
        """
        try:
            budget.spend()
            data = (await client.contracts.get_contract_data()).data
        except (APIError, BudgetExceededError) as exc:
            return _error(exc)
        info, balance, cards = data.contractData, data.balanceData, data.cardsData
        return _json(
            {
                "contract_number": info.contract_number,
                "status": info.contract_status_name,
                "available": rub(to_decimal(balance.available_amount)),
                "own_funds": rub(to_decimal(balance.own_balance)),
                "month_spent": rub(to_decimal(balance.consumption_for_month)),
                "month_liters": balance.consumption_for_month_volume,
                # Объём прошлого месяца задаёт ставку сервисного сбора: отдаём его и ставку по
                # типовой редакции, чтобы цену со сбором считать без загрузки операций прошлого месяца.
                "prev_month_liters": balance.consumption_for_prev_month_volume,
                "service_fee_percent_typical": str(service_fee_for_volume(to_decimal(balance.consumption_for_prev_month_volume)))
                if balance.consumption_for_prev_month_volume is not None
                else None,
                "service_fee_terms": TERMS_EDITION,
                "discount_scheme": info.discount_scheme,
                # Коды договора расшифровываются бесплатными справочниками API.
                "discount_scheme_name": await dictionary_label("DiscountScheme", info.discount_scheme),
                "payment_term": await dictionary_label("PaymentTerm", data.payment_term_id),
                "payment_scheme": await dictionary_label("PaymentScheme", data.payment_scheme_id),
                "cards_total": cards.cards_quantity_all,
                "cards_active": cards.cards_quantity_active,
                # Необязательные лимитные поля: null, а не «0 ₽».
                "limit_available": _rub_optional(info.balance_amount_limiter),
                "accumulated": _rub_optional(info.current_amount_limiter),
                "credit_limit": _rub_optional(info.credit_limit),
            }
        )

    @beta_async_tool
    async def get_finance(
        kind: str, date_from: str | None = None, date_to: str | None = None, limit: int = 10
    ) -> str:
        """Платежи, счета или закрывающие документы.

        kind=payments — тарифный запрос; kind=invoices и kind=documents — бесплатные.
        Для documents нужен период до месяца.
        """
        normalized = {
            "payment": "payments",
            "payments": "payments",
            "invoice": "invoices",
            "invoices": "invoices",
            "document": "documents",
            "documents": "documents",
        }.get((kind or "").strip().lower())
        if normalized is None:
            return _json(
                {"error": "bad_request", "message": "kind: payments, invoices или documents."}
            )
        if date_from and date_to and (problem := _check_dates(date_from, date_to)):
            return _json({"error": "bad_dates", "message": problem})
        if normalized == "documents" and not (date_from and date_to):
            return _json(
                {"error": "bad_dates", "message": "Для документов нужны date_from и date_to."}
            )
        top = max(1, min(int(limit), 50))
        try:
            if normalized == "payments":
                budget.spend()
                response = await client.contracts.get_payments()
                items = list(response.data.result or [])
                if date_from:
                    items = [i for i in items if i.date[:10] >= date_from]
                if date_to:
                    items = [i for i in items if i.date[:10] <= date_to]
                items.sort(key=lambda i: i.date, reverse=True)
                total = len(items)
                items = items[:top]
                payload = [
                    {
                        "id": i.id,
                        "date": i.date,
                        "amount": rub(to_decimal(i.amount)),
                        "amount_client": rub(to_decimal(i.amount_client)),
                        "currency": i.currency,
                        "description": i.description,
                        "payment_name": i.payment_name,
                        "payment_number": i.payment_number,
                    }
                    for i in items
                ]
            elif normalized == "invoices":
                response = await client.contracts.get_invoices()
                items = list(response.data.result or [])
                if date_from:
                    items = [i for i in items if _iso(i.date_start)[:10] >= date_from]
                if date_to:
                    items = [i for i in items if _iso(i.date_start)[:10] <= date_to]
                items.sort(key=lambda i: _iso(i.date_start), reverse=True)
                total = len(items)
                items = items[:top]
                payload = [
                    {
                        "id": i.id,
                        "number": i.ref_number,
                        "date_from": _iso(i.date_start),
                        "date_to": _iso(i.date_end),
                        "amount": rub(to_decimal(i.amount)),
                        "paid_amount": rub(to_decimal(i.paid_amount)),
                        "status": i.status,
                        "comment": i.comment,
                    }
                    for i in items
                ]
            else:
                response = await client.contracts.get_documents(
                    date_start=date_from, date_end=date_to, on_page=top
                )
                items = list(response.data.result or [])
                total = response.data.total_count
                payload = [
                    {
                        "id": i.id,
                        "name": i.name,
                        "number": i.number,
                        "date": _doc_date(i.date),
                        "total": rub(to_decimal(i.total)),
                        "vat": rub(to_decimal(i.vat)),
                        "sum": rub(to_decimal(i.sum)),
                        "currency": i.currency,
                        "consignee": i.consignee,
                    }
                    for i in items
                ]
        except (APIError, BudgetExceededError, RequestValidationError) as exc:
            return _error(exc)
        shown = payload[:top]
        return _json(
            {
                "kind": normalized,
                "period": {"from": date_from, "to": date_to},
                "total": total,
                "shown": len(shown),
                **(
                    {"hint": "Показана часть: сузь период или увеличь limit."}
                    if total > len(shown)
                    else {}
                ),
                "items": shown,
            }
        )

    @beta_async_tool
    async def lookup_knowledge(question: str) -> str:
        """Условия договора топливных карт: цены, скидки, сбор, блокировка, отказ, документы, претензии, API.

        Ищет по выжимкам договора; бесплатно, без запросов к API. Возвращает разделы
        с источником и редакцией.

        Args:
            question: Вопрос клиента или ключевые слова, например "вернуть аванс".
        """
        found = search_knowledge(knowledge, question)
        if not found:
            return _json(
                {
                    "found": 0,
                    "message": "В базе знаний ответа нет: направь к менеджеру по договору или на горячую линию.",
                }
            )
        return _json({"found": len(found), "sections": found})

    @beta_async_tool
    async def get_api_usage() -> str:
        """Месячный пакет запросов API, расход и остаток до блокировки. Бесплатно."""
        try:
            info = (await client.auth.get_info()).data
        except APIError as exc:
            return _error(exc)
        return _json(
            {
                "period": {"from": _iso(info.from_), "to": _iso(info.to)},
                **quota_payload(info),
                "agent_billable_calls_this_session": budget.spent,
                "agent_billable_limit": budget.limit,
            }
        )

    return {
        "get_account_context": get_account_context,
        "get_contract_summary": get_contract_summary,
        "get_finance": get_finance,
        "lookup_knowledge": lookup_knowledge,
        "get_api_usage": get_api_usage,
    }
