"""Инструменты агента — тонкая обёртка над apisdkopti24, разложенная по темам.

Правила, по которым написаны инструменты:

* Инструмент отвечает на один понятный вопрос и возвращает короткий JSON.
  Сырые ответы API (десятки полей) в контекст модели не попадают.
* Список ограничен по размеру (``limit``), а лишнее отсекается кодом.
* Тарифицируемые методы Opti24 списывают бюджет сессии (``BillingBudget``).
* Изменяющий инструмент спрашивает человека через ``ApprovalGate``.
* Ошибка не роняет агента: инструмент возвращает понятный текст ошибки,
  и модель может объяснить её пользователю или попробовать другой путь.
* Платные данные не запрашиваются дважды: операции за период кешируются на
  сессию, и ``spending_report`` с ``find_suspicious_fills`` за тот же период
  стоят как один запрос.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from apisdkopti24 import APIClient

from ..safety import ApprovalGate, BillingBudget
from . import account, cards, operations, reference, reports, rules
from .common import MAX_LIST, TransactionsUnavailableError, load_transactions
from .context import make_context

__all__ = ["MAX_LIST", "TransactionsUnavailableError", "build_tools", "load_transactions"]


def build_tools(
    client: APIClient,
    budget: BillingBudget,
    gate: ApprovalGate,
    reports_dir: Path | None = None,
) -> list[Any]:
    """Собирает инструменты, привязанные к клиенту, бюджету и подтверждению."""
    ctx = make_context(client, budget, gate, reports_dir)
    tools: dict[str, Any] = {}
    for module in (account, cards, rules, operations, reports, reference):
        tools.update(module.build(ctx))
    read_tools = [
        tools["get_account_context"],
        tools["get_contract_summary"],
        tools["list_cards"],
        tools["list_card_groups"],
        tools["get_card_profile"],
        tools["get_limits"],
        tools["get_access_rules"],
        tools["diagnose_fueling"],
        tools["spending_report"],
        tools["list_transactions"],
        tools["get_transaction_detail"],
        tools["find_suspicious_fills"],
        tools["fuel_price_analytics"],
        tools["get_finance"],
        tools["get_report_status"],
        tools["download_fuel_report"],
        tools["lookup_knowledge"],
        tools["lookup_dictionary"],
        tools["get_api_usage"],
    ]
    write_tools = [
        tools["block_cards"],
        tools["set_card_comment"],
        tools["set_limit"],
        tools["set_region_limit"],
        tools["set_restriction"],
        tools["order_fuel_report"],
        tools["remove_limit"],
        tools["remove_region_limit"],
        tools["remove_restriction"],
    ]
    # Изменяющие инструменты всегда в списке: в режиме read_only гейт их отклонит,
    # а модель получит понятный ответ и объяснит пользователю, что режим только чтение.
    return [*read_tools, *write_tools]

