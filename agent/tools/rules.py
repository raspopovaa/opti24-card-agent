"""Инструменты агента: Правила заправки: продуктовые лимиты, региональные лимиты, товарные ограничители."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal, cast

from anthropic import beta_async_tool
from apisdkopti24 import APIError
from apisdkopti24.errors import RequestValidationError
from apisdkopti24.models.limits import (
    LimitAmountRequest,
    LimitRequestItem,
    LimitSumRequest,
    LimitTimeRequest,
    LimitTransactionsRequest,
)
from apisdkopti24.models.region_limits import RegionLimitRequestItem
from apisdkopti24.models.restrictions import RestrictionRequestItem

from ..safety import BudgetExceededError
from .common import MAX_LIST, _error, _json
from .context import ToolContext


def build(ctx: ToolContext) -> dict[str, Any]:
    """Инструменты темы по имени; порядок в списке агента задаёт agent/tools/__init__.py."""
    client = ctx.client
    budget = ctx.budget
    limits_payload = ctx.limits_payload
    confirm_write = ctx.confirm_write
    default_group_id = ctx.default_group_id

    def with_default_group(card_id: str | None, group_id: str | None) -> tuple[str | None, dict[str, str]]:
        """Группа по умолчанию, если не указаны ни карта, ни группа (только для чтения правил)."""
        if card_id or group_id or not default_group_id:
            return group_id, {}
        return default_group_id, {"group_id": f"{default_group_id} (OPTI24_DEFAULT_GROUP_ID)"}

    @beta_async_tool
    async def get_limits(card_id: str | None = None, group_id: str | None = None) -> str:
        """Продуктовые лимиты: продукт, период, установлено и израсходовано. Бесплатно.

        Без card_id и group_id возвращает лимиты всего договора. Если указаны оба,
        это ошибка: выбери что-то одно.
        """
        if card_id and group_id:
            return _json(
                {"error": "bad_request", "message": "Укажи либо card_id, либо group_id (не оба)."}
            )
        group_id, defaults = with_default_group(card_id, group_id)
        try:
            response = await client.limits.get_limits(card_id=card_id, group_id=group_id)
        except (APIError, RequestValidationError) as exc:
            return _error(exc)
        items = response.data.result or []
        total = response.data.total_count
        shown = items[:MAX_LIST]
        hint = (
            {"hint": "Показана часть лимитов: уточни card_id или group_id."}
            if total and total > len(shown)
            else {}
        )
        return _json(
            {
                "total": total,
                "shown": len(shown),
                **hint,
                **({"defaults": defaults} if defaults else {}),
                "limits": await limits_payload(shown),
            }
        )

    @beta_async_tool
    async def get_access_rules(
        card_id: str | None = None, group_id: str | None = None
    ) -> str:
        """Лимиты, региональные лимиты и товарные ограничители. 2 тарифных запроса.

        Используй перед изменением правил. Без card_id и group_id — правила договора.
        """
        if card_id and group_id:
            return _json(
                {"error": "bad_request", "message": "Укажи либо card_id, либо group_id (не оба)."}
            )
        group_id, defaults = with_default_group(card_id, group_id)
        try:
            limits_response = await client.limits.get_limits(card_id=card_id, group_id=group_id)
            budget.spend()
            regions_response = await client.region_limits.get_region_limits(
                card_id=card_id, group_id=group_id
            )
            budget.spend()
            restrictions_response = await client.restrictions.get_restrictions(
                card_id=card_id, group_id=group_id
            )
        except (APIError, BudgetExceededError, RequestValidationError) as exc:
            return _error(exc)
        regions = [
            {
                "id": item.id,
                "card_id": item.card_id,
                "group_id": item.group_id,
                "country": item.country,
                "region": item.region,
                "service_center": item.service_center,
                "partner": item.partner,
                "access": "allow" if item.limit_type == 1 else "deny",
                "limit_type": item.limit_type,
                "date": item.date,
            }
            for item in (regions_response.data.result or [])[:MAX_LIST]
        ]
        restrictions = [
            {
                "id": item.id,
                "card_id": item.card_id,
                "group_id": item.group_id,
                "product_type": item.productType,
                "product": item.productTypeName,
                "product_group": item.productGroup,
                "product_group_name": item.productGroupName,
                "access": "allow" if item.restriction_type == 1 else "deny",
                "restriction_type": item.restriction_type,
                "date": item.date,
            }
            for item in (restrictions_response.data.result or [])[:MAX_LIST]
        ]
        limits = await limits_payload(limits_response.data.result or [])
        return _json(
            {
                **({"defaults": defaults} if defaults else {}),
                "limits_total": limits_response.data.total_count,
                "limits": limits,
                "region_limits_total": regions_response.data.total_count,
                "region_limits": regions,
                "restrictions_total": restrictions_response.data.total_count,
                "restrictions": restrictions,
            }
        )

    @beta_async_tool
    async def set_limit(
        card_id: str | None = None,
        group_id: str | None = None,
        product_type: str | None = None,
        product_group: str | None = None,
        liters: float | None = None,
        rubles: float | None = None,
        period_number: int = 1,
        period_type: int = 0,
        transactions_count: int | None = None,
        limit_id: str | None = None,
    ) -> str:
        """Установить или изменить продуктовый лимит. ИЗМЕНЯЕТ ДАННЫЕ, по тарифу.

        Нужно указать ровно одну цель (card_id или group_id), хотя бы один размер
        (liters и/или rubles) и период. period_type: 3 — сутки, 7 — неделя, другие
        коды уточняй по get_limits. limit_id меняет существующий лимит.
        """
        card_id = (card_id or "").strip() or None
        group_id = (group_id or "").strip() or None
        if bool(card_id) == bool(group_id):
            return _json(
                {"error": "bad_request", "message": "Укажи ровно одну цель: card_id или group_id."}
            )
        if period_type not in {2, 3, 4, 5, 6, 7}:
            return _json(
                {
                    "error": "bad_request",
                    "message": "period_type: код 2–7; 3 — сутки, 7 — неделя. Уточни период.",
                }
            )
        if period_number < 1:
            return _json({"error": "bad_request", "message": "period_number должен быть ≥ 1."})
        if liters is None and rubles is None:
            return _json({"error": "bad_request", "message": "Нужен liters и/или rubles."})
        if liters is not None and liters <= 0:
            return _json({"error": "bad_request", "message": "liters должен быть больше нуля."})
        if rubles is not None and rubles <= 0:
            return _json({"error": "bad_request", "message": "rubles должен быть больше нуля."})
        if transactions_count is not None and transactions_count <= 0:
            return _json(
                {"error": "bad_request", "message": "transactions_count должен быть больше нуля."}
            )
        try:
            item = LimitRequestItem(
                id=limit_id,
                card_id=card_id,
                group_id=group_id,
                productType=product_type,
                productGroup=product_group,
                amount=(
                    LimitAmountRequest(unit="LIT", value=float(liters))
                    if liters is not None
                    else None
                ),
                sum=(
                    LimitSumRequest(currency="810", value=Decimal(str(rubles)))
                    if rubles is not None
                    else None
                ),
                time=LimitTimeRequest(number=period_number, type=cast(Literal[2, 3, 4, 5, 6, 7], period_type)),
                transactions=(
                    LimitTransactionsRequest(count=transactions_count)
                    if transactions_count is not None
                    else None
                ),
            )
        except Exception as exc:  # noqa: BLE001 — ошибку модели показываем как есть
            return _json({"error": "bad_request", "message": str(exc)[:300]})
        target = f"карте {card_id}" if card_id else f"группе {group_id}"
        sizes = ", ".join(
            part
            for part in (
                f"{liters:g} л" if liters is not None else "",
                f"{rubles:g} ₽" if rubles is not None else "",
            )
            if part
        )
        product = f" (продукт {product_type})" if product_type else ""
        change = f", изменить лимит {limit_id}" if limit_id else ""
        denied = await confirm_write(
            "set_limit",
            f"Установить лимит {sizes} на {period_number}×код {period_type} "
            f"для {target}{product}{change}?",
        )
        if denied:
            return denied
        try:
            budget.spend()
            response = await client.limits.set_limit(limits=[item])
        except (APIError, BudgetExceededError, RequestValidationError) as exc:
            return _error(exc)
        return _json({"status": "saved", "limit_ids": list(response.data or [])})

    @beta_async_tool
    async def set_region_limit(
        card_id: str | None = None,
        group_id: str | None = None,
        country: str = "RUS",
        region: str | None = None,
        service_center: str | None = None,
        partner: str | None = None,
        limit_type: int = 0,
        region_limit_id: str | None = None,
    ) -> str:
        """Разрешить или запретить регион/АЗС. ИЗМЕНЯЕТ ДАННЫЕ, по тарифу.

        limit_type: 1 — разрешающий, 2 — запрещающий. Нужно указать ровно одну
        цель (card_id или group_id). region_limit_id меняет существующее правило.
        """
        card_id = (card_id or "").strip() or None
        group_id = (group_id or "").strip() or None
        country = (country or "").strip()
        if bool(card_id) == bool(group_id):
            return _json(
                {"error": "bad_request", "message": "Укажи ровно одну цель: card_id или group_id."}
            )
        if limit_type not in {1, 2}:
            return _json({"error": "bad_request", "message": "limit_type: 1 — разрешить, 2 — запретить."})
        if not country:
            return _json({"error": "bad_request", "message": "Нужен код страны, например RUS."})
        try:
            item = RegionLimitRequestItem(
                id=region_limit_id,
                card_id=card_id,
                group_id=group_id,
                country=country,
                region=region,
                service_center=service_center,
                partner=partner,
                limit_type=cast(Literal[1, 2], limit_type),
            )
        except Exception as exc:  # noqa: BLE001
            return _json({"error": "bad_request", "message": str(exc)[:300]})
        verb = "Разрешить" if limit_type == 1 else "Запретить"
        target = f"карте {card_id}" if card_id else f"группе {group_id}"
        place = f"{country}" + (f"/{region}" if region else "")
        extra = f", АЗС {service_center}" if service_center else ""
        denied = await confirm_write(
            "set_region_limit", f"{verb} регион {place}{extra} для {target}?"
        )
        if denied:
            return denied
        try:
            budget.spend()
            response = await client.region_limits.set_region_limit(region_limits=[item])
        except (APIError, BudgetExceededError, RequestValidationError) as exc:
            return _error(exc)
        return _json({"status": "saved", "region_limit_ids": list(response.data or [])})

    @beta_async_tool
    async def set_restriction(
        card_id: str | None = None,
        group_id: str | None = None,
        product_type: str = "",
        product_group: str | None = None,
        restriction_type: int = 0,
        restriction_id: str | None = None,
    ) -> str:
        """Разрешить или запретить продукт. ИЗМЕНЯЕТ ДАННЫЕ, по тарифу.

        restriction_type: 1 — разрешающий, 2 — запрещающий. product_type — код
        продукта из get_dictionary("ProductType"), например 1-CK231. Нужно
        указать ровно одну цель (card_id или group_id).
        """
        card_id = (card_id or "").strip() or None
        group_id = (group_id or "").strip() or None
        product_type = (product_type or "").strip()
        if bool(card_id) == bool(group_id):
            return _json(
                {"error": "bad_request", "message": "Укажи ровно одну цель: card_id или group_id."}
            )
        if not product_type:
            return _json(
                {"error": "bad_request", "message": "Нужен product_type, например 1-CK231."}
            )
        if restriction_type not in {1, 2}:
            return _json(
                {"error": "bad_request", "message": "restriction_type: 1 — разрешить, 2 — запретить."}
            )
        try:
            item = RestrictionRequestItem(
                id=restriction_id,
                card_id=card_id,
                group_id=group_id,
                productType=product_type,
                productGroup=product_group,
                restriction_type=cast(Literal[1, 2], restriction_type),
            )
        except Exception as exc:  # noqa: BLE001
            return _json({"error": "bad_request", "message": str(exc)[:300]})
        verb = "Разрешить" if restriction_type == 1 else "Запретить"
        target = f"карте {card_id}" if card_id else f"группе {group_id}"
        denied = await confirm_write(
            "set_restriction", f"{verb} продукт {product_type} для {target}?"
        )
        if denied:
            return denied
        try:
            budget.spend()
            response = await client.restrictions.set_restriction(restrictions=[item])
        except (APIError, BudgetExceededError, RequestValidationError) as exc:
            return _error(exc)
        return _json({"status": "saved", "restriction_ids": list(response.data or [])})

    @beta_async_tool
    async def remove_limit(limit_id: str, group_id: str | None = None) -> str:
        """Удалить продуктовый лимит. ИЗМЕНЯЕТ ДАННЫЕ, по тарифу, необратимо."""
        limit_id = str(limit_id).strip()
        if not limit_id:
            return _json({"error": "bad_request", "message": "Нужен limit_id."})
        group_id = (group_id or "").strip() or None
        target = f" группы {group_id}" if group_id else ""
        denied = await confirm_write(
            "remove_limit", f"Удалить продуктовый лимит {limit_id}{target}? Это необратимо."
        )
        if denied:
            return denied
        try:
            budget.spend()
            response = await client.limits.remove_limit(limit_id=limit_id, group_id=group_id)
        except (APIError, BudgetExceededError, RequestValidationError) as exc:
            return _error(exc)
        return _json(
            {"status": "removed", "limit_id": limit_id, "removed": bool(response.data)}
        )

    @beta_async_tool
    async def remove_region_limit(
        region_limit_id: str, group_id: str | None = None
    ) -> str:
        """Удалить региональный лимит. ИЗМЕНЯЕТ ДАННЫЕ, по тарифу, необратимо."""
        region_limit_id = str(region_limit_id).strip()
        if not region_limit_id:
            return _json({"error": "bad_request", "message": "Нужен region_limit_id."})
        group_id = (group_id or "").strip() or None
        target = f" группы {group_id}" if group_id else ""
        denied = await confirm_write(
            "remove_region_limit",
            f"Удалить региональный лимит {region_limit_id}{target}? Это необратимо.",
        )
        if denied:
            return denied
        try:
            budget.spend()
            response = await client.region_limits.remove_region_limit(
                regionlimit_id=region_limit_id, group_id=group_id
            )
        except (APIError, BudgetExceededError, RequestValidationError) as exc:
            return _error(exc)
        return _json(
            {
                "status": "removed",
                "region_limit_id": region_limit_id,
                "removed": bool(response.data),
            }
        )

    @beta_async_tool
    async def remove_restriction(
        restriction_id: str, group_id: str | None = None
    ) -> str:
        """Удалить товарный ограничитель. ИЗМЕНЯЕТ ДАННЫЕ, по тарифу, необратимо."""
        restriction_id = str(restriction_id).strip()
        if not restriction_id:
            return _json({"error": "bad_request", "message": "Нужен restriction_id."})
        group_id = (group_id or "").strip() or None
        target = f" группы {group_id}" if group_id else ""
        denied = await confirm_write(
            "remove_restriction",
            f"Удалить товарный ограничитель {restriction_id}{target}? Это необратимо.",
        )
        if denied:
            return denied
        try:
            budget.spend()
            response = await client.restrictions.remove_restriction(
                restriction_id=restriction_id, group_id=group_id
            )
        except (APIError, BudgetExceededError, RequestValidationError) as exc:
            return _error(exc)
        return _json(
            {
                "status": "removed",
                "restriction_id": restriction_id,
                "removed": bool(response.data),
            }
        )

    return {
        "get_limits": get_limits,
        "get_access_rules": get_access_rules,
        "set_limit": set_limit,
        "set_region_limit": set_region_limit,
        "set_restriction": set_restriction,
        "remove_limit": remove_limit,
        "remove_region_limit": remove_region_limit,
        "remove_restriction": remove_restriction,
    }
