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

from ..config import Mode
from ..safety import BudgetExceededError
from .common import MAX_LIST, _error, _json, describe_limit, limit_matches, period_words, verified
from .context import ToolContext

# Региональные лимиты и ограничители бесплатно не перечитать: get_access_rules — 2 тарифицируемых запроса.
PAID_CHECK = verified(
    None,
    "Автоматически не сверено: проверка — get_access_rules (2 тарифицируемых запроса). "
    "Предложи пользователю, а не вызывай сам.",
)


def build(ctx: ToolContext) -> dict[str, Any]:
    """Инструменты темы по имени; порядок в списке агента задаёт agent/tools/__init__.py."""
    client = ctx.client
    budget = ctx.budget
    limits_payload = ctx.limits_payload
    confirm_write = ctx.confirm_write
    perform = ctx.perform
    finish = ctx.finish
    target_line = ctx.target_line
    rules_seen = ctx.rules_seen
    default_group_id = ctx.default_group_id

    def rule_text(kind: str, item: Any) -> str:
        """Региональный лимит или ограничитель словами — для подтверждения изменения."""
        whom = f"карты {item.card_id}" if item.card_id else f"группы {item.group_id}"
        if kind == "region":
            verb = "разрешён" if item.limit_type == 1 else "запрещён"
            place = "/".join(str(x) for x in (item.country, item.region) if x) or "регион не указан"
            station = f", АЗС {item.service_center}" if item.service_center else ""
            return f"региональный лимит {item.id}: {place}{station} {verb} для {whom}"
        verb = "разрешён" if item.restriction_type == 1 else "запрещён"
        product = item.productTypeName or item.productType or item.productGroupName or item.productGroup
        return f"товарный ограничитель {item.id}: продукт {product} {verb} для {whom}"

    def seen_for(kind: str, card_id: str | None, group_id: str | None) -> list[str]:
        return [
            str(rule["text"])
            for rule in rules_seen.values()
            if rule["kind"] == kind and (rule["card_id"], rule["group_id"]) == (card_id, group_id)
        ]

    def current_rules(kind: str, card_id: str | None, group_id: str | None) -> str:
        if not any(rule["kind"] == kind for rule in rules_seen.values()):
            return "Текущие правила этого вида не загружены: get_access_rules в этой сессии не вызывался."
        found = seen_for(kind, card_id, group_id)
        return "Текущие правила цели: " + ("; ".join(found) if found else "нет") + "."

    def read_only() -> bool:
        return ctx.gate.mode is Mode.READ_ONLY

    async def limit_check(
        card_id: str | None, group_id: str | None, limit_ids: list[str], same: Any, *, removed: bool = False
    ) -> dict[str, str]:
        """Сверка лимитов после изменения по бесплатному get_limits: оператор мог ответить и не применить."""
        if not limit_ids:
            return verified(None, "Оператор не вернул ID лимита: изменение не применено.")
        try:
            now = {str(i.id): i for i in (await client.limits.get_limits(card_id=card_id, group_id=group_id)).data.result or []}
        except (APIError, RequestValidationError):
            return verified(None, "get_limits не ответил: проверь лимиты цели позже.")
        if removed:
            gone = all(i not in now for i in limit_ids)
            return verified(
                gone, "Лимита больше нет в get_limits." if gone else "Лимит ещё виден в get_limits: проверь позже."
            )
        wrong = [i for i in limit_ids if i not in now or not same(now[i])]
        if wrong:
            return verified(
                False,
                f"В get_limits лимит {', '.join(wrong)} не совпал с подтверждённым: покажи пользователю "
                "текущий лимит из get_limits и не повторяй изменение без человека.",
            )
        return verified(True, "Лимит в get_limits совпал с подтверждённым.")

    def not_found(card_id: str | None, group_id: str | None) -> str:
        what = f"Карты с ID {card_id}" if card_id else f"Группы с ID {group_id}"
        return _json(
            {
                "error": "not_found",
                "message": f"{what} нет в договоре: ничего не изменено. Найди её через list_cards "
                "или list_card_groups и не угадывай ID.",
            }
        )

    def with_default_group(card_id: str | None, group_id: str | None) -> tuple[str | None, dict[str, str]]:
        """Группа по умолчанию, если не указаны ни карта, ни группа (только для чтения правил)."""
        if card_id or group_id or not default_group_id:
            return group_id, {}
        return default_group_id, {"group_id": f"{default_group_id} (OPTI24_DEFAULT_GROUP_ID)"}

    def old_owner(item: Any) -> str:
        return f"карта {item.card_id}" if item.card_id else (f"группа {item.group_id}" if item.group_id else "договор")

    async def rule_change_summary(
        kind: str, rule_id: str | None, card_id: str | None, group_id: str | None, change: str
    ) -> tuple[str, dict[str, object], str | None]:
        """Текст подтверждения правила, запись цели для журнала и готовый отказ, если цель неизвестна."""
        audit_target: dict[str, object] = {"card_id": card_id} if card_id else {"group_id": group_id}
        if rule_id:
            audit_target["rule_id"] = rule_id
        if read_only():
            return "", audit_target, None
        whom, found_target = await target_line(card_id, group_id)
        if whom is None:
            return "", audit_target, not_found(card_id, group_id)
        audit_target = {**found_target, **({"rule_id": rule_id} if rule_id else {})}
        if rule_id:
            old = rules_seen.get(f"{kind}:{rule_id}")
            if old is None:
                return "", audit_target, _json(
                    {
                        "error": "unknown_rule",
                        "message": f"Правила {rule_id} нет в ответах get_access_rules этой сессии: ничего не "
                        "изменено. Сначала вызови get_access_rules по этой карте или группе.",
                    }
                )
            return f"Изменить правило: {whom}.\nБыло: {old['text']}.\nСтанет: {change}.", audit_target, None
        return f"{change}: {whom}.\n{current_rules(kind, card_id, group_id)}", audit_target, None

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
        for kind, items in (
            ("region", regions_response.data.result or []),
            ("restriction", restrictions_response.data.result or []),
        ):
            for item in items:
                rules_seen[f"{kind}:{item.id}"] = {
                    "kind": kind,
                    "card_id": item.card_id,
                    "group_id": item.group_id,
                    "text": rule_text(kind, item),
                }
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

        Одна цель (card_id или group_id), liters и/или rubles, период. period_type:
        3 — сутки, 7 — неделя, другие коды — по get_limits. limit_id меняет лимит.
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
        sizes = ", ".join(
            part
            for part in (
                f"{liters:g} л" if liters is not None else "",
                f"{rubles:g} ₽" if rubles is not None else "",
            )
            if part
        )
        args = {
            "card_id": card_id, "group_id": group_id, "product_type": product_type, "product_group": product_group,
            "liters": liters, "rubles": rubles, "period_number": period_number, "period_type": period_type,
            "transactions_count": transactions_count, "limit_id": limit_id,
        }
        summary = ""
        audit_target: dict[str, object] = {"card_id": card_id} if card_id else {"group_id": group_id}
        if not read_only():
            whom, audit_target = await target_line(card_id, group_id)
            if whom is None:
                return not_found(card_id, group_id)
            try:
                existing = (await client.limits.get_limits(card_id=card_id, group_id=group_id)).data.result or []
            except (APIError, RequestValidationError) as exc:
                return _error(exc)
            new = (
                f"{sizes} на {period_words(period_number, period_type)}, продукт "
                f"{product_type or product_group or 'все продукты'}"
                + (f", не больше {transactions_count} операций" if transactions_count else "")
            )
            if limit_id:
                old = next((item for item in existing if str(item.id) == limit_id), None)
                if old is None:
                    return _json(
                        {
                            "error": "not_found",
                            "message": f"Лимита {limit_id} нет у этой цели в get_limits: ничего не изменено. "
                            "Возьми limit_id из get_limits по той же карте или группе.",
                        }
                    )
                summary = f"Изменить лимит: {whom}.\nБыло: {describe_limit(old)}.\nСтанет: {new}."
            else:
                now = "; ".join(describe_limit(item) for item in existing[:5]) or "нет"
                summary = f"Установить новый лимит: {whom}.\nСтанет: {new}.\nУже есть у цели: {now}."
        approval, denied = await confirm_write("set_limit", summary, 1, audit_target, args)
        if denied:
            return denied
        response, failed = await perform(
            approval, lambda: client.limits.set_limit(limits=[item]), "лимиты цели в get_limits"
        )
        if failed:
            return failed
        limit_ids = [str(i) for i in (response.data or [])]
        check = await limit_check(
            card_id, group_id, limit_ids, lambda found: limit_matches(found, liters, rubles, (period_number, period_type))
        )
        finish(
            approval, "done" if limit_ids else "not_done", {"limit_ids": limit_ids, "verified": check["status"]}
        )
        return _json({"status": "saved", "limit_ids": limit_ids, "verified": check})

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
        place = f"{country}" + (f"/{region}" if region else "")
        extra = f", АЗС {service_center}" if service_center else ""
        args = {
            "card_id": card_id, "group_id": group_id, "country": country, "region": region,
            "service_center": service_center, "partner": partner, "limit_type": limit_type,
            "region_limit_id": region_limit_id,
        }
        summary, audit_target, problem = await rule_change_summary(
            "region", region_limit_id, card_id, group_id, f"{verb} регион {place}{extra}"
        )
        if problem:
            return problem
        approval, denied = await confirm_write("set_region_limit", summary, 1, audit_target, args)
        if denied:
            return denied
        response, failed = await perform(
            approval,
            lambda: client.region_limits.set_region_limit(region_limits=[item]),
            "региональные лимиты цели в get_access_rules",
        )
        if failed:
            return failed
        ids = list(response.data or [])
        finish(approval, "done" if ids else "not_done", {"region_limit_ids": ids, "verified": "not_checked"})
        return _json({"status": "saved", "region_limit_ids": ids, "verified": PAID_CHECK})

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
        args = {
            "card_id": card_id, "group_id": group_id, "product_type": product_type,
            "product_group": product_group, "restriction_type": restriction_type, "restriction_id": restriction_id,
        }
        summary, audit_target, problem = await rule_change_summary(
            "restriction", restriction_id, card_id, group_id, f"{verb} продукт {product_type}"
        )
        if problem:
            return problem
        approval, denied = await confirm_write("set_restriction", summary, 1, audit_target, args)
        if denied:
            return denied
        response, failed = await perform(
            approval,
            lambda: client.restrictions.set_restriction(restrictions=[item]),
            "товарные ограничители цели в get_access_rules",
        )
        if failed:
            return failed
        ids = list(response.data or [])
        finish(approval, "done" if ids else "not_done", {"restriction_ids": ids, "verified": "not_checked"})
        return _json({"status": "saved", "restriction_ids": ids, "verified": PAID_CHECK})

    @beta_async_tool
    async def remove_limit(limit_id: str, card_id: str | None = None, group_id: str | None = None) -> str:
        """Удалить продуктовый лимит карты или группы. ИЗМЕНЯЕТ ДАННЫЕ, по тарифу, необратимо."""
        limit_id = str(limit_id).strip()
        if not limit_id:
            return _json({"error": "bad_request", "message": "Нужен limit_id."})
        card_id = (card_id or "").strip() or None
        group_id = (group_id or "").strip() or None
        summary = ""
        audit_target: dict[str, object] = {"limit_id": limit_id, **({"group_id": group_id} if group_id else {})}
        if not read_only():
            try:
                existing = (await client.limits.get_limits(card_id=card_id, group_id=group_id)).data.result or []
            except (APIError, RequestValidationError) as exc:
                return _error(exc)
            old = next((item for item in existing if str(item.id) == limit_id), None)
            if old is None:
                return _json(
                    {
                        "error": "not_found",
                        "message": f"Лимита {limit_id} нет в get_limits по этой цели: ничего не удалено. "
                        "Передай card_id или group_id, у которых он показан в get_limits.",
                    }
                )
            summary = f"Удалить {describe_limit(old)} ({old_owner(old)})? Это необратимо."
        approval, denied = await confirm_write("remove_limit", summary, 1, audit_target, {"limit_id": limit_id})
        if denied:
            return denied
        response, failed = await perform(
            approval,
            lambda: client.limits.remove_limit(limit_id=limit_id, group_id=group_id),
            "лимиты цели в get_limits",
        )
        if failed:
            return failed
        check = (
            await limit_check(card_id, group_id, [limit_id], lambda _: False, removed=True)
            if response.data
            else verified(None, "Оператор не удалил лимит.")
        )
        finish(approval, "done" if response.data else "not_done", {"verified": check["status"]})
        return _json({"status": "removed", "limit_id": limit_id, "removed": bool(response.data), "verified": check})

    async def remove_rule(kind: str, action: str, key: str, rule_id: str, group_id: str | None, call: Any) -> str:
        """Удаление регионального лимита или ограничителя по правилу из последнего get_access_rules."""
        rule_id = str(rule_id).strip()
        if not rule_id:
            return _json({"error": "bad_request", "message": "Нужен ID правила."})
        summary = ""
        if not read_only():
            rule = rules_seen.get(f"{kind}:{rule_id}")
            if rule is None:
                return _json(
                    {
                        "error": "unknown_rule",
                        "message": f"Правила {rule_id} нет в ответах get_access_rules этой сессии: ничего не "
                        "удалено. Сначала вызови get_access_rules по карте или группе, "
                        "чтобы человек видел, что удаляется.",
                    }
                )
            summary = f"Удалить {rule['text']}? Это необратимо."
        approval, denied = await confirm_write(
            action, summary, 1, {"rule_id": rule_id, **({"group_id": group_id} if group_id else {})}, {"id": rule_id}
        )
        if denied:
            return denied
        response, failed = await perform(approval, lambda: call(rule_id), "правила цели в get_access_rules")
        if failed:
            return failed
        finish(approval, "done" if response.data else "not_done", {"verified": "not_checked"})
        if response.data:
            rules_seen.pop(f"{kind}:{rule_id}", None)
        return _json({"status": "removed", key: rule_id, "removed": bool(response.data), "verified": PAID_CHECK})

    @beta_async_tool
    async def remove_region_limit(region_limit_id: str, group_id: str | None = None) -> str:
        """Удалить региональный лимит. ИЗМЕНЯЕТ ДАННЫЕ, по тарифу, необратимо."""
        group_id = (group_id or "").strip() or None
        return await remove_rule(
            "region",
            "remove_region_limit",
            "region_limit_id",
            region_limit_id,
            group_id,
            lambda rid: client.region_limits.remove_region_limit(regionlimit_id=rid, group_id=group_id),
        )

    @beta_async_tool
    async def remove_restriction(restriction_id: str, group_id: str | None = None) -> str:
        """Удалить товарный ограничитель. ИЗМЕНЯЕТ ДАННЫЕ, по тарифу, необратимо."""
        group_id = (group_id or "").strip() or None
        return await remove_rule(
            "restriction",
            "remove_restriction",
            "restriction_id",
            restriction_id,
            group_id,
            lambda rid: client.restrictions.remove_restriction(restriction_id=rid, group_id=group_id),
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
