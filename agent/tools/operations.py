"""Инструменты агента: Операции: расходы, аналитика цены, аномалии и разбор отказа на АЗС."""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any

from anthropic import beta_async_tool
from apisdkopti24 import APIError
from apisdkopti24.errors import RequestValidationError

from ..analysis import AnomalyRules, find_anomalies, summarize
from ..analysis import fuel_price_analytics as price_analytics
from ..decline_codes import decode as decode_decline
from ..money import rub, to_decimal
from ..pricing import TERMS_EDITION, configured_service_fee, service_fee_for_volume, tiers_payload
from ..safety import BudgetExceededError
from . import common
from .common import TransactionsUnavailableError, _check_dates, _error, _json, _rub_optional
from .context import ToolContext


def build(ctx: ToolContext) -> dict[str, Any]:
    """Инструменты темы по имени; порядок в списке агента задаёт agent/tools/__init__.py."""
    client = ctx.client
    budget = ctx.budget
    transactions = ctx.transactions
    product_label = ctx.product_label
    regions_for = ctx.regions_for
    station_info = ctx.station_info
    purchase_check = ctx.purchase_check

    @beta_async_tool
    async def spending_report(date_from: str, date_to: str, card_id: str | None = None) -> str:
        """Сводка расходов за период: сумма, литры и цена литра по топливу, топ-5 карт.

        Тарифный запрос API: 1 запрос на каждую страницу из 100 операций.
        Период не больше месяца; операции за загруженный период — из кеша.

        Args:
            date_from: Начало периода, ГГГГ-ММ-ДД.
            date_to: Конец периода, ГГГГ-ММ-ДД.
            card_id: Ограничить одной картой (необязательно).
        """
        if problem := _check_dates(date_from, date_to):
            return _json({"error": "bad_dates", "message": problem})
        try:
            items = await transactions(date_from, date_to, card_id)
        except (APIError, BudgetExceededError, RequestValidationError, TransactionsUnavailableError) as exc:
            return _error(exc)
        return _json({"period": f"{date_from}…{date_to}", **summarize(items)})

    @beta_async_tool
    async def fuel_price_analytics(
        date_from: str,
        date_to: str,
        card_id: str | None = None,
        product: str | None = None,
        service_fee_percent: float | None = None,
        previous_month_liters: float | None = None,
    ) -> str:
        """Цена топлива для клиента со скидкой и сервисным сбором по топливу и регионам РФ, где дешевле.

        Цена клиента = сумма операции / литры. Тариф как у spending_report (операции
        кешируются); регион АЗС бесплатно. Период не больше месяца.

        Args:
            date_from: Начало периода, ГГГГ-ММ-ДД.
            date_to: Конец периода, ГГГГ-ММ-ДД.
            card_id: Ограничить одной картой (необязательно).
            product: Название топлива как в операциях, например "ДТ" (необязательно).
            service_fee_percent: Сервисный сбор договора, % (если известен).
            previous_month_liters: Литры за прошлый месяц: по ним выбирается ставка сбора.
        """
        if problem := _check_dates(date_from, date_to):
            return _json({"error": "bad_dates", "message": problem})
        try:
            items = await transactions(date_from, date_to, card_id)
        except (APIError, BudgetExceededError, RequestValidationError, TransactionsUnavailableError) as exc:
            return _error(exc)
        if product:
            needle = product.casefold()
            items = [t for t in items if needle in (t.product_name or "").casefold()]
        region_of, unchecked = await regions_for({t.poi_id for t in items})
        fee: Decimal | None
        if service_fee_percent is not None:
            fee, fee_source = to_decimal(service_fee_percent), "указан в вопросе"
        elif (fee := configured_service_fee()) is not None:
            fee_source = "OPTI24_SERVICE_FEE_PERCENT"
        elif previous_month_liters is not None:
            fee = service_fee_for_volume(to_decimal(previous_month_liters))
            fee_source = f"ставка по объёму прошлого месяца, {TERMS_EDITION}"
        else:
            fee_source = "неизвестен"
        result: dict[str, object] = {
            "period": f"{date_from}…{date_to}",
            **price_analytics(items, region_of, service_fee_percent=fee),
            "service_fee_source": fee_source,
        }
        if fee is None:
            result["service_fee_tiers"] = {"terms": TERMS_EDITION, "tiers": tiers_payload()}
        if unchecked:
            result["hint"] = (
                f"Регион не проверен у {unchecked} АЗС: за один вопрос проверяется "
                f"не больше {common.MAX_POI_LOOKUPS}. Повтори вопрос, чтобы проверить остальные."
            )
        return _json(result)

    @beta_async_tool
    async def list_transactions(
        date_from: str,
        date_to: str,
        card_id: str | None = None,
        product: str | None = None,
        min_sum: float | None = None,
        limit: int = 20,
    ) -> str:
        """Конкретные операции за период: дата, карта, АЗС, продукт, литры, цена, сумма.

        Тарифный запрос API; операции за загруженный период — из кеша.
        """
        if problem := _check_dates(date_from, date_to):
            return _json({"error": "bad_dates", "message": problem})
        try:
            items = await transactions(date_from, date_to, card_id)
        except (APIError, BudgetExceededError, RequestValidationError, TransactionsUnavailableError) as exc:
            return _error(exc)

        rows = []
        for t in items:
            if product and product.lower() not in (t.product_name or "").lower():
                continue
            if min_sum is not None and to_decimal(t.sum) < to_decimal(min_sum):
                continue
            rows.append(t)
        rows.sort(key=lambda t: t.timestamp, reverse=True)
        total = len(rows)
        top = max(1, min(int(limit), 50))
        shown = rows[:top]
        return _json(
            {
                "period": f"{date_from}…{date_to}",
                "total": total,
                "shown": len(shown),
                **(
                    {"hint": "Показана часть операций: сузь период, карту или продукт."}
                    if total > len(shown)
                    else {}
                ),
                "transactions": [
                    {
                        "date": f"{t.timestamp:%Y-%m-%d %H:%M}",
                        "card_id": t.card_id,
                        "card_number": t.card_number,
                        "poi_id": t.poi_id,
                        "type": t.type,
                        "product": t.product_name,
                        "liters": str(t.qty),
                        "price": rub(to_decimal(t.price)),
                        "sum": rub(to_decimal(t.sum)),
                        "is_storno": t.is_storno,
                        "is_manual_correction": t.is_manual_correction,
                    }
                    for t in shown
                ],
            }
        )

    @beta_async_tool
    async def get_transaction_detail(transaction_id: str) -> str:
        """Детали одной операции по ID. Бесплатный запрос API."""
        try:
            response = await client.transactions.get_transaction_detail(
                transaction_id=transaction_id
            )
            items = response.data.result or []
        except (APIError, RequestValidationError) as exc:
            return _error(exc)
        if not items:
            return _json({"error": "not_found", "message": "Операция не найдена."})
        t = items[0]
        return _json(
            {
                "id": str(t.id),
                "date": f"{t.timestamp:%Y-%m-%d %H:%M}",
                "card_id": t.card_id,
                "card_number": t.card_number,
                "poi_id": t.poi_id,
                "terminal_id": t.terminal_id,
                "check_id": str(t.check_id),
                "type": t.type,
                "product": t.product_name,
                "liters": str(t.qty),
                "price": rub(to_decimal(t.price)),
                "sum": rub(to_decimal(t.sum)),
                "discount": rub(to_decimal(t.discount)),
                "is_storno": t.is_storno,
                "is_manual_correction": t.is_manual_correction,
            }
        )

    @beta_async_tool
    async def find_suspicious_fills(date_from: str, date_to: str, card_id: str | None = None) -> str:
        """Подозрительные заправки за период: дорогие, ночные, крупные, частые.

        Тарифный запрос API: 1 запрос на каждую страницу из 100 операций; за
        период, уже загруженный spending_report, бесплатно.

        Args:
            date_from: Начало периода, ГГГГ-ММ-ДД.
            date_to: Конец периода, ГГГГ-ММ-ДД.
            card_id: Ограничить одной картой (необязательно).
        """
        if problem := _check_dates(date_from, date_to):
            return _json({"error": "bad_dates", "message": problem})
        try:
            items = await transactions(date_from, date_to, card_id)
        except (APIError, BudgetExceededError, RequestValidationError, TransactionsUnavailableError) as exc:
            return _error(exc)
        return _json(find_anomalies(items, AnomalyRules()))

    @beta_async_tool
    async def diagnose_fueling(
        card_id: str | None = None,
        query: str | None = None,
        decline_code: str | None = None,
        decline_text: str | None = None,
        poi_id: str | None = None,
        product: str | None = None,
        liters: float | None = None,
    ) -> str:
        """Почему карту не приняли на АЗС: статус, check_purchase, лимиты, баланс, регионы, PIN.

        Укажи card_id или query (номер/комментарий). Код и текст причины — из чека
        отказа. poi_id — ID АЗС или её номер/адрес из чека; с product и liters нужен для
        check_purchase. 4 тарифных запроса; причины вероятные, не гарантия.
        """
        card_id = (card_id or "").strip() or None
        query = (query or "").strip() or None
        if bool(card_id) == bool(query):
            return _json(
                {"error": "bad_request", "message": "Укажи ровно одно: card_id или query."}
            )
        code = (decline_code or "").strip()[:20] or None
        text = (decline_text or "").strip()[:120] or None
        receipt: dict[str, object] = {"decline_code": code, "decline_text": text}
        try:
            if query:
                found = await client.cards.get_cards_v2(q=query, onpage=2)
                cards = found.data.result or []
                if not cards:
                    return _json({"error": "not_found", "message": "Карта не найдена."})
                if len(cards) > 1:
                    return _json(
                        {
                            "status": "ambiguous",
                            "message": "Найдено несколько карт: уточни ID или номер.",
                            "candidates": [
                                {
                                    "id": c.id,
                                    "number": c.number,
                                    "comment": c.comment,
                                    "status": c.status_name or c.status,
                                    "group": c.group_name,
                                }
                                for c in cards
                            ],
                        }
                    )
                card_id = cards[0].id

            # 1. Статус карты.
            budget.spend()
            detail = (await client.cards.get_card_detail(card_id=card_id)).data.result or []
            if not detail:
                return _json({"error": "not_found", "message": "Карта не найдена."})
            card = detail[0]
            status_ok = card.status == "Active"

            # 2. Прямой ответ сервера: пройдёт ли покупка (бесплатно, только рабочий API).
            station = await station_info(poi_id) if poi_id else None
            purchase = await purchase_check(card_id, poi_id, product, liters, station)

            # 3. Продуктовые лимиты и остаток по ним.
            limits_response = await client.limits.get_limits(card_id=card_id)
            limits = limits_response.data.result or []
            limit_rows = []
            for item in limits:
                exceeded = any(
                    part is not None and to_decimal(part.value) > 0
                    and to_decimal(part.used) >= to_decimal(part.value)
                    for part in (item.amount, item.sum)
                )
                if item.transactions and item.transactions.count > 0:
                    exceeded = exceeded or item.transactions.occured >= item.transactions.count
                limit_rows.append(
                    {
                        "id": item.id,
                        "product": await product_label(item.productType),
                        "period": {"number": item.time.number, "type": item.time.type},
                        "amount": item.amount.model_dump() if item.amount else None,
                        "sum": item.sum.model_dump() if item.sum else None,
                        "exceeded": exceeded,
                    }
                )
            exceeded_limits = [row for row in limit_rows if row["exceeded"]]

            # 4. Баланс договора.
            budget.spend()
            contract = (await client.contracts.get_contract_data()).data
            available = to_decimal(contract.balanceData.available_amount)

            # 5. Региональные лимиты и товарные ограничители.
            budget.spend()
            regions_response = await client.region_limits.get_region_limits(card_id=card_id)
            regions = regions_response.data.result or []

            def region_row(r: Any) -> dict[str, object]:
                return {
                    "id": r.id,
                    "country": r.country,
                    "region": r.region,
                    "service_center": r.service_center,
                }

            deny_regions = [region_row(r) for r in regions if r.limit_type == 2]
            allow_regions = [region_row(r) for r in regions if r.limit_type == 1]
            station_region = station.regionCode if station else None

            def matches_station(row: dict[str, object]) -> bool:
                station_ids = {sid for sid in (getattr(station, "id", None), poi_id) if sid}
                return station is not None and (
                    row["service_center"] in station_ids
                    or (station_region is not None and row["region"] == station_region)
                )

            denied_here = [row for row in deny_regions if matches_station(row)]
            # Есть разрешающие правила, но ни одно не про эту АЗС или её регион.
            outside_allowed = bool(station and allow_regions) and not any(
                matches_station(row) for row in allow_regions
            )

            budget.spend()
            restrictions_response = await client.restrictions.get_restrictions(card_id=card_id)
            restrictions = restrictions_response.data.result or []
            deny_restrictions = [
                {"id": r.id, "product_type": r.productType, "product": r.productTypeName}
                for r in restrictions
                if r.restriction_type == 2
            ]
            allow_restrictions = [
                {"id": r.id, "product_type": r.productType, "product": r.productTypeName}
                for r in restrictions
                if r.restriction_type == 1
            ]
        except (APIError, BudgetExceededError, RequestValidationError) as exc:
            return _error(exc)

        causes: list[str] = []
        steps: list[str] = []
        if not status_ok:
            causes.append(f"карта в статусе {card.status}: это может блокировать заправку")
            steps.append("Проверить блокировку и при необходимости разблокировать по решению человека.")
        if purchase.get("allowed") is False:
            causes.append("сервер отвечает, что покупка этого товара по карте на этой АЗС невозможна")
            steps.append("Сверить остальные проверки: они подскажут, какое правило мешает.")
        if exceeded_limits:
            causes.append("исчерпан продуктовый лимит")
            steps.append("Проверить лимит и при необходимости изменить его или дождаться сброса периода.")
        if available <= 0:
            causes.append(f"на договоре доступно {rub(available)}")
            steps.append("Пополнить договор.")
        if denied_here or outside_allowed:
            causes.append("региональный лимит не пускает на эту АЗС или в этот регион")
            steps.append("Проверить регион/АЗС и при необходимости изменить региональное правило.")
        elif deny_regions:
            causes.append("есть запрещающий региональный лимит; без АЗС из чека нельзя сказать, мешает ли он")
            steps.append("Узнать АЗС из чека отказа и повторить проверку с poi_id.")
        if deny_restrictions:
            causes.append("есть запрещающий товарный ограничитель")
            steps.append("Проверить продукт и при необходимости изменить ограничитель.")
        # get_card_detail отдаёт счётчик попыток ввода PIN: ненулевой — повод проверить PIN.
        pin_counter = getattr(card, "pin_counter", None)
        if pin_counter:
            causes.append(f"по карте были неверные вводы PIN (счётчик {pin_counter}): после нескольких ошибок карта блокируется")
            steps.append("Уточнить у водителя PIN; сбросить счётчик можно в личном кабинете или через поддержку.")
        if text and re.search(r"pin|пин", text, re.IGNORECASE):
            causes.append("в чеке отказа упомянут PIN: возможно, превышено число неверных вводов")
            steps.append("Сбросить счётчик неверных вводов PIN в личном кабинете или через поддержку оператора.")
        decoded = decode_decline(code) if code else None
        if not causes and not decoded:
            causes.append("явной причины в данных карты и договора нет")
            steps.append(
                "Повторить заправку; если отказ повторяется — позвонить в поддержку по телефону "
                "из чека и назвать код ответа, дату, время и АЗС."
            )

        checks: dict[str, dict[str, object]] = {
            "card_status": {"ok": status_ok, "value": card.status},
            "check_purchase": purchase,
            "limits": {"ok": not exceeded_limits, "items": limit_rows},
            "contract_balance": {"ok": available > 0, "value": rub(available)},
            "region_limits": {
                # None — запрет есть, но без АЗС из чека не видно, касается ли он заправки.
                "ok": False
                if denied_here or outside_allowed
                else (None if deny_regions and not poi_id else True),
                "station_region": station_region,
                "deny": deny_regions,
                "allow": allow_regions,
            },
            "restrictions": {
                "ok": not deny_restrictions,
                "deny": deny_restrictions,
                "allow": allow_restrictions,
            },
            "pin_attempts": {
                "ok": None if pin_counter is None else pin_counter == 0,
                "counter": pin_counter,
                "resets_left": getattr(card, "pin_reset", None),
            },
        }
        if decoded:
            # Код терминала — прямое указание причины: он идёт первым, а проверки его подтверждают.
            causes.insert(0, f"по коду ответа {decoded['code']} из чека: {decoded['meaning']}")
            steps.insert(0, str(decoded["action"]))
            related = decoded["related_check"]
            if related and checks.get(str(related), {}).get("ok") is True:
                decoded["note"] = (
                    "проверка, которая соответствует коду, сейчас в порядке: состояние могло "
                    "измениться после отказа (например, счёт пополнили) — уточни время отказа"
                )
            receipt = {**receipt, "decoded": decoded}
        return _json(
            {
                "card": {
                    "id": card.id,
                    "number": card.number,
                    "status": card.status,
                    "comment": card.comment,
                    "available": _rub_optional(card.available),
                },
                "receipt": receipt,
                "checks": checks,
                "likely_causes": causes,
                "next_steps": steps,
            }
        )

    return {
        "spending_report": spending_report,
        "fuel_price_analytics": fuel_price_analytics,
        "list_transactions": list_transactions,
        "get_transaction_detail": get_transaction_detail,
        "find_suspicious_fills": find_suspicious_fills,
        "diagnose_fueling": diagnose_fueling,
    }
