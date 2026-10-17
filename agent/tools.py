"""Инструменты агента — тонкая обёртка над apisdkopti24.

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

import json
import os
import re
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from anthropic import beta_async_tool
from apisdkopti24 import APIClient, APIError
from apisdkopti24.errors import ContractSelectionError, RequestValidationError
from apisdkopti24.models.limits import (
    LimitAmountRequest,
    LimitRequestItem,
    LimitSumRequest,
    LimitTimeRequest,
    LimitTransactionsRequest,
)
from apisdkopti24.models.region_limits import RegionLimitRequestItem
from apisdkopti24.models.restrictions import RestrictionRequestItem

from .analysis import AnomalyRules, find_anomalies, summarize
from .analysis import fuel_price_analytics as price_analytics
from .config import Mode
from .money import rub, to_decimal
from .pricing import TERMS_EDITION, configured_service_fee, service_fee_for_volume, tiers_payload
from .safety import ApprovalGate, BillingBudget, BudgetExceededError

MAX_LIST = 25
# Бесплатный пакет запросов API в месяц по текущему тарифу; после него API блокируется.
# Пакет зависит от договора, поэтому значение настраивается (OPTI24_MONTHLY_QUOTA).
DEFAULT_MONTHLY_QUOTA = 500
QUOTA_WARNING_LEFT = 50
# Регион АЗС ищется отдельным бесплатным запросом на каждую АЗС; при лимите SDK
# 1 запрос в секунду больше 40 АЗС за вопрос — слишком долгое ожидание.
MAX_POI_LOOKUPS = 40


def _json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


class TransactionsUnavailableError(Exception):
    """Стенд не отдаёт операции (на DEMO get_transactions_v2 отвечает HTTP 405)."""


def _error(exc: Exception) -> str:
    # Текст исключений SDK очищен от секретов, но всё равно ограничиваем длину.
    return _json({"error": type(exc).__name__, "message": str(exc)[:300]})


def _rub_optional(value: object) -> str | None:
    """Сумма или None: не показываем «0 ₽» там, где API вернул null."""
    if value is None or value == "":
        return None
    return rub(to_decimal(value))


def _iso(value: object) -> str:
    return value.isoformat() if isinstance(value, datetime) else str(value)


def _doc_date(value: object) -> str:
    """Дата документа: UNIX timestamp или строка."""
    try:
        return datetime.fromtimestamp(int(value)).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        return str(value)


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


def build_tools(
    client: APIClient,
    budget: BillingBudget,
    gate: ApprovalGate,
    reports_dir: Path | None = None,
) -> list[Any]:
    """Собирает инструменты, привязанные к клиенту, бюджету и подтверждению."""

    reports_root = Path(
        reports_dir or os.environ.get("AGENT_REPORTS_DIR", "reports")
    ).expanduser()

    monthly_quota = int(os.environ.get("OPTI24_MONTHLY_QUOTA", DEFAULT_MONTHLY_QUOTA))

    def quota_payload(info: Any) -> dict[str, object]:
        """Расход месячного пакета; бюджет сессии не больше остатка, чтобы не довести до блокировки."""
        package = info.client_info.Queries
        quota = int(package) if package else monthly_quota
        used = int(info.methods.all or 0)
        left = max(quota - used, 0)
        budget.limit = min(budget.limit, budget.spent + left)
        payload: dict[str, object] = {
            "monthly_quota": quota,
            "quota_source": "API" if package else "настройка OPTI24_MONTHLY_QUOTA",
            "calls_in_period": used,
            "left_in_period": left,
        }
        if left <= QUOTA_WARNING_LEFT:
            payload["warning"] = (
                f"В пакете осталось {left} запросов. Когда пакет закончится, API заблокируется "
                "до конца периода — для агента и для всех интеграций договора."
            )
        return payload

    # Операции за период: (date_from, date_to, card_id) -> список. Живёт столько же,
    # сколько бюджет сессии, поэтому повторный вопрос о том же периоде бесплатен.
    tx_cache: dict[tuple[str, str, str | None], list[Any]] = {}

    async def transactions(date_from: str, date_to: str, card_id: str | None) -> list[Any]:
        key = (date_from, date_to, card_id or None)
        if key not in tx_cache:
            tx_cache[key], _ = await load_transactions(client, budget, date_from, date_to, card_id)
        return tx_cache[key]

    # Названия продуктов: справочник бесплатный, поэтому кешируем на сессию.
    product_names: dict[str, str] = {}

    async def product_label(code: str | None) -> str | None:
        if not code:
            return None
        if not product_names:
            try:
                data = (await client.dictionaries.get_dictionary(name="ProductType")).data
                for item in data.result or []:
                    product_names[item.id] = item.name or item.value or item.id
            except Exception:  # noqa: BLE001 — справочник не критичен для ответа
                product_names["__failed__"] = ""
        return product_names.get(code)

    # Названия товаров в прайсе АЗС (справочник Goods): бесплатно, кешируем на сессию.
    goods_names: dict[str, str] = {}

    async def goods_label(code: str) -> str | None:
        if not goods_names:
            try:
                data = (await client.dictionaries.get_dictionary(name="Goods")).data
                for item in (data.result if data else None) or []:
                    label = item.name or item.value or item.id
                    goods_names[item.id] = label
                    if item.code:
                        goods_names[item.code] = label
            except APIError:
                goods_names["__failed__"] = ""
        return goods_names.get(code)

    # Регион АЗС: ID АЗС -> название региона. Справочник и АЗС бесплатны, кешируем на сессию.
    region_names: dict[str, str] = {}
    poi_regions: dict[str, str] = {}

    async def region_label(code: str) -> str:
        if not region_names:
            try:
                data = (await client.dictionaries.get_dictionary(name="Region")).data
                for item in (data.result if data else None) or []:
                    label = item.name or item.value or item.id
                    region_names[item.id] = label
                    if item.code:
                        region_names[item.code] = label
            except APIError:
                region_names["__failed__"] = ""
        return region_names.get(code) or f"регион {code}"

    async def regions_for(poi_ids: set[str]) -> tuple[dict[str, str], int]:
        """Регионы для АЗС из кеша и до MAX_POI_LOOKUPS новых запросов; второе — сколько АЗС не проверено."""
        missing = sorted(poi for poi in poi_ids if poi and poi not in poi_regions)
        for poi in missing[:MAX_POI_LOOKUPS]:
            try:
                data = (await client.dictionaries.get_azs_list_v1(id=poi)).data
            except APIError:
                continue
            stations = (data.result if data else None) or []
            if stations and stations[0].regionCode:
                poi_regions[poi] = await region_label(stations[0].regionCode)
        return {poi: poi_regions[poi] for poi in poi_ids if poi in poi_regions}, max(
            0, len(missing) - MAX_POI_LOOKUPS
        )

    async def station_info(poi: str) -> Any | None:
        """АЗС по ID, а если не нашлась — по номеру или адресу из чека, когда совпадение одно.

        Бесплатно, есть на DEMO. None — АЗС не найдена или совпадений несколько.
        """
        for params in ({"id": poi}, {"q": poi, "onpage": 2}):
            try:
                data = (await client.dictionaries.get_azs_list_v1(**params)).data
            except (APIError, RequestValidationError):
                continue
            stations = (data.result if data else None) or []
            if len(stations) == 1:
                return stations[0]
        return None

    async def purchase_check(
        card_id: str, poi_id: str | None, product: str | None, liters: float | None, station: Any | None
    ) -> dict[str, object]:
        """check_purchase по товару с ценой из прайса АЗС; без данных — пропуск с причиной."""
        if not (poi_id and product and liters and liters > 0):
            return {"allowed": None, "skipped": "нужны poi_id АЗС, product и liters из чека или от водителя"}
        if station is None:
            return {"allowed": None, "skipped": "АЗС не найдена или совпадений несколько: уточни ID АЗС"}
        needle = product.casefold()
        prices = station.prices or []
        offered = [(price, await goods_label(price.GoodsCode)) for price in prices]
        match = next(
            (price for price, label in offered if needle in (label or price.GoodsCode).casefold()),
            None,
        )
        if match is None:
            return {
                "allowed": None,
                "skipped": "товара нет в прайсе АЗС",
                "station_goods": [label or price.GoodsCode for price, label in offered][:MAX_LIST],
            }
        goods = [{"code": match.GoodsCode, "quantity": float(liters), "price": float(to_decimal(match.Price))}]
        station_id = getattr(station, "id", None) or poi_id
        try:
            response = await client.final_prices.check_purchase(card_id=card_id, poi_id=station_id, goods=goods)
        except (APIError, RequestValidationError) as exc:
            # Метод только на рабочем API и только для роли «Администратор».
            return {"allowed": None, "skipped": f"проверка не выполнена: {type(exc).__name__}"}
        return {"allowed": bool(response.data), "goods": goods}

    async def limits_payload(items: list[Any]) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for item in items[:MAX_LIST]:
            result.append(
                {
                    "id": item.id,
                    "card_id": item.card_id,
                    "group_id": item.group_id,
                    "product": await product_label(item.productType),
                    "product_type": item.productType,
                    "product_group": item.productGroup,
                    "period": {"number": item.time.number, "type": item.time.type},
                    "amount": item.amount.model_dump() if item.amount else None,
                    "sum": item.sum.model_dump() if item.sum else None,
                    "term": item.term.model_dump() if item.term else None,
                    "transactions": item.transactions.model_dump() if item.transactions else None,
                }
            )
        return result

    async def confirm_write(action: str, summary: str, objects: int = 1) -> str | None:
        """None — можно выполнять; строка — готовый отказ (read_only или declined)."""
        if gate.mode is Mode.READ_ONLY:
            gate._audit(action, objects, False)
            return _json(
                {
                    "status": "not_allowed",
                    "message": "Агент запущен в режиме только чтения (AGENT_MODE=read_only): "
                    "изменение отключено настройкой, данные не тронуты, оператор запрос не получал. "
                    "Так и скажи пользователю; включить изменение можно запуском с AGENT_MODE=confirm.",
                }
            )
        if not await gate.approve(action, summary, objects):
            return _json(
                {"status": "declined", "message": "Пользователь не подтвердил действие: данные не тронуты."}
            )
        return None

    @beta_async_tool
    async def get_account_context() -> str:
        """Договор, роль, режим только чтения, пакет и период статистики. Бесплатно.

        Вызывай в начале разговора, чтобы понять, какой договор выбран и можно ли
        что-то менять. Если договоров несколько, покажи их номера пользователю.
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
                "period": {"from": _iso(info.from_), "to": _iso(info.to)},
                **quota_payload(info),
            }
        )

    @beta_async_tool
    async def get_contract_summary() -> str:
        """Баланс договора, расход месяца и карты. Платный запрос API (1 шт.).

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
                "cards_total": cards.cards_quantity_all,
                "cards_active": cards.cards_quantity_active,
                # Необязательные лимитные поля: null, а не «0 ₽».
                "limit_available": _rub_optional(info.balance_amount_limiter),
                "accumulated": _rub_optional(info.current_amount_limiter),
                "credit_limit": _rub_optional(info.credit_limit),
            }
        )

    @beta_async_tool
    async def list_cards(
        status: str | None = None, group_id: str | None = None, query: str | None = None
    ) -> str:
        """Список топливных карт договора: ID, номер, статус, группа, комментарий. Бесплатно.

        Нужен, чтобы найти ID карты для get_limits, spending_report и block_cards.
        Показывает не больше 25 карт; если total больше shown, сузь фильтр.

        Args:
            status: Системный код статуса из get_dictionary("CardStatus"):
                "Active", "Locked(Client)", "Locked(Manager)", "Inactive" и т. п.
                Русские названия ("Активна") API не принимает.
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
        total = response.data.total_count
        hint = (
            {"hint": "Показана часть карт: уточни query, status или group_id."}
            if total and total > len(cards)
            else {}
        )
        return _json(
            {
                "total": total,
                "shown": len(cards),
                **hint,
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
    async def get_card_profile(
        card_id: str, include_drivers: bool = True, include_limits: bool = True
    ) -> str:
        """Карта, водитель, группа и лимиты. Платно: 1–2 запроса API.

        Вызывай после list_cards, когда нужно понять, кто закреплён за картой и
        какие у неё лимиты. include_drivers=False экономит один платный запрос.
        """
        try:
            budget.spend()
            detail_response = await client.cards.get_card_detail(card_id=card_id)
            details = detail_response.data.result or []
            if not details:
                return _json({"error": "not_found", "message": "Карта не найдена."})
            card = details[0]

            # Группа берётся из бесплатного списка карт по номеру.
            group = None
            try:
                cards_response = await client.cards.get_cards_v2(q=card.number, onpage=1)
                cards = cards_response.data.result or []
                group = cards[0].group_name if cards else None
            except Exception:  # noqa: BLE001 — группа не критична для профиля
                group = None

            profile: dict[str, object] = {
                "id": card.id,
                "number": card.number,
                "status": card.status,
                "comment": card.comment,
                "group": group,
                "product": card.product,
                "carrier": card.carrier,
                "available": _rub_optional(card.available),
                "currency": card.currency,
                "last_used_at": _iso(card.date_last_usage) if card.date_last_usage else None,
                "last_station": card.servicecenter_last_usage_name,
                "mpc": card.mpc,
                "pin_reset": card.pin_reset,
                "pin_counter": card.pin_counter,
            }

            if include_drivers:
                budget.spend()
                drivers_response = await client.cards.get_card_drivers(card_id=card_id)
                profile["drivers"] = [
                    {
                        "name": " ".join(
                            part for part in (d.last_name, d.first_name, d.middle_name) if part
                        ),
                        "position": d.position,
                        "role": d.role,
                        "phone": d.mobile_phone,
                    }
                    for d in (drivers_response.data.result or [])
                ]

            if include_limits:
                limits_response = await client.limits.get_limits(card_id=card_id)
                profile["limits_total"] = limits_response.data.total_count
                profile["limits"] = await limits_payload(limits_response.data.result or [])
        except (APIError, RequestValidationError, BudgetExceededError) as exc:
            return _error(exc)
        return _json(profile)

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
        """Продуктовые лимиты: продукт, период, установлено и израсходовано. Бесплатно.

        Без card_id и group_id возвращает лимиты всего договора. Если указаны оба,
        это ошибка: выбери что-то одно.
        """
        if card_id and group_id:
            return _json(
                {"error": "bad_request", "message": "Укажи либо card_id, либо group_id (не оба)."}
            )
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
                "limits": await limits_payload(shown),
            }
        )

    @beta_async_tool
    async def get_access_rules(
        card_id: str | None = None, group_id: str | None = None
    ) -> str:
        """Лимиты, региональные лимиты и товарные ограничители. 2 платных запроса.

        Используй перед изменением правил и для вопроса «почему карта не работает».
        Без card_id и group_id показывает правила всего договора.
        """
        if card_id and group_id:
            return _json(
                {"error": "bad_request", "message": "Укажи либо card_id, либо group_id (не оба)."}
            )
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
                "limits_total": limits_response.data.total_count,
                "limits": limits,
                "region_limits_total": regions_response.data.total_count,
                "region_limits": regions,
                "restrictions_total": restrictions_response.data.total_count,
                "restrictions": restrictions,
            }
        )

    @beta_async_tool
    async def spending_report(date_from: str, date_to: str, card_id: str | None = None) -> str:
        """Сводка расходов за период: сумма, литры и цена литра по топливу, топ-5 карт.

        Платный запрос API: 1 запрос на каждую страницу из 100 операций.
        Период не больше месяца. Повторный вызов за тот же период (и
        find_suspicious_fills за него) бесплатен: операции уже загружены.

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

        Цена клиента = сумма операции / литры. Платно как spending_report (операции
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
                f"не больше {MAX_POI_LOOKUPS}. Повтори вопрос, чтобы проверить остальные."
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

        Платный запрос API, но за период, уже загруженный spending_report или
        find_suspicious_fills, страницы повторно не оплачиваются.
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

        Платный запрос API: 1 запрос на каждую страницу из 100 операций; за
        период, уже загруженный spending_report, бесплатно. Правила проверки
        заданы в коде и приходят в поле rules: пересказывай их, а не придумывай.

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
    async def get_finance(
        kind: str, date_from: str | None = None, date_to: str | None = None, limit: int = 10
    ) -> str:
        """Платежи, счета или закрывающие документы.

        kind=payments — платный запрос; kind=invoices и kind=documents — бесплатные.
        Для documents нужны date_from и date_to. Период не больше месяца.
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
    async def get_report_status() -> str:
        """Последние заказанные отчёты: job_id, название, формат, дата. Бесплатно."""
        try:
            response = await client.reports.get_report_job_list_v1()
            items = list(response.data or [])
        except (APIError, RequestValidationError) as exc:
            return _error(exc)
        rows = [
            {
                "job_id": item.job_id,
                "date": item.date,
                "report_name": item.report_name,
                "report_format": item.report_format,
                "contract_id": item.contract_id,
            }
            for item in items
        ]
        rows.sort(key=lambda row: row["date"], reverse=True)
        return _json(
            {
                "total": len(rows),
                "shown": len(rows[:10]),
                **(
                    {"hint": "Показаны последние 10 отчётов."}
                    if len(rows) > 10
                    else {}
                ),
                "reports": rows[:10],
            }
        )

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

    @beta_async_tool
    async def block_cards(card_ids: list[str], reason: str, block: bool = True) -> str:
        """Заблокировать или разблокировать карты. ИЗМЕНЯЕТ ДАННЫЕ, нужно подтверждение.

        Вызывай только по прямой просьбе пользователя и по конкретным картам.
        block=True блокирует, block=False разблокирует; разблокировка — отдельное
        решение человека.

        Args:
            card_ids: ID карт (из list_cards), не больше 10.
            reason: Причина одной фразой, для подтверждения человеку.
            block: True — заблокировать, False — разблокировать.
        """
        card_ids = list(dict.fromkeys(str(c).strip() for c in card_ids or [] if str(c).strip()))
        if not card_ids or len(card_ids) > 10:
            return _json({"error": "bad_request", "message": "Нужно от 1 до 10 карт."})
        action = "block_cards" if block else "unblock_cards"
        verb = "Заблокировать" if block else "Разблокировать"
        if gate.mode is Mode.READ_ONLY:
            gate._audit(action, len(card_ids), False)
            return _json(
                {
                    "status": "not_allowed",
                    "message": "Агент запущен в режиме только чтения (AGENT_MODE=read_only): "
                    "изменение отключено настройкой, карты НЕ тронуты, оператор запрос не получал. "
                    "Так и скажи пользователю; включить изменение можно запуском с AGENT_MODE=confirm.",
                }
            )
        approved = await gate.approve(
            action,
            # Человек видит, какие именно карты меняются; в журнал ID не попадают.
            f"{verb} карт: {len(card_ids)} ({', '.join(card_ids)}). Причина: {reason}",
            len(card_ids),
        )
        if not approved:
            return _json(
                {
                    "status": "declined",
                    "message": "Пользователь не подтвердил действие: карты не тронуты.",
                }
            )
        try:
            budget.spend()
            response = await client.cards.block_card(card_ids=card_ids, block=block)
        except (APIError, BudgetExceededError, RequestValidationError) as exc:
            return _error(exc)
        done = [str(item) for item in (response.data or [])]
        not_done = [card_id for card_id in card_ids if card_id not in done]
        return _json(
            {
                "status": "partial" if not_done else ("blocked" if block else "unblocked"),
                "requested": len(card_ids),
                "done": done,
                "not_done": not_done,
            }
        )

    @beta_async_tool
    async def set_card_comment(card_id: str, comment: str) -> str:
        """Установить комментарий карты (водитель, машина). ИЗМЕНЯЕТ ДАННЫЕ.

        Платный запрос, нужно подтверждение человека. Комментарий 1–90 символов.
        """
        card_id = str(card_id).strip()
        comment = str(comment).strip()
        if not card_id:
            return _json({"error": "bad_request", "message": "Нужен card_id."})
        if not 1 <= len(comment) <= 90:
            return _json(
                {"error": "bad_request", "message": "Комментарий должен быть от 1 до 90 символов."}
            )
        denied = await confirm_write(
            "set_card_comment",
            f"Установить комментарий «{comment}» карте {card_id}?",
        )
        if denied:
            return denied
        try:
            budget.spend()
            response = await client.cards.set_card_comment(card_id=card_id, comment=comment)
        except (APIError, BudgetExceededError, RequestValidationError) as exc:
            return _error(exc)
        return _json(
            {
                "status": "updated",
                "card_id": card_id,
                "comment": comment,
                "saved": bool(response.data),
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
        """Установить или изменить продуктовый лимит. ИЗМЕНЯЕТ ДАННЫЕ, платно.

        Нужно указать ровно одну цель (card_id или group_id), хотя бы один размер
        (liters и/или rubles) и период. period_type — код периода API: 3 — сутки
        (пример SDK), 7 — неделя (описание SDK); остальные коды уточняй по
        get_limits или у администратора. limit_id меняет существующий лимит.
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
                product_type=product_type,
                product_group=product_group,
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
                time=LimitTimeRequest(number=period_number, type=period_type),
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
        """Разрешить или запретить регион/АЗС. ИЗМЕНЯЕТ ДАННЫЕ, платно.

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
                limit_type=limit_type,
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
        """Разрешить или запретить продукт. ИЗМЕНЯЕТ ДАННЫЕ, платно.

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
                product_type=product_type,
                product_group=product_group,
                restriction_type=restriction_type,
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
    async def order_fuel_report(
        date_from: str,
        date_to: str,
        report_format: str = "xlsx",
        email: str | None = None,
        card_ids: list[str] | None = None,
        group_ids: list[str] | None = None,
    ) -> str:
        """Заказать транзакционный отчёт за период. ИЗМЕНЯЕТ ДАННЫЕ, платно.

        Отчёт формируется асинхронно: вернёт job_id, статус смотри
        get_report_status. format: xlsx, xml, pdf или csv. Если указан email,
        отчёт уйдёт на него; без email доступен по ссылке в личном кабинете.
        """
        if problem := _check_dates(date_from, date_to):
            return _json({"error": "bad_dates", "message": problem})
        report_format = (report_format or "").strip().lower()
        if report_format not in {"xlsx", "xml", "pdf", "csv"}:
            return _json(
                {"error": "bad_request", "message": "format: xlsx, xml, pdf или csv."}
            )
        email = (email or "").strip() or None
        if email and "@" not in email:
            return _json({"error": "bad_request", "message": "Похоже, email указан неверно."})
        cards = [str(item).strip() for item in (card_ids or []) if str(item).strip()] or None
        groups = [str(item).strip() for item in (group_ids or []) if str(item).strip()] or None
        summary = (
            f"Заказать отчёт {report_format} за {date_from}…{date_to}"
            + (f" по картам {len(cards)}" if cards else "")
            + (f" по группам {len(groups)}" if groups else "")
            + (f" и отправить на {email}" if email else "")
            + "? Отчёт платный и формируется асинхронно."
        )
        denied = await confirm_write("order_fuel_report", summary, len(cards or groups or [1]))
        if denied:
            return denied
        try:
            budget.spend()
            response = await client.reports.order_report_v1(
                start=date_from,
                end=date_to,
                report_format=report_format,
                email=email,
                cards_list=cards,
                group_id=groups,
            )
        except (APIError, BudgetExceededError, RequestValidationError) as exc:
            return _error(exc)
        return _json(
            {
                "status": "ordered",
                "period": f"{date_from}…{date_to}",
                "format": report_format,
                "email": email,
                "job_ids": list(response.data or []),
            }
        )

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
        """Почему отказ на АЗС: статус карты, check_purchase, лимиты, баланс, регионы и ограничители, PIN.

        Укажи card_id или query (номер/комментарий). Код и текст причины — из чека
        отказа. poi_id — ID АЗС или её номер/адрес из чека; с product и liters нужен для
        check_purchase. 4 платных запроса; называет вероятные причины, а не гарантированные.
        """
        card_id = (card_id or "").strip() or None
        query = (query or "").strip() or None
        if bool(card_id) == bool(query):
            return _json(
                {"error": "bad_request", "message": "Укажи ровно одно: card_id или query."}
            )
        receipt = {
            "decline_code": (decline_code or "").strip()[:20] or None,
            "decline_text": (decline_text or "").strip()[:120] or None,
        }
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
        if receipt["decline_text"] and re.search(r"pin|пин", receipt["decline_text"], re.IGNORECASE):
            causes.append("в чеке отказа упомянут PIN: возможно, превышено число неверных вводов")
            steps.append("Сбросить счётчик неверных вводов PIN в личном кабинете или через поддержку оператора.")
        if not causes:
            causes.append("явной причины в данных карты и договора нет")
            steps.append(
                "Повторить заправку; если отказ повторяется — позвонить в поддержку по телефону "
                "из чека и назвать код ответа, дату, время и АЗС."
            )

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
                "checks": {
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
                        "ok": None,
                        "note": "API не показывает счётчик неверных вводов PIN; спроси водителя.",
                    },
                },
                "likely_causes": causes,
                "next_steps": steps,
            }
        )

    @beta_async_tool
    async def download_fuel_report(
        job_id: str, report_format: str = "xlsx", destination: str | None = None
    ) -> str:
        """Скачать готовый отчёт в папку reports. Платный запрос API.

        Сначала проверь get_report_status: job_id должен быть готов. Файл
        сохраняется локально и не вставляется в чат. report_format нужен только
        для расширения; destination — имя файла без пути.
        """
        job_id = str(job_id).strip()
        if not job_id:
            return _json({"error": "bad_request", "message": "Нужен job_id."})
        fmt = (report_format or "xlsx").strip().lower()
        if fmt not in {"xlsx", "xml", "pdf", "csv"}:
            return _json(
                {"error": "bad_request", "message": "report_format: xlsx, xml, pdf или csv."}
            )
        if destination:
            name = Path(destination).name
            if name != destination or name in {"", ".", ".."}:
                return _json(
                    {"error": "bad_request", "message": "destination — только имя файла без пути."}
                )
        else:
            safe = re.sub(r"[^A-Za-z0-9._-]", "_", job_id)[:80] or "report"
            name = f"fuel-report-{safe}.{fmt}"
        root = reports_root.resolve()
        path = (root / name).resolve()
        if path.parent != root:
            return _json({"error": "bad_request", "message": "Недопустимый путь файла."})
        try:
            root.mkdir(parents=True, exist_ok=True)
            budget.spend()
            saved = Path(
                await client.reports.download_report_file_v1_to(
                    job_id=job_id, destination=path
                )
            )
        except (APIError, BudgetExceededError, RequestValidationError, OSError) as exc:
            return _error(exc)
        return _json(
            {
                "status": "downloaded",
                "job_id": job_id,
                "path": str(saved),
                "bytes": saved.stat().st_size if saved.exists() else None,
            }
        )

    @beta_async_tool
    async def remove_limit(limit_id: str, group_id: str | None = None) -> str:
        """Удалить продуктовый лимит. ИЗМЕНЯЕТ ДАННЫЕ, платно, необратимо."""
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
        """Удалить региональный лимит. ИЗМЕНЯЕТ ДАННЫЕ, платно, необратимо."""
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
        """Удалить товарный ограничитель. ИЗМЕНЯЕТ ДАННЫЕ, платно, необратимо."""
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

    read_tools = [
        get_account_context,
        get_contract_summary,
        list_cards,
        list_card_groups,
        get_card_profile,
        get_limits,
        get_access_rules,
        diagnose_fueling,
        spending_report,
        list_transactions,
        get_transaction_detail,
        find_suspicious_fills,
        fuel_price_analytics,
        get_finance,
        get_report_status,
        download_fuel_report,
        get_api_usage,
    ]
    write_tools = [
        block_cards,
        set_card_comment,
        set_limit,
        set_region_limit,
        set_restriction,
        order_fuel_report,
        remove_limit,
        remove_region_limit,
        remove_restriction,
    ]
    # Изменяющие инструменты всегда в списке: в режиме read_only гейт их отклонит,
    # а модель получит понятный ответ и объяснит пользователю, что режим только чтение.
    return [*read_tools, *write_tools]
