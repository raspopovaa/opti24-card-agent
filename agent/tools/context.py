"""Общее состояние инструментов одной сессии: клиент, бюджет, подтверждение и кеши.

Инструменты разных тем пользуются одним кешем операций, справочников и АЗС,
поэтому кеши живут в контексте, а не в модулях инструментов.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from apisdkopti24 import APIClient, APIError
from apisdkopti24.errors import RequestValidationError

from ..config import Mode
from ..knowledge import load_sections
from ..money import to_decimal
from ..safety import ApprovalGate, BillingBudget
from . import common
from .common import DEFAULT_MONTHLY_QUOTA, MAX_LIST, QUOTA_WARNING_LEFT, _json, load_transactions


@dataclass(frozen=True, slots=True)
class ToolContext:
    client: Any
    budget: Any
    gate: Any
    reports_root: Any
    knowledge: Any
    monthly_quota: Any
    quota_payload: Any
    transactions: Any
    product_label: Any
    goods_label: Any
    region_label: Any
    regions_for: Any
    station_info: Any
    purchase_check: Any
    limits_payload: Any
    confirm_write: Any


def make_context(
    client: APIClient,
    budget: BillingBudget,
    gate: ApprovalGate,
    reports_dir: Path | None = None,
) -> ToolContext:

    reports_root = Path(
        reports_dir or os.environ.get("AGENT_REPORTS_DIR", "reports")
    ).expanduser()

    monthly_quota = int(os.environ.get("OPTI24_MONTHLY_QUOTA", DEFAULT_MONTHLY_QUOTA))
    # Размер пакета «известен», если его сообщил API или его явно задали: только тогда он
    # ограничивает бюджет. Значение по умолчанию — ориентир: общий счётчик стенда DEMO
    # показывает больше 500 запросов, и без этой оговорки платные вызовы были бы закрыты.
    quota_configured = os.environ.get("OPTI24_MONTHLY_QUOTA") is not None
    # Выжимки договора маленькие и читаются один раз при старте, а не в асинхронном пути.
    knowledge = load_sections()

    def quota_payload(info: Any) -> dict[str, object]:
        """Расход месячного пакета; при известном размере бюджет сессии не больше остатка."""
        package = info.client_info.Queries
        quota = int(package) if package else monthly_quota
        used = int(info.methods.all or 0)
        left = max(quota - used, 0)
        enforced = bool(package) or quota_configured
        if enforced:
            budget.limit = min(budget.limit, budget.spent + left)
        payload: dict[str, object] = {
            "monthly_quota": quota,
            "quota_source": "API" if package else (
                "настройка OPTI24_MONTHLY_QUOTA" if quota_configured else "по умолчанию (ориентир)"
            ),
            "quota_enforced": enforced,
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
                for item in (data.result if data else None) or []:
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
        """Регионы для АЗС из кеша и до common.MAX_POI_LOOKUPS новых запросов; второе — сколько АЗС не проверено."""
        missing = sorted(poi for poi in poi_ids if poi and poi not in poi_regions)
        for poi in missing[:common.MAX_POI_LOOKUPS]:
            try:
                data = (await client.dictionaries.get_azs_list_v1(id=poi)).data
            except APIError:
                continue
            stations = (data.result if data else None) or []
            if stations and stations[0].regionCode:
                poi_regions[poi] = await region_label(stations[0].regionCode)
        return {poi: poi_regions[poi] for poi in poi_ids if poi in poi_regions}, max(
            0, len(missing) - common.MAX_POI_LOOKUPS
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

    return ToolContext(
        client=client,
        budget=budget,
        gate=gate,
        reports_root=reports_root,
        knowledge=knowledge,
        monthly_quota=monthly_quota,
        quota_payload=quota_payload,
        transactions=transactions,
        product_label=product_label,
        goods_label=goods_label,
        region_label=region_label,
        regions_for=regions_for,
        station_info=station_info,
        purchase_check=purchase_check,
        limits_payload=limits_payload,
        confirm_write=confirm_write,
    )
