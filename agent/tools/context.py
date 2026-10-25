"""Общее состояние инструментов одной сессии: клиент, бюджет, подтверждение и кеши.

Инструменты разных тем пользуются одним кешем операций, справочников и АЗС,
поэтому кеши живут в контексте, а не в модулях инструментов.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from apisdkopti24 import APIClient, APIError
from apisdkopti24.errors import RequestValidationError, ServerError
from apisdkopti24.models.transactions import TransactionItemV2

from ..config import Mode
from ..db import OperationStore, open_store
from ..knowledge import load_sections
from ..money import to_decimal
from ..rag import KnowledgeIndex, embedder_from_env, index_path
from ..safety import Approval, ApprovalGate, BillingBudget, BudgetExceededError, mask_card_number
from . import common
from .common import (
    DEFAULT_MONTHLY_QUOTA,
    MAX_LIST,
    QUOTA_WARNING_LEFT,
    _error,
    _json,
    describe_card,
    limit_left,
    load_transactions,
    percent,
)

# Поиск карты по ID для подтверждения: страниц бесплатного списка карт и карт на странице.
CARD_LOOKUP_PAGES = 20
CARD_LOOKUP_PAGE_SIZE = 100


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
    transactions_source: Any
    dictionary: Any
    dictionary_label: Any
    product_label: Any
    goods_label: Any
    region_label: Any
    regions_for: Any
    station_info: Any
    purchase_check: Any
    limits_payload: Any
    confirm_write: Any
    perform: Any
    finish: Any
    cards_by_id: Any
    target_line: Any
    rules_seen: Any
    # Значения по умолчанию договора (переменные окружения) и почта пользователя из профиля.
    default_product: str | None
    default_group_id: str | None
    profile_email: Any


def make_context(
    client: APIClient,
    budget: BillingBudget,
    gate: ApprovalGate,
    reports_dir: Path | None = None,
    store: OperationStore | None | bool = True,
) -> ToolContext:
    """store=True — база из AGENT_DB_PATH (см. agent.db.open_store), None/False — без базы."""

    reports_root = Path(
        reports_dir or os.environ.get("AGENT_REPORTS_DIR", "reports")
    ).expanduser()

    monthly_quota = int(os.environ.get("OPTI24_MONTHLY_QUOTA", DEFAULT_MONTHLY_QUOTA))
    # Подставляются, только если пользователь не назвал своё: инструмент сообщает, что взял значение по умолчанию.
    default_product = os.environ.get("OPTI24_DEFAULT_PRODUCT", "").strip() or None
    default_group_id = os.environ.get("OPTI24_DEFAULT_GROUP_ID", "").strip() or None
    profile: dict[str, str | None] = {}

    async def profile_email() -> str | None:
        """Почта пользователя, под которым агент вошёл в API: auth_user бесплатен, читается раз за сессию."""
        if "email" not in profile:
            auth = await client.auth.auth_user(contract_id=getattr(client, "contract_id", None))
            profile["email"] = (auth.data.email or "").strip() or None
        return profile["email"]
    # Размер пакета «известен», если его сообщил API или его явно задали: только тогда он
    # ограничивает бюджет. Значение по умолчанию — ориентир: общий счётчик стенда DEMO
    # показывает больше 500 запросов, и без этой оговорки тарифные вызовы были бы закрыты.
    quota_configured = os.environ.get("OPTI24_MONTHLY_QUOTA") is not None
    # Выжимки договора маленькие и читаются один раз при старте, а не в асинхронном пути.
    knowledge = KnowledgeIndex(load_sections(), embedder_from_env(), index_path())

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
            # Доли считает код: модель не делит сама.
            "used_percent": percent(used, quota),
            "left_percent": percent(left, quota),
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
    # Откуда взяты операции периода: из локальной базы (и на какой момент) или из API.
    tx_source: dict[tuple[str, str, str | None], dict[str, str]] = {}
    # Договор становится известен после входа в API, поэтому база открывается при первом вопросе.
    opened: dict[str, OperationStore | None] = {} if store is True else {"store": store or None}

    def operation_store() -> OperationStore | None:
        if "store" not in opened:
            contract_id = getattr(client, "contract_id", None)
            if not contract_id:
                return None
            opened["store"] = open_store(contract_id)
        return opened["store"]

    async def transactions(date_from: str, date_to: str, card_id: str | None) -> list[Any]:
        key = (date_from, date_to, card_id or None)
        if key in tx_cache:
            return tx_cache[key]

        now = datetime.now().replace(microsecond=0)
        day_from, day_to = date.fromisoformat(date_from), date.fromisoformat(date_to)
        operations = operation_store()
        saved = operations.lookup(day_from, day_to, card_id, now) if operations else None
        if saved is not None:
            tx_cache[key] = [TransactionItemV2.model_validate(raw) for raw in saved.items]
            tx_source[key] = {
                "data_source": "локальная база",
                "data_as_of": saved.loaded_at.isoformat(timespec="minutes"),
            }
            return tx_cache[key]

        items, total = await load_transactions(client, budget, date_from, date_to, card_id)
        tx_cache[key] = items
        tx_source[key] = {"data_source": "API", "data_as_of": now.isoformat(timespec="minutes")}
        # Неполную загрузку не сохраняем как период; сохраняем только модели SDK.
        if operations and len(items) >= total and all(isinstance(i, TransactionItemV2) for i in items):
            raw = [item.model_dump(mode="json", by_alias=True) for item in items]
            operations.save(day_from, day_to, card_id, raw, now)
        return items

    def transactions_source(date_from: str, date_to: str, card_id: str | None) -> dict[str, str]:
        """data_source и data_as_of для ответа инструмента; пусто, если период не загружали."""
        return tx_source.get((date_from, date_to, card_id or None), {})

    # Справочники API бесплатны: каждый загружается один раз за сессию. Код и ID записи → название.
    dictionaries: dict[str, list[Any]] = {}

    async def dictionary(name: str) -> list[Any]:
        """Записи справочника; пустой список, если справочник недоступен (ответ без расшифровки)."""
        if name not in dictionaries:
            try:
                data = (await client.dictionaries.get_dictionary(name=name)).data
                dictionaries[name] = list((data.result if data else None) or [])
            except Exception:  # noqa: BLE001 — справочник не критичен для ответа
                dictionaries[name] = []
        return dictionaries[name]

    async def dictionary_label(name: str, code: object) -> str | None:
        """Название записи справочника по её ID или коду."""
        if code is None or code == "":
            return None
        key = str(code)
        for item in await dictionary(name):
            if key in (str(item.id), str(getattr(item, "code", None) or "")):
                return str(item.name or item.value or item.id)
        return None

    async def product_label(code: str | None) -> str | None:
        return await dictionary_label("ProductType", code)

    async def goods_label(code: str) -> str | None:
        return await dictionary_label("Goods", code)

    # Регион АЗС: ID АЗС -> название региона. Справочник и АЗС бесплатны, кешируем на сессию.
    poi_regions: dict[str, str] = {}

    async def region_label(code: str) -> str:
        return await dictionary_label("Region", code) or f"регион {code}"

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
                    "amount_left": limit_left(item.amount),
                    "sum_left": limit_left(item.sum),
                    "term": item.term.model_dump() if item.term else None,
                    "transactions": item.transactions.model_dump() if item.transactions else None,
                }
            )
        return result

    async def confirm_write(
        action: str,
        summary: str,
        objects: int = 1,
        target: dict[str, object] | None = None,
        args: dict[str, object] | None = None,
    ) -> tuple[Approval | None, str | None]:
        """(решение, None) — можно выполнять; (None, строка) — готовый отказ модели."""
        if gate.mode is Mode.READ_ONLY:
            gate.deny(action, objects, target=target, args=args)
            return None, _json(
                {
                    "status": "not_allowed",
                    "message": "Агент запущен в режиме только чтения (AGENT_MODE=read_only): "
                    "изменение отключено настройкой, данные не тронуты, оператор запрос не получал. "
                    "Так и скажи пользователю; включить изменение можно запуском с AGENT_MODE=confirm.",
                }
            )
        if budget.left < 1:  # не спрашиваем человека о том, что выполнить уже нельзя
            return None, _json(
                {
                    "error": "BudgetExceededError",
                    "message": f"Исчерпан бюджет тарифицируемых запросов на сессию ({budget.limit}): "
                    "изменение не выполнено. Предложи продолжить в новой сессии.",
                }
            )
        approval = await gate.approve(action, summary, objects, target=target, args=args)
        if not approval:
            return None, _json(
                {"status": "declined", "message": "Пользователь не подтвердил действие: данные не тронуты."}
            )
        return approval, None

    async def perform(approval: Approval, call: Any, check: str) -> tuple[Any, str | None]:
        """Выполняет подтверждённое изменение: (ответ API, None) или (None, ответ модели об ошибке).

        Отказ API до выполнения (4xx, проверка запроса) — изменение не выполнено. Таймаут, обрыв
        связи, ошибка сервера или непонятный ответ — исход неизвестен: оператор мог уже применить
        изменение, и повтор создал бы второе правило. ``check`` — чем проверить текущее состояние.
        """
        try:
            budget.spend()
        except BudgetExceededError as exc:
            gate.outcome(approval, "not_sent", {"error": type(exc).__name__})
            return None, _error(exc)
        try:
            response = await call()
        except RequestValidationError as exc:
            gate.outcome(approval, "not_sent", {"error": type(exc).__name__})
            return None, _error(exc)
        except APIError as exc:
            if not isinstance(exc, ServerError):
                gate.outcome(approval, "error", {"error": type(exc).__name__, "http": exc.http_status_code})
                return None, _error(exc)
            failure: Exception = exc
        except Exception as exc:  # noqa: BLE001 — исход неизвестен при любой другой ошибке
            failure = exc
        else:
            return response, None
        gate.outcome(approval, "unknown", {"error": type(failure).__name__})
        return None, _json(
            {
                "status": "unknown",
                "error": type(failure).__name__,
                "message": "Ответ оператора не получен: изменение могло выполниться. НЕ вызывай инструмент "
                f"повторно. Скажи пользователю, что результат неизвестен, и предложи проверить: {check}.",
            }
        )

    def finish(approval: Approval, status: str, detail: dict[str, object] | None = None) -> None:
        """Исход подтверждённого изменения в журнал аудита."""
        gate.outcome(approval, status, detail)

    async def cards_by_id(card_ids: list[str]) -> dict[str, Any]:
        """Карты договора по ID из бесплатного списка карт; ненайденных в ответе нет.

        API не ищет карту по ID, поэтому список читается страницами, пока не найдутся все.
        """
        wanted = set(card_ids)
        found: dict[str, Any] = {}
        seen = 0
        for page in range(1, CARD_LOOKUP_PAGES + 1):
            try:
                response = await client.cards.get_cards_v2(page=page, onpage=CARD_LOOKUP_PAGE_SIZE)
            except (APIError, RequestValidationError):
                break
            batch = response.data.result or []
            seen += len(batch)
            found.update({str(c.id): c for c in batch if str(c.id) in wanted})
            if wanted <= found.keys() or not batch or seen >= (response.data.total_count or 0):
                break
        return found

    async def group_name(group_id: str) -> str | None:
        """Название группы карт из бесплатного списка групп; None, если группы нет."""
        try:
            groups = (await client.card_groups.get_card_groups()).data.result or []
        except (APIError, RequestValidationError):
            return None
        return next((g.name for g in groups if str(g.id) == group_id), None)

    async def target_line(card_id: str | None, group_id: str | None) -> tuple[str | None, dict[str, object]]:
        """Цель изменения словами из API и её запись для журнала; (None, …) — карта или группа не найдена."""
        if card_id:
            card = (await cards_by_id([card_id])).get(card_id)
            if card is None:
                return None, {"card_id": card_id}
            return f"карта {describe_card(card)}", {"card_id": card_id, "card": mask_card_number(card.number)}
        assert group_id
        name = await group_name(group_id)
        if name is None:
            return None, {"group_id": group_id}
        return f"группа «{name}» (ID {group_id})", {"group_id": group_id}

    # Правила из последнего get_access_rules: по ним подтверждение показывает, что меняется или удаляется.
    rules_seen: dict[str, dict[str, object]] = {}

    return ToolContext(
        client=client,
        budget=budget,
        gate=gate,
        reports_root=reports_root,
        knowledge=knowledge,
        monthly_quota=monthly_quota,
        quota_payload=quota_payload,
        transactions=transactions,
        transactions_source=transactions_source,
        dictionary=dictionary,
        dictionary_label=dictionary_label,
        product_label=product_label,
        goods_label=goods_label,
        region_label=region_label,
        regions_for=regions_for,
        station_info=station_info,
        purchase_check=purchase_check,
        limits_payload=limits_payload,
        confirm_write=confirm_write,
        perform=perform,
        finish=finish,
        cards_by_id=cards_by_id,
        target_line=target_line,
        rules_seen=rules_seen,
        default_product=default_product,
        default_group_id=default_group_id,
        profile_email=profile_email,
    )
