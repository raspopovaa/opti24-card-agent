"""Демо-режим: синтетический договор вместо API Opti24.

На первой встрече клиент не даёт ключ API, а стенд DEMO не отдаёт операции (HTTP 405).
``SyntheticOpti24`` повторяет ту часть ``APIClient`` из apisdkopti24, которой пользуются
инструменты агента, и отвечает выдуманными, но правдоподобными данными: перевозчик
с магистральными тягачами и городскими фургонами, операции с начала прошлого месяца,
лимиты, правила и несколько заправок, которые требуют проверки.

Данные одинаковы при каждом запуске (фиксированный seed) и сдвигаются вместе с сегодняшней
датой. Изменения (блокировка, лимиты, комментарии) живут только в памяти процесса.
Ни одного запроса к Opti24 демо-режим не делает; модель при этом нужна настоящая.
"""

from __future__ import annotations

import random
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, cast

from apisdkopti24 import APIClient

DEMO_CONTRACT_ID = "demo-contract"
DEMO_CONTRACT_NUMBER = "ДЕМО-2026/01"
DEMO_ORG = "ООО «Демо Логистик» (выдуманная компания)"
DEMO_NOTE = "Демо-режим: синтетический договор, данные выдуманы, запросов к Opti24 нет."

# Справочники. Коды выдуманы: настоящие коды берите в get_dictionary рабочего договора.
PRODUCT_TYPES = {
    "DEMO-FUEL": "Топливо",
    "DEMO-DT": "ДТ",
    "DEMO-AI92": "АИ-92",
    "DEMO-AI95": "АИ-95",
    "DEMO-WASH": "Мойка",
}
GOODS = {"DT": "ДТ", "AI92": "АИ-92", "AI95": "АИ-95"}
REGIONS = {
    "77": "Москва",
    "50": "Московская область",
    "69": "Тверская область",
    "78": "Санкт-Петербург",
    "52": "Нижегородская область",
}


class Dump:
    """Часть ответа с полями и model_dump(), как у моделей SDK."""

    def __init__(self, **data: object) -> None:
        self.__dict__.update(data)

    def model_dump(self) -> dict[str, object]:
        """Поля как словарь."""
        return dict(self.__dict__)


@dataclass(frozen=True, slots=True)
class Station:
    id: str
    name: str
    region: str
    prices: dict[str, Decimal]


STATIONS = [
    Station("poi-101", "АЗС №101, МКАД 47 км", "77", {"DT": Decimal("71.90"), "AI92": Decimal("58.40")}),
    Station("poi-102", "АЗС №102, Варшавское ш.", "77", {"DT": Decimal("72.30"), "AI92": Decimal("58.90")}),
    Station("poi-205", "АЗС №205, М-11 км 98", "50", {"DT": Decimal("70.80"), "AI92": Decimal("57.60")}),
    Station("poi-318", "АЗС №318, М-11 км 167", "69", {"DT": Decimal("69.40"), "AI92": Decimal("56.90")}),
    Station("poi-420", "АЗС №420, КАД 12 км", "78", {"DT": Decimal("70.10"), "AI92": Decimal("57.80")}),
    Station("poi-512", "АЗС №512, М-7 км 402", "52", {"DT": Decimal("68.90"), "AI92": Decimal("56.30")}),
    # Дорогая АЗС у склада: на ней одна из заправок, которые требуют проверки.
    Station("poi-777", "АЗС «Экспресс», Подольск", "50", {"DT": Decimal("79.50"), "AI92": Decimal("65.90")}),
]
DISCOUNT_PER_LITER = Decimal("1.20")

# (ID, номер, группа, комментарий, водитель, топливо, тип машины)
CARDS = [
    ("c-01", "7826010000100001", "g-long", "Вольво FH, А101АА77, Петров", ("Петров", "Сергей"), "DT", "truck"),
    ("c-02", "7826010000100002", "g-long", "Скания R450, А102АА77, Смирнов", ("Смирнов", "Олег"), "DT", "truck"),
    ("c-03", "7826010000100003", "g-long", "КАМАЗ 54901, А103АА77, Кузнецов", ("Кузнецов", "Иван"), "DT", "truck"),
    ("c-04", "7826010000100004", "g-long", "МАН TGX, А104АА77, Попов", ("Попов", "Андрей"), "DT", "truck"),
    ("c-05", "7826010000100005", "g-long", "Вольво FH, А105АА77, Васильев", ("Васильев", "Павел"), "DT", "truck"),
    ("c-06", "7826010000100006", "g-city", "Газель Next, В201ВВ77, Соколов", ("Соколов", "Денис"), "AI92", "van"),
    ("c-07", "7826010000100007", "g-city", "Газель Next, В202ВВ77, Михайлов", ("Михайлов", "Артём"), "AI92", "van"),
    ("c-08", "7826010000100008", "g-city", "Форд Транзит, В203ВВ77, Новиков", ("Новиков", "Роман"), "AI92", "van"),
    ("c-09", "7826010000100009", "g-city", "Газель Next, В204ВВ77, Фёдоров", ("Фёдоров", "Илья"), "AI92", "van"),
    ("c-10", "7826010000100010", "g-city", "Резерв, без водителя", None, "AI92", "van"),
]
GROUPS = {"g-long": "Магистраль", "g-city": "Город"}


class SyntheticOpti24:
    """Поддельный клиент Opti24 с синтетическим договором; интерфейс как у APIClient."""

    contract_id = DEMO_CONTRACT_ID
    synthetic = True

    def __init__(self, today: date | None = None, seed: int = 24) -> None:
        self.today = today or date.today()
        self._rng = random.Random(seed)
        self._next_id = 1
        self.cards_state: dict[str, dict[str, Any]] = {
            cid: {"number": number, "group": group, "comment": comment, "status": "Active", "driver": driver,
                  "fuel": fuel, "kind": kind, "pin_counter": 0}
            for cid, number, group, comment, driver, fuel, kind in CARDS
        }
        # Карта потеряна и заблокирована клиентом; у другой три неверных PIN подряд — для разбора отказа.
        self.cards_state["c-10"]["status"] = "Locked(Client)"
        self.cards_state["c-08"]["pin_counter"] = 3
        self.operations = self._generate_transactions()
        self.limit_rules: dict[str, Any] = {}
        self.region_rules: dict[str, Any] = {}
        self.restriction_rules: dict[str, Any] = {}
        self.report_jobs: list[Any] = []
        self._seed_rules()
        self.queries_used = 37

        self.auth = NS(auth_user=self._auth_user, get_info=self._get_info, logoff=self._logoff)
        self.contracts = NS(
            get_contract_data=self._contract_data,
            get_payments=self._payments,
            get_invoices=self._invoices,
            get_documents=self._documents,
        )
        self.card_groups = NS(get_card_groups=self._card_groups)
        self.cards = NS(
            get_cards_v2=self._cards_v2,
            get_card_detail=self._card_detail,
            get_card_drivers=self._card_drivers,
            block_card=self._block_card,
            set_card_comment=self._set_card_comment,
        )
        self.limits = NS(get_limits=self._get_limits, set_limit=self._set_limit, remove_limit=self._remove_limit)
        self.region_limits = NS(
            get_region_limits=self._get_region_limits,
            set_region_limit=self._set_region_limit,
            remove_region_limit=self._remove_region_limit,
        )
        self.restrictions = NS(
            get_restrictions=self._get_restrictions,
            set_restriction=self._set_restriction,
            remove_restriction=self._remove_restriction,
        )
        self.transactions = NS(
            get_transactions_v2=self._transactions_v2,
            get_card_transactions_v2=self._card_transactions_v2,
            get_transaction_detail=self._transaction_detail,
        )
        self.dictionaries = NS(get_dictionary=self._dictionary, get_azs_list_v1=self._azs_list)
        self.final_prices = NS(check_purchase=self._check_purchase)
        self.reports = NS(
            order_report_v1=self._order_report,
            get_report_job_list_v1=self._report_jobs,
            download_report_file_v1_to=self._download_report,
        )

    # --- генерация ---------------------------------------------------------------------------

    def _new_id(self, prefix: str) -> str:
        self._next_id += 1
        return f"{prefix}-{self._next_id:04d}"

    def _fill(self, card_id: str, when: datetime, station: Station, liters: Decimal) -> Any:
        fuel = str(self.cards_state[card_id]["fuel"])
        price = station.prices[fuel]
        discount = (liters * DISCOUNT_PER_LITER).quantize(Decimal("0.01"))
        amount = (price * liters - discount).quantize(Decimal("0.01"))
        return NS(
            id=self._new_id("t"),
            timestamp=when,
            card_id=card_id,
            card_number=self.cards_state[card_id]["number"],
            poi_id=station.id,
            terminal_id=f"{station.id}-1",
            check_id=str(self._next_id),
            type="P",
            product_name=GOODS[fuel],
            qty=float(liters),
            price=str(price),
            sum=str(amount),
            sum_no_discount=str((price * liters).quantize(Decimal("0.01"))),
            discount=str(discount),
            is_storno=False,
            is_manual_correction=False,
        )

    def _generate_transactions(self) -> list[Any]:
        rng = self._rng
        start = (self.today.replace(day=1) - timedelta(days=1)).replace(day=1)
        days = (self.today - start).days
        regular = [s for s in STATIONS if s.id != "poi-777"]
        items: list[Any] = []
        for cid, *_ in CARDS:
            state = self.cards_state[cid]
            if cid == "c-10":
                continue  # карта в резерве и заблокирована
            truck = state["kind"] == "truck"
            day = rng.randint(0, 2)
            while day < days:
                station = rng.choice(regular if truck else regular[:3])
                hour = rng.randint(5, 23) if truck else rng.randint(8, 19)
                when = datetime.combine(start + timedelta(days=day), time(hour, rng.choice([5, 20, 35, 50])))
                liters = Decimal(rng.randint(250, 480) if truck else rng.randint(35, 60))
                items.append(self._fill(cid, when, station, liters))
                day += rng.randint(2, 3) if truck else rng.randint(1, 2)
        # Заправки, которые требуют проверки: эвристика find_suspicious_fills должна их показать.
        mid = start + timedelta(days=min(days - 1, 20))
        items += [
            self._fill("c-06", datetime.combine(mid, time(2, 40)), regular[0], Decimal("58")),  # ночью
            self._fill("c-07", datetime.combine(mid, time(14, 10)), STATIONS[-1], Decimal("52")),  # дорогая АЗС
            self._fill("c-09", datetime.combine(mid, time(9, 15)), regular[1], Decimal("118")),  # больше бака фургона
            self._fill("c-09", datetime.combine(mid, time(12, 30)), regular[1], Decimal("45")),
            self._fill("c-09", datetime.combine(mid, time(17, 45)), regular[2], Decimal("47")),  # 3 раза за день
        ]
        storno = self._fill("c-02", datetime.combine(mid, time(11, 0)), regular[3], Decimal("300"))
        reverse = NS(**{**storno.__dict__, "id": self._new_id("t"), "is_storno": True,
                        "qty": -storno.qty, "sum": f"-{storno.sum}", "discount": f"-{storno.discount}",
                        "sum_no_discount": f"-{storno.sum_no_discount}"})
        items += [storno, reverse]
        return sorted(items, key=lambda t: t.timestamp)

    def _seed_rules(self) -> None:
        for cid, *_ in CARDS:
            truck = self.cards_state[cid]["kind"] == "truck"
            fuel = "DEMO-DT" if truck else "DEMO-AI92"
            lid = self._new_id("l")
            self.limit_rules[lid] = self._limit(lid, cid, None, fuel, liters=Decimal(1500) if truck else None,
                                           rubles=None if truck else Decimal(15000), period=(1, 7))
        rid = self._new_id("r")
        self.region_rules[rid] = NS(id=rid, card_id=None, group_id="g-city", country="RUS", region="77",
                                     service_center=None, partner=None, limit_type=1, date=str(self.today))
        rid = self._new_id("r")
        self.region_rules[rid] = NS(id=rid, card_id=None, group_id="g-city", country="RUS", region="50",
                                     service_center=None, partner=None, limit_type=1, date=str(self.today))
        sid = self._new_id("s")
        self.restriction_rules[sid] = NS(id=sid, card_id=None, group_id="g-long", productType="DEMO-WASH",
                                    productTypeName="Мойка", productGroup=None, productGroupName=None,
                                    restriction_type=2, date=str(self.today))

    def _week_used(self, card_id: str | None, group_id: str | None, liters: bool) -> Decimal:
        since = self.today - timedelta(days=self.today.weekday())
        total = Decimal(0)
        for t in self.operations:
            if t.timestamp.date() < since:
                continue
            state = self.cards_state[t.card_id]
            if (card_id and t.card_id != card_id) or (group_id and state["group"] != group_id):
                continue
            total += Decimal(str(t.qty)) if liters else Decimal(t.sum)
        return total

    def _limit(self, lid: str, card_id: str | None, group_id: str | None, product: str | None, *,
               liters: Decimal | None, rubles: Decimal | None, period: tuple[int, int],
               product_group: str | None = None, count: int | None = None) -> Any:
        amount = Dump(value=float(liters), used=float(self._week_used(card_id, group_id, True)), unit="LIT") \
            if liters is not None else None
        money = Dump(value=float(rubles), used=float(self._week_used(card_id, group_id, False)), currency="810") \
            if rubles is not None else None
        return NS(id=lid, card_id=card_id, group_id=group_id, productType=product, productGroup=product_group,
                  amount=amount, sum=money, term=None,
                  transactions=Dump(count=count) if count else None, time=NS(number=period[0], type=period[1]))

    # --- договор и доступ ----------------------------------------------------------------------

    async def _auth_user(self, **_: object) -> Any:
        contract = NS(id=DEMO_CONTRACT_ID, number=DEMO_CONTRACT_NUMBER, cards_count=len(CARDS))
        return NS(data=NS(contracts=[contract], org_name=DEMO_ORG, role_name="Администратор",
                          read_only=False, email="demo@example.invalid"))

    async def _get_info(self, **_: object) -> Any:
        first = self.today.replace(day=1)
        nxt = (first + timedelta(days=32)).replace(day=1)
        return NS(data=NS(from_=datetime.combine(first, time()), to=datetime.combine(nxt, time()),
                          client_info=NS(Queries=500), methods=NS(all=self.queries_used)))

    async def _logoff(self, **_: object) -> None:
        return None

    def _month_spent(self) -> tuple[Decimal, Decimal, Decimal]:
        first = self.today.replace(day=1)
        prev_first = (first - timedelta(days=1)).replace(day=1)
        month = [t for t in self.operations if t.timestamp.date() >= first]
        prev = [t for t in self.operations if prev_first <= t.timestamp.date() < first]
        spent = sum((Decimal(t.sum) for t in month), Decimal(0))
        liters = sum((Decimal(str(t.qty)) for t in month), Decimal(0))
        prev_liters = sum((Decimal(str(t.qty)) for t in prev), Decimal(0))
        return spent, liters, prev_liters

    async def _contract_data(self, **_: object) -> Any:
        spent, liters, prev_liters = self._month_spent()
        own = Decimal("1850000.00")
        active = sum(1 for c in self.cards_state.values() if c["status"] == "Active")
        return NS(data=NS(
            payment_term_id=None,
            payment_scheme_id=None,
            contractData=NS(contract_number=DEMO_CONTRACT_NUMBER, contract_status_name="Действует",
                            discount_scheme=None, balance_amount_limiter=None,
                            current_amount_limiter=str(own - spent), credit_limit=None),
            balanceData=NS(available_amount=str(own - spent), own_balance=str(own), balance=str(own - spent),
                           last_payment_sum="1500000.00", last_payment_date=f"{self.today.replace(day=3):%d.%m.%Y}",
                           currency="RUB",
                           consumption_for_month=str(spent), consumption_for_month_volume=str(liters),
                           consumption_for_prev_month_volume=str(prev_liters)),
            cardsData=NS(cards_quantity_all=str(len(CARDS)), cards_quantity_active=str(active),
                         card_groups_quantity_all=str(len(GROUPS))),
        ))

    async def _payments(self, **_: object) -> Any:
        first = self.today.replace(day=1)
        prev = (first - timedelta(days=1)).replace(day=1)
        items = [
            NS(id=f"p-{d:%Y%m}", date=f"{d + timedelta(days=2):%Y-%m-%d}T10:00:00", amount="1500000.00",
               amount_client="1500000.00", currency="810;RUR", description="Оплата по счёту за топливо",
               payment_name="Платёжное поручение", payment_number=f"{d:%m}01")
            for d in (prev, first)
        ]
        return NS(data=NS(result=items, total_count=len(items)))

    async def _invoices(self, **_: object) -> Any:
        first = self.today.replace(day=1)
        last = (first + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        item = NS(id="i-1", ref_number=f"СЧ-{first:%m}/ДЕМО", date_start=str(first), date_end=str(last),
                  amount="1500000.00", paid_amount="1500000.00", status="PAID", comment=None)
        return NS(data=NS(result=[item], total_count=1))

    async def _documents(self, **_: object) -> Any:
        prev_last = self.today.replace(day=1) - timedelta(days=1)
        stamp = int(datetime.combine(prev_last, time(12)).timestamp())
        item = NS(id="d-1", name="УПД", number=f"{prev_last:%m}-001", date=stamp, total=1480000.0,
                  vat=246666.67, sum=1233333.33, currency="руб.", consignee=DEMO_ORG)
        return NS(data=NS(result=[item], total_count=1))

    # --- карты ---------------------------------------------------------------------------------

    def _card(self, cid: str) -> Any:
        state = self.cards_state[cid]
        status = str(state["status"])
        return NS(id=cid, number=state["number"], status=status,
                  status_name="Активна" if status == "Active" else "Заблокирована",
                  group_id=state["group"], group_name=GROUPS[str(state["group"])], comment=state["comment"])

    async def _cards_v2(self, *, status: str | None = None, group_id: str | None = None, q: str | None = None,
                        page: int = 1, onpage: int = 100, **_: object) -> Any:
        items = [self._card(cid) for cid in self.cards_state]
        if status:
            items = [c for c in items if c.status == status]
        if group_id:
            items = [c for c in items if c.group_id == group_id]
        if q:
            needle = q.casefold()
            items = [c for c in items if needle in f"{c.number} {c.comment}".casefold()]
        start = (page - 1) * onpage
        return NS(data=NS(result=items[start:start + onpage], total_count=len(items)))

    async def _card_detail(self, *, card_id: str, **_: object) -> Any:
        if card_id not in self.cards_state:
            return NS(data=NS(result=[], total_count=0))
        state = self.cards_state[card_id]
        last = next((t for t in reversed(self.operations) if t.card_id == card_id), None)
        station = next((s.name for s in STATIONS if last and s.id == last.poi_id), None)
        item = NS(id=card_id, number=state["number"], status=state["status"], comment=state["comment"],
                  product="limit", carrier="Plastic", available=None, currency="RUB",
                  date_last_usage=last.timestamp if last else None, servicecenter_last_usage_name=station,
                  mpc=False, pin_reset=3, pin_counter=state["pin_counter"])
        return NS(data=NS(result=[item], total_count=1))

    async def _card_drivers(self, *, card_id: str, **_: object) -> Any:
        driver = self.cards_state.get(card_id, {}).get("driver")
        if not driver:
            return NS(data=NS(result=[], total_count=0))
        last, first = str(driver[0]), str(driver[1])
        item = NS(last_name=last, first_name=first, middle_name=None, position="Водитель", role="Driver",
                  mobile_phone=f"+7 900 000-{card_id[-2:]}-{card_id[-2:]}")
        return NS(data=NS(result=[item], total_count=1))

    async def _block_card(self, *, card_ids: list[str], block: bool, **_: object) -> Any:
        done = [cid for cid in card_ids if cid in self.cards_state]
        for cid in done:
            self.cards_state[cid]["status"] = "Locked(Client)" if block else "Active"
        self.queries_used += 1
        return NS(data=done)

    async def _set_card_comment(self, *, card_id: str, comment: str, **_: object) -> Any:
        if card_id not in self.cards_state:
            return NS(data=False)
        self.cards_state[card_id]["comment"] = comment
        self.queries_used += 1
        return NS(data=True)

    async def _card_groups(self, **_: object) -> Any:
        items = [NS(id=gid, name=name, cards_count=sum(1 for c in self.cards_state.values() if c["group"] == gid))
                 for gid, name in GROUPS.items()]
        return NS(data=NS(result=items, total_count=len(items)))

    # --- правила -------------------------------------------------------------------------------

    def _owned(self, rules: dict[str, Any], card_id: str | None, group_id: str | None) -> list[Any]:
        if card_id:
            group = self.cards_state.get(card_id, {}).get("group")
            return [r for r in rules.values() if r.card_id == card_id or (r.group_id and r.group_id == group)]
        if group_id:
            return [r for r in rules.values() if r.group_id == group_id]
        return list(rules.values())

    async def _get_limits(self, *, card_id: str | None = None, group_id: str | None = None, **_: object) -> Any:
        items = [r for r in self._owned(self.limit_rules, card_id, group_id) if not card_id or r.card_id == card_id]
        return NS(data=NS(result=items, total_count=len(items)))

    async def _set_limit(self, *, limits: list[Any], **_: object) -> Any:
        ids = []
        for item in limits:
            lid = item.id or self._new_id("l")
            self.limit_rules[lid] = self._limit(
                lid, item.card_id, item.group_id, item.product_type,
                liters=Decimal(str(item.amount.value)) if item.amount else None,
                rubles=Decimal(str(item.sum.value)) if item.sum else None,
                period=(item.time.number, item.time.type), product_group=item.product_group,
                count=item.transactions.count if item.transactions else None,
            )
            ids.append(lid)
        self.queries_used += 1
        return NS(data=ids)

    async def _remove_limit(self, *, limit_id: str, **_: object) -> Any:
        self.queries_used += 1
        return NS(data=self.limit_rules.pop(limit_id, None) is not None)

    async def _get_region_limits(self, *, card_id: str | None = None, group_id: str | None = None,
                                 **_: object) -> Any:
        items = self._owned(self.region_rules, card_id, group_id)
        self.queries_used += 1
        return NS(data=NS(result=items, total_count=len(items)))

    async def _set_region_limit(self, *, region_limits: list[Any], **_: object) -> Any:
        ids = []
        for item in region_limits:
            rid = item.id or self._new_id("r")
            self.region_rules[rid] = NS(id=rid, card_id=item.card_id, group_id=item.group_id, country=item.country,
                                         region=item.region, service_center=item.service_center,
                                         partner=item.partner, limit_type=item.limit_type, date=str(self.today))
            ids.append(rid)
        self.queries_used += 1
        return NS(data=ids)

    async def _remove_region_limit(self, *, regionlimit_id: str, **_: object) -> Any:
        self.queries_used += 1
        return NS(data=self.region_rules.pop(regionlimit_id, None) is not None)

    async def _get_restrictions(self, *, card_id: str | None = None, group_id: str | None = None,
                                **_: object) -> Any:
        items = self._owned(self.restriction_rules, card_id, group_id)
        self.queries_used += 1
        return NS(data=NS(result=items, total_count=len(items)))

    async def _set_restriction(self, *, restrictions: list[Any], **_: object) -> Any:
        ids = []
        for item in restrictions:
            sid = item.id or self._new_id("s")
            self.restriction_rules[sid] = NS(id=sid, card_id=item.card_id, group_id=item.group_id,
                                        productType=item.product_type,
                                        productTypeName=PRODUCT_TYPES.get(str(item.product_type)),
                                        productGroup=item.product_group, productGroupName=None,
                                        restriction_type=item.restriction_type, date=str(self.today))
            ids.append(sid)
        self.queries_used += 1
        return NS(data=ids)

    async def _remove_restriction(self, *, restriction_id: str, **_: object) -> Any:
        self.queries_used += 1
        return NS(data=self.restriction_rules.pop(restriction_id, None) is not None)

    # --- операции, справочники, отчёты ---------------------------------------------------------

    def _period(self, date_from: object, date_to: object, card_id: str | None = None) -> list[Any]:
        lo, hi = str(date_from)[:10], str(date_to)[:10]
        return [t for t in self.operations
                if lo <= t.timestamp.date().isoformat() <= hi and (card_id is None or t.card_id == card_id)]

    async def _transactions_v2(self, *, date_from: object, date_to: object, page_limit: int = 100,
                               page_offset: int = 0, **_: object) -> Any:
        items = self._period(date_from, date_to)
        self.queries_used += 1
        return NS(data=NS(result=items[page_offset:page_offset + page_limit], total_count=len(items)))

    async def _card_transactions_v2(self, *, card_id: str, date_from: object, date_to: object,
                                    page_limit: int = 100, page_offset: int = 0, **_: object) -> Any:
        items = self._period(date_from, date_to, card_id)
        self.queries_used += 1
        return NS(data=NS(result=items[page_offset:page_offset + page_limit], total_count=len(items)))

    async def _transaction_detail(self, *, transaction_id: object, **_: object) -> Any:
        items = [t for t in self.operations if str(t.id) == str(transaction_id)]
        return NS(data=NS(result=items, total_count=len(items)))

    async def _dictionary(self, *, name: str, **_: object) -> Any:
        table = {"ProductType": PRODUCT_TYPES, "Goods": GOODS, "Region": REGIONS}.get(name, {})
        items = [NS(id=code, code=code, name=label, value=None) for code, label in table.items()]
        return NS(data=NS(result=items, total_count=len(items)))

    async def _azs_list(self, *, id: str | None = None, q: str | None = None, **_: object) -> Any:  # noqa: A002
        if id:
            found = [s for s in STATIONS if s.id == id]
        else:
            needle = str(q or "").casefold()
            found = [s for s in STATIONS if needle and (needle in s.name.casefold() or needle in s.id)]
        items = [NS(id=s.id, name=s.name, regionCode=s.region,
                    prices=[NS(GoodsCode=code, Price=str(price)) for code, price in s.prices.items()])
                 for s in found]
        return NS(data=NS(result=items, total_count=len(items)))

    async def _check_purchase(self, *, card_id: str, **_: object) -> Any:
        state = self.cards_state.get(card_id, {})
        return NS(data=bool(state) and state["status"] == "Active" and state["pin_counter"] < 3)

    async def _order_report(self, **kwargs: object) -> Any:
        job = NS(job_id=self._new_id("job"), date=str(self.today), report_name="Транзакционный отчёт (демо)",
                 report_format=str(kwargs.get("report_format") or "csv"), contract_id=DEMO_CONTRACT_ID,
                 client_id="demo", user_id="demo", period=(kwargs.get("start"), kwargs.get("end")))
        self.report_jobs.append(job)
        self.queries_used += 1
        return NS(data=[job.job_id])

    async def _report_jobs(self, **_: object) -> Any:
        return NS(data=list(self.report_jobs))

    async def _download_report(self, *, job_id: str, destination: Path) -> Path:
        job = next((r for r in self.report_jobs if r.job_id == job_id), None)
        rows = self._period(*job.period) if job else []
        destination.parent.mkdir(parents=True, exist_ok=True)
        lines = ["дата;карта;АЗС;топливо;литры;цена;сумма"] + [
            f"{t.timestamp:%d.%m.%Y %H:%M};{t.card_number};{t.poi_id};{t.product_name};{t.qty};{t.price};{t.sum}"
            for t in rows
        ]
        destination.with_suffix(".csv").write_text("\n".join(lines), encoding="utf-8")
        self.queries_used += 1
        return destination.with_suffix(".csv")


DEMO_BANNER = (
    "ДЕМО: синтетический договор «ДЕМО-2026/01», данные выдуманы, запросов к Opti24 нет. "
    "Изменения живут только в памяти этого запуска."
)


@asynccontextmanager
async def demo_client() -> AsyncIterator[APIClient]:
    """Клиент демо-режима в виде контекста, как APIClient: вход и выход ничего не делают в сети.

    Приведение типа нужно только для агента и дашборда: они принимают APIClient, а методы здесь те же.
    """
    yield cast(APIClient, SyntheticOpti24())
