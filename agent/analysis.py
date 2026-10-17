"""Расчёты по операциям — обычный Python, без модели.

Главный приём экономии токенов: модель не читает сотни операций и не считает
их в уме. Код агрегирует данные и отдаёт модели короткую сводку, а модель
решает, что с ней делать и как объяснить человеку.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from statistics import median
from typing import Protocol

from .money import rub, to_decimal

PURCHASE = "P"


class Transaction(Protocol):
    """Поля TransactionItemV2 из apisdkopti24, которые нужны расчётам."""

    id: int | str
    timestamp: datetime
    card_id: str
    poi_id: str
    type: str
    product_name: str
    qty: int | float
    price: float | str
    sum: float | str
    is_storno: bool


@dataclass(frozen=True, slots=True)
class AnomalyRules:
    price_tolerance: Decimal = Decimal("0.07")  # дороже медианы по топливу на 7 %
    max_fills_per_day: int = 2
    max_liters_per_fill: Decimal = Decimal("150")
    night_hours: range = range(0, 5)  # 00:00–04:59 по местному времени АЗС


def _purchases(transactions: Iterable[Transaction]) -> list[Transaction]:
    return [t for t in transactions if t.type == PURCHASE and not t.is_storno]


def summarize(transactions: Iterable[Transaction], top: int = 5) -> dict[str, object]:
    items = list(transactions)
    purchases = _purchases(items)
    total = sum((to_decimal(t.sum) for t in purchases), Decimal("0"))

    by_product: dict[str, list[Decimal]] = defaultdict(lambda: [Decimal("0"), Decimal("0")])
    by_card: Counter[str] = Counter()
    card_sums: dict[str, Decimal] = defaultdict(Decimal)
    for t in purchases:
        acc = by_product[t.product_name]
        acc[0] += to_decimal(t.qty)
        acc[1] += to_decimal(t.sum)
        by_card[t.card_id] += 1
        card_sums[t.card_id] += to_decimal(t.sum)

    products = [
        {
            "product": name,
            "liters": str(liters),
            "sum": rub(amount),
            "avg_price_per_liter": rub(amount / liters) if liters else None,
        }
        for name, (liters, amount) in sorted(by_product.items(), key=lambda kv: -kv[1][1])
    ]
    top_cards = [
        {"card_id": card_id, "operations": by_card[card_id], "sum": rub(amount)}
        for card_id, amount in sorted(card_sums.items(), key=lambda kv: -kv[1])[:top]
    ]
    return {
        "operations": len(purchases),
        "refunds_and_storno": len(items) - len(purchases),
        "total": rub(total),
        "by_product": products,
        "top_cards": top_cards,
    }


def charts(transactions: Iterable[Transaction], recent: int = 10) -> dict[str, object]:
    """Данные для графиков: расход по дням, карта «день недели × час», последние операции."""
    purchases = _purchases(transactions)
    daily: dict[str, list[Decimal]] = defaultdict(lambda: [Decimal("0"), Decimal("0")])
    heat = [[0] * 24 for _ in range(7)]
    for t in purchases:
        acc = daily[t.timestamp.date().isoformat()]
        acc[0] += to_decimal(t.sum)
        acc[1] += to_decimal(t.qty)
        heat[t.timestamp.weekday()][t.timestamp.hour] += 1
    last = sorted(purchases, key=lambda t: t.timestamp, reverse=True)[:recent]
    return {
        "daily": [
            {"date": day, "sum": float(v[0]), "liters": float(v[1])} for day, v in sorted(daily.items())
        ],
        "heatmap": heat,
        "recent": [
            {
                "time": f"{t.timestamp:%d.%m %H:%M}",
                "card": t.card_id,
                "product": t.product_name,
                "liters": str(t.qty),
                "price": rub(to_decimal(t.price)),
                "sum": rub(to_decimal(t.sum)),
            }
            for t in last
        ],
    }


def find_anomalies(
    transactions: Iterable[Transaction], rules: AnomalyRules | None = None, top: int = 10
) -> dict[str, object]:
    rules = rules or AnomalyRules()
    purchases = _purchases(transactions)

    prices: dict[str, list[Decimal]] = defaultdict(list)
    for t in purchases:
        prices[t.product_name].append(to_decimal(t.price))
    medians = {name: Decimal(median(values)) for name, values in prices.items()}

    fills_per_day: Counter[tuple[str, str]] = Counter(
        (t.card_id, t.timestamp.date().isoformat()) for t in purchases
    )

    findings: list[dict[str, object]] = []
    for t in purchases:
        reasons: list[str] = []
        price = to_decimal(t.price)
        limit = medians[t.product_name] * (1 + rules.price_tolerance)
        if price > limit:
            reasons.append(
                f"цена {rub(price)}/л выше медианы {rub(medians[t.product_name])}/л"
            )
        if to_decimal(t.qty) > rules.max_liters_per_fill:
            reasons.append(f"объём {t.qty} л за одну заправку")
        if t.timestamp.hour in rules.night_hours:
            reasons.append(f"ночная заправка в {t.timestamp:%H:%M}")
        day = t.timestamp.date().isoformat()
        if fills_per_day[(t.card_id, day)] > rules.max_fills_per_day:
            reasons.append(f"{fills_per_day[(t.card_id, day)]} заправки по карте за день")
        if reasons:
            findings.append(
                {
                    "transaction_id": str(t.id),
                    "card_id": t.card_id,
                    "date": f"{t.timestamp:%Y-%m-%d %H:%M}",
                    "product": t.product_name,
                    "liters": str(t.qty),
                    "sum": rub(to_decimal(t.sum)),
                    "reasons": reasons,
                }
            )

    findings.sort(key=lambda f: -len(f["reasons"]))  # type: ignore[arg-type]
    return {
        "checked_operations": len(purchases),
        "suspicious_total": len(findings),
        "shown": findings[:top],
        "rules": {
            "price_tolerance_percent": str(rules.price_tolerance * 100),
            "max_fills_per_day": rules.max_fills_per_day,
            "max_liters_per_fill": str(rules.max_liters_per_fill),
            "night_hours": f"{rules.night_hours.start:02d}:00–{rules.night_hours.stop - 1:02d}:59",
        },
    }


UNKNOWN_REGION = "регион не определён"


@dataclass(slots=True)
class _PriceAcc:
    liters: Decimal = Decimal("0")
    paid: Decimal = Decimal("0")  # списано с договора: со скидкой и наценками в цене операции
    station: Decimal = Decimal("0")  # по цене АЗС без скидки
    operations: int = 0

    def add(self, t: Transaction) -> None:
        self.liters += to_decimal(t.qty)
        self.paid += to_decimal(t.sum)
        # sum_no_discount есть у операций v2; если его нет, считаем, что скидки не было.
        self.station += to_decimal(getattr(t, "sum_no_discount", None) or t.sum)
        self.operations += 1

    def client_price(self) -> Decimal | None:
        return self.paid / self.liters if self.liters else None

    def payload(self) -> dict[str, object]:
        client = self.client_price()
        station = self.station / self.liters if self.liters else None
        benefit = self.station - self.paid
        return {
            "operations": self.operations,
            "liters": str(self.liters),
            "paid": rub(self.paid),
            "client_price_per_liter": rub(client) if client is not None else None,
            "station_price_per_liter": rub(station) if station is not None else None,
            # Положительное — скидка, отрицательное — цена выше цены АЗС (наценка в операции).
            "discount_total": rub(benefit),
            "discount_per_liter": rub(benefit / self.liters) if self.liters else None,
        }


def fuel_price_analytics(
    transactions: Iterable[Transaction],
    region_of: dict[str, str],
    min_operations: int = 3,
    top: int = 10,
) -> dict[str, object]:
    """Цена топлива для клиента по продуктам и регионам и где заправляться дешевле.

    Цена клиента — списанная сумма, делённая на литры: в неё входят скидка и
    любые наценки, которые API провёл в самой операции. Сборы, выставленные
    отдельно (счётом или документом), в операциях не видны и сюда не входят.
    ``region_of`` сопоставляет ID АЗС с названием региона.
    """
    purchases = _purchases(transactions)
    by_product: dict[str, _PriceAcc] = defaultdict(_PriceAcc)
    by_region: dict[str, _PriceAcc] = defaultdict(_PriceAcc)
    by_pair: dict[tuple[str, str], _PriceAcc] = defaultdict(_PriceAcc)
    for t in purchases:
        region = region_of.get(t.poi_id) or UNKNOWN_REGION
        by_product[t.product_name].add(t)
        by_region[region].add(t)
        by_pair[(t.product_name, region)].add(t)

    matrix: list[dict[str, object]] = []
    savings: list[tuple[Decimal, dict[str, object]]] = []
    for product in sorted(by_product, key=lambda p: -by_product[p].liters):
        rows = sorted(
            (
                (region, acc)
                for (name, region), acc in by_pair.items()
                if name == product and region != UNKNOWN_REGION
            ),
            key=lambda row: row[1].client_price() or Decimal("0"),
        )
        matrix.append(
            {"product": product, "regions": [{"region": r, **acc.payload()} for r, acc in rows[:top]]}
        )
        # Сравниваем только регионы, где операций достаточно, чтобы средняя цена что-то значила.
        reliable = [(r, acc) for r, acc in rows if acc.operations >= min_operations]
        if len(reliable) < 2:
            continue
        best_region, best = reliable[0]
        best_price = best.client_price() or Decimal("0")
        for region, acc in reliable[1:]:
            diff = (acc.client_price() or Decimal("0")) - best_price
            if diff <= 0:
                continue
            saving = diff * acc.liters
            savings.append(
                (
                    saving,
                    {
                        "product": product,
                        "expensive_region": region,
                        "expensive_price_per_liter": rub(acc.client_price() or Decimal("0")),
                        "cheaper_region": best_region,
                        "cheaper_price_per_liter": rub(best_price),
                        "difference_per_liter": rub(diff),
                        "liters_in_expensive_region": str(acc.liters),
                        "possible_saving": rub(saving),
                    },
                )
            )

    savings.sort(key=lambda pair: -pair[0])
    regions = sorted(by_region.items(), key=lambda kv: -kv[1].liters)
    return {
        "operations": len(purchases),
        "by_product": [{"product": p, **by_product[p].payload()} for p in by_product],
        "by_region": [{"region": r, **acc.payload()} for r, acc in regions[:top]],
        "regions_total": len(regions),
        "price_by_product_and_region": matrix,
        "cheaper_options": [entry for _, entry in savings[:5]],
        "unknown_region_operations": by_region[UNKNOWN_REGION].operations
        if UNKNOWN_REGION in by_region
        else 0,
        "method": {
            "client_price": "списанная сумма / литры: со скидкой и наценками в операции",
            "station_price": "сумма без скидки / литры",
            "not_included": "сборы, выставленные отдельно от операций",
            "comparison": f"сравниваются регионы, где по топливу не меньше {min_operations} операций; "
            "possible_saving = разница цены × литры, залитые в более дорогом регионе за период",
        },
    }
