"""Разбор транзакционного отчёта Opti24 (CSV) — обычный Python, без модели.

В операциях API сервисного сбора и цены стелы нет, а в транзакционном отчёте они
есть в каждой строке. Код читает файл, считает суммы и отдаёт модели короткую
сводку: модель не читает тысячи строк и не считает сбор сама.

Формат отчёта «Транзакционный» (CSV, UTF-8 с BOM, запятая — разделитель полей,
десятичная запятая в числах). Смысл колонок проверен на выгрузке:
- «Сумма к оплате» = «Сумма без скидки» − «Сумма скидки»; отрицательная скидка —
  наценка к цене стелы;
- «Сумма сервисного сбора» отрицательная (списана с клиента) и в «Сумму к оплате»
  не входит: клиент платит к оплате плюс сбор;
- возврат и сторно приходят с отрицательными литрами и суммой.
"""

from __future__ import annotations

import csv
import io
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .analysis import AnomalyRules
from .money import rub

PURCHASE = "Покупка"
LITERS = "Литры"
UNKNOWN_REGION = "Не определено"

REQUIRED_COLUMNS = (
    "Номер карты",
    "Время транзакции",
    "Тип транзакции",
    "Принадлежность АЗС",
    "Товар",
    "Количество",
    "Единица измерения",
    "Сумма без скидки",
    "Сумма скидки",
    "Сумма сервисного сбора",
    "Сумма к оплате",
)


class ReportFormatError(ValueError):
    """Файл не похож на транзакционный отчёт: нет нужных колонок или чисел."""


def _number(raw: str | None, column: str, line: int) -> Decimal:
    text = (raw or "").strip().replace(" ", "").replace(" ", "").replace(",", ".")
    if not text:
        return Decimal("0")
    try:
        return Decimal(text)
    except InvalidOperation:
        raise ReportFormatError(f"строка {line}: в колонке «{column}» не число: {text[:20]!r}") from None


@dataclass(frozen=True, slots=True)
class ReportRow:  # pylint: disable=too-many-instance-attributes
    """Одна строка отчёта: только колонки, которые нужны сводке."""

    card: str
    comment: str
    group: str
    moment: datetime
    kind: str
    owner_type: str
    station: str
    address: str
    region: str
    product: str
    quantity: Decimal
    liters: bool
    station_sum: Decimal  # по цене стелы, без скидки
    discount: Decimal  # отрицательная — наценка
    fee: Decimal  # сбор, списанный с клиента: положительный, как и списанная сумма
    paid: Decimal  # к оплате, без сбора


def _row(raw: Mapping[str, str], line: int) -> ReportRow:
    try:
        moment = datetime.fromisoformat((raw.get("Время транзакции") or "").strip())
    except ValueError:
        raise ReportFormatError(f"строка {line}: непонятное время операции") from None
    return ReportRow(
        card=(raw.get("Номер карты") or "").strip(),
        comment=(raw.get("Комментарий клиента") or "").strip(),
        group=(raw.get("Группа карт") or "").strip(),
        moment=moment,
        kind=(raw.get("Тип транзакции") or "").strip(),
        owner_type=(raw.get("Принадлежность АЗС") or "").strip() or "не указано",
        station=(raw.get("АЗС") or "").strip(),
        address=(raw.get("Адрес АЗС") or "").strip(),
        region=(raw.get("Регион АЗС") or "").strip() or UNKNOWN_REGION,
        product=(raw.get("Товар") or "").strip(),
        quantity=_number(raw.get("Количество"), "Количество", line),
        liters=(raw.get("Единица измерения") or "").strip() == LITERS,
        station_sum=_number(raw.get("Сумма без скидки"), "Сумма без скидки", line),
        discount=_number(raw.get("Сумма скидки"), "Сумма скидки", line),
        # В отчёте сбор со знаком минус: это списание. Храним как положительную сумму.
        fee=-_number(raw.get("Сумма сервисного сбора"), "Сумма сервисного сбора", line),
        paid=_number(raw.get("Сумма к оплате"), "Сумма к оплате", line),
    )


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1251")  # выгрузки из Excel на Windows


def read_rows(data: bytes) -> list[ReportRow]:
    """Строки отчёта из байтов файла; ReportFormatError, если это не транзакционный отчёт."""
    reader = csv.DictReader(io.StringIO(_decode(data)))
    missing = [name for name in REQUIRED_COLUMNS if name not in (reader.fieldnames or [])]
    if missing:
        raise ReportFormatError("нет колонок транзакционного отчёта: " + ", ".join(missing))
    return [_row(raw, line) for line, raw in enumerate(reader, start=2)]


@dataclass(slots=True)
class _Acc:
    operations: int = 0
    liters: Decimal = Decimal("0")
    station_sum: Decimal = Decimal("0")
    discount: Decimal = Decimal("0")
    fee: Decimal = Decimal("0")
    paid: Decimal = Decimal("0")

    def add(self, row: ReportRow) -> None:
        """Добавить строку в итог."""
        self.operations += 1
        if row.liters:
            self.liters += row.quantity
        self.station_sum += row.station_sum
        self.discount += row.discount
        self.fee += row.fee
        self.paid += row.paid

    def payload(self) -> dict[str, object]:
        """Итог для ответа модели: суммы в рублях, цены за литр."""
        out: dict[str, object] = {
            "operations": self.operations,
            "liters": str(self.liters),
            "paid": rub(self.paid),
            "service_fee": rub(self.fee),
            "paid_with_fee": rub(self.paid + self.fee),
            "discount": rub(self.discount),
        }
        if self.paid:
            out["service_fee_percent"] = str((self.fee / self.paid * 100).quantize(Decimal("0.01")))
        if self.liters:
            out["price_per_liter"] = rub(self.paid / self.liters)
            out["price_with_fee_per_liter"] = rub((self.paid + self.fee) / self.liters)
            out["station_price_per_liter"] = rub(self.station_sum / self.liters)
            out["service_fee_per_liter"] = rub(self.fee / self.liters)
        return out


def _grouped(rows: Iterable[ReportRow], key: str, top: int) -> tuple[list[dict[str, object]], int]:
    accs: dict[str, _Acc] = defaultdict(_Acc)
    for row in rows:
        accs[str(getattr(row, key))].add(row)
    ordered = sorted(accs.items(), key=lambda kv: -kv[1].paid)
    return [{key: name, **acc.payload()} for name, acc in ordered[:top]], len(ordered)


def _example(row: ReportRow, reason: str) -> dict[str, object]:
    return {
        "card_number": row.card,
        "card_comment": row.comment,
        "date": f"{row.moment:%Y-%m-%d %H:%M}",
        "station": row.station,
        "address": row.address,
        "product": row.product,
        "liters": str(row.quantity),
        "paid": rub(row.paid),
        "reason": reason,
    }


def _signals(fuel: list[ReportRow], goods: list[ReportRow], rules: AnomalyRules, examples: int) -> dict[str, object]:
    per_day: Counter[tuple[str, str]] = Counter((r.card, r.moment.date().isoformat()) for r in fuel)
    large = sorted((r for r in fuel if r.quantity > rules.max_liters_per_fill), key=lambda r: -r.quantity)
    night = [r for r in fuel if r.moment.hour in rules.night_hours]
    busy = sorted(
        ((card, day, count) for (card, day), count in per_day.items() if count >= rules.max_fills_per_day),
        key=lambda item: -item[2],
    )
    comments = {r.card: r.comment for r in fuel}
    return {
        "large_fills": {
            "count": len(large),
            "examples": [_example(r, f"{r.quantity} л за одну заправку") for r in large[:examples]],
        },
        "night_fills": {
            "count": len(night),
            "examples": [_example(r, f"ночная заправка в {r.moment:%H:%M}") for r in night[:examples]],
        },
        "many_fills_per_day": {
            "count": len(busy),
            "examples": [
                {"card_number": card, "card_comment": comments.get(card, ""), "date": day, "fills": count}
                for card, day, count in busy[:examples]
            ],
        },
        "non_fuel_goods": {
            "count": len(goods),
            "paid": rub(sum((r.paid for r in goods), Decimal("0"))),
            "examples": [_example(r, "нетопливный товар") for r in goods[:examples]],
        },
        "rules": {
            "max_liters_per_fill": str(rules.max_liters_per_fill),
            "liters_rule": "требует проверки больше этого объёма за одну заправку",
            "max_fills_per_day": rules.max_fills_per_day,
            "fills_per_day_rule": "порог достигается при этом количестве заправок за день и больше",
            "night_hours": f"{rules.night_hours.start:02d}:00–{rules.night_hours.stop - 1:02d}:59",
            "note": "Эвристика, а не доказательство: для грузовиков крупные и ночные заправки часто нормальны.",
        },
    }


def _markups(rows: list[ReportRow], top: int) -> dict[str, object]:
    """Операции, где клиент заплатил больше цены стелы (отрицательная скидка)."""
    marked = [r for r in rows if r.kind == PURCHASE and r.discount < 0]
    by_station: dict[tuple[str, str, str], Decimal] = defaultdict(Decimal)
    for r in marked:
        by_station[(r.owner_type, r.station, r.address)] += -r.discount
    ordered = sorted(by_station.items(), key=lambda kv: -kv[1])
    return {
        "operations": len(marked),
        "overpaid": rub(sum((-r.discount for r in marked), Decimal("0"))),
        "top_stations": [
            {"owner_type": owner, "station": station, "address": address, "overpaid": rub(amount)}
            for (owner, station, address), amount in ordered[:top]
        ],
    }


def _totals(rows: list[ReportRow]) -> dict[str, object]:
    total = _Acc()
    returns = _Acc()
    for row in rows:
        total.add(row)
        if row.kind != PURCHASE:
            returns.add(row)
    return {"total": total.payload(), "returns_and_storno": returns.payload()}


def _breakdowns(rows: list[ReportRow], top: int) -> tuple[dict[str, object], list[str]]:
    liters_rows = [r for r in rows if r.liters]
    products, products_total = _grouped(liters_rows, "product", 8)
    owners, _ = _grouped(liters_rows, "owner_type", 8)
    regions, regions_total = _grouped(liters_rows, "region", 8)
    cards, cards_total = _grouped(rows, "card", top)
    groups, groups_total = _grouped(rows, "group", top)
    comments = {r.card: r.comment for r in rows}
    for card in cards:
        card["card_comment"] = comments.get(str(card["card"]), "")
    hints: list[str] = []
    if products_total > len(products) or regions_total > len(regions):
        hints.append("В разрезах по топливу и регионам показаны первые 8 по сумме.")
    if unknown := sum(1 for r in liters_rows if r.region == UNKNOWN_REGION):
        hints.append(f"У {unknown} операций регион «{UNKNOWN_REGION}».")
    return {
        "cards": cards_total,
        "groups": groups_total,
        "by_month": _by_month(rows),
        "by_product": products,
        "by_station_owner": owners,
        "by_region": regions,
        "top_cards": cards,
        "top_groups": groups,
    }, hints


def summarize_report(rows: list[ReportRow], rules: AnomalyRules | None = None, top: int = 5) -> dict[str, object]:
    """Сводка отчёта: итог со сбором, разрезы, наценки, возвраты и сигналы для проверки."""
    if not rows:
        return {"rows": 0, "empty": "В отчёте нет операций. Так и скажи."}
    fuel_purchases = [r for r in rows if r.liters and r.kind == PURCHASE]
    goods = [r for r in rows if not r.liters and r.kind == PURCHASE]
    breakdowns, hints = _breakdowns(rows, top)
    result: dict[str, object] = {
        "period": f"{min(r.moment for r in rows):%Y-%m-%d}…{max(r.moment for r in rows):%Y-%m-%d}",
        "rows": len(rows),
        **_totals(rows),
        **breakdowns,
        "markup_over_station_price": _markups(rows, top),
        "signals": _signals(fuel_purchases, goods, rules or AnomalyRules(), examples=3),
        "fields": (
            "paid — «Сумма к оплате» без сбора; service_fee — «Сумма сервисного сбора» из отчёта, "
            "показан положительным; paid_with_fee = paid + service_fee; discount < 0 — наценка к цене стелы; "
            "цены за литр — суммы, делённые на литры. Всё посчитано кодом из файла отчёта."
        ),
    }
    if hints:
        result["hint"] = " ".join(hints)
    return result


def _by_month(rows: list[ReportRow]) -> list[dict[str, object]]:
    accs: dict[str, _Acc] = defaultdict(_Acc)
    for row in rows:
        accs[f"{row.moment:%Y-%m}"].add(row)
    return [{"month": month, **accs[month].payload()} for month in sorted(accs)]


def summarize_file(path: Path, rules: AnomalyRules | None = None) -> dict[str, object]:
    """Сводка по файлу отчёта csv."""
    return summarize_report(read_rows(path.read_bytes()), rules)
