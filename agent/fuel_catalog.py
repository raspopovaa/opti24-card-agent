"""Каталог топлива: один вид топлива — два ID API и слова, которыми его называют клиенты.

У вида топлива в API два ID без общего ключа:
- ``goods`` — код справочника Goods: цены АЗС (``prices[].GoodsCode``), фильтры АЗС,
  ``check_purchase`` и ``get_final_prices``;
- ``product_group`` — ID справочника ProductGroup: лимиты, товарные ограничители, операции.
  Категория у всего топлива одна: ProductType ``1-CK231`` «Топливо».

Соответствие составлено по названиям в справочниках (выгрузка 2026-11-08) и сверяется с
живыми справочниками во время работы (``check_against``). Каталог нужен, чтобы ID для
изменений выбирал код, а не модель: «солярка» всегда ДТ ``1-CK237``, а не «ДТ Премиум».
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

FUEL_PRODUCT_TYPE = "1-CK231"


@dataclass(frozen=True, slots=True)
class FuelKind:
    name: str
    goods: str | None
    product_group: str | None
    aliases: tuple[str, ...] = ()


FUELS: tuple[FuelKind, ...] = (
    FuelKind("АИ-80", "00000000000001", "1-CK232", ("80",)),
    FuelKind("АИ-92", "00000000000004", "1-CK235", ("92", "девяносто второй")),
    FuelKind("АИ-92 Премиум", "00000000000054", "1-35FMHF5", ("92 премиум",)),
    FuelKind("АИ-92 ОПТИ", "00000000000024", None, ("92 опти",)),
    FuelKind("АИ-93", "00000000000040", None, ("93",)),
    FuelKind("АИ-95", "00000000000003", "1-CK234", ("95", "девяносто пятый")),
    FuelKind("АИ-95 Премиум", "00000000000053", "1-35FMHF7", ("95 премиум",)),
    FuelKind("АИ-95 ОПТИ", "00000000000023", None, ("95 опти",)),
    FuelKind("АИ-98", "00000000000002", "1-CK233", ("98",)),
    FuelKind("АИ-98 Премиум", None, "1-35FWIX5", ("98 премиум",)),
    FuelKind("АИ-100", "00000000000019", "1-5A07K7A", ("100",)),
    FuelKind("G-92", "00000000000011", "1-CK23B", ("g92", "джи 92")),
    FuelKind("G-95", "00000000000007", "1-CK238", ("g95", "джи 95")),
    FuelKind("G-98", "00000000000009", "1-CK239", ("g98",)),
    FuelKind("G-100", "00000000000017", "1-50GTO2W", ("g100",)),
    FuelKind("G-100 с присадкой", "00000000000018", None, ("g100 с присадкой",)),
    FuelKind("ДТ", "00000000000006", "1-CK237", ("дт", "дизель", "дизтопливо", "дизельное топливо", "солярка", "соляра")),
    FuelKind("ДТ Премиум", "00000000000056", "1-VF193", ("дт премиум", "дизель премиум")),
    FuelKind("ДТ Зимнее", "00000000000021", None, ("дт зимнее", "зимний дизель", "зимнее дт")),
    FuelKind("ДТ ОПТИ", "00000000000008", None, ("дт опти",)),
    FuelKind("G-ДТ", "00000000000010", "1-CK23A", ("g-дт", "gдт", "джи дт")),
    FuelKind("СУГ", "00000000000005", "1-CK236", ("суг", "пропан", "сжиженный газ", "газ пропан")),
    FuelKind("КПГ", "00000000000012", "1-CK23C", ("кпг", "метан", "компримированный газ")),
    FuelKind("AdBlue", "9900000050678", None, ("adblue", "адблю", "мочевина")),
)


def _norm(text: str) -> str:
    text = text.casefold().replace("ё", "е")
    text = re.sub(r"\b(аи|ai|бензин|топливо)\b", " ", text)
    text = re.sub(r"[-\s_]+", " ", text)
    text = text.replace("-й", "").replace(" й", "")
    return re.sub(r"(\d+)\s*(й|ый|ой)\b", r"\1", text).strip()


def _keys(kind: FuelKind) -> set[str]:
    return {_norm(kind.name), *(_norm(alias) for alias in kind.aliases)}


def resolve(text: str | None) -> FuelKind | None:
    """Вид топлива по слову клиента, коду Goods или ID ProductGroup; None — не узнали."""
    if not text or not text.strip():
        return None
    raw = text.strip()
    for kind in FUELS:
        if raw in (kind.goods, kind.product_group):
            return kind
    key = _norm(raw)
    for kind in FUELS:
        if key in _keys(kind):
            return kind
    return None


def is_api_id(value: str) -> bool:
    """Похоже на ID API (``1-…`` или код Goods), а не на слово клиента."""
    return bool(re.fullmatch(r"1-[0-9A-Z]+|\d{13,14}", value.strip()))


def check_against(goods_ids: Iterable[str], group_ids: Iterable[str]) -> list[str]:
    """ID каталога, которых нет в живых справочниках Goods и ProductGroup.

    Пустой справочник (недоступен) не проверяется. Непустой список — повод обновить каталог.
    """
    goods, groups = set(goods_ids), set(group_ids)
    missing: list[str] = []
    for kind in FUELS:
        if goods and kind.goods and kind.goods not in goods:
            missing.append(f"{kind.name}: Goods {kind.goods}")
        if groups and kind.product_group and kind.product_group not in groups:
            missing.append(f"{kind.name}: ProductGroup {kind.product_group}")
    return missing
