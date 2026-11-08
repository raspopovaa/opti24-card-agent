"""Каталог топлива: слово клиента → ID API и сверка со справочниками."""

import pytest

from agent import fuel_catalog
from agent.fuel_catalog import check_against, resolve


@pytest.mark.parametrize(
    ("word", "name"),
    [
        ("солярка", "ДТ"), ("Дизель", "ДТ"), ("дт", "ДТ"), ("ДТ Премиум", "ДТ Премиум"),
        ("92-й", "АИ-92"), ("бензин 92", "АИ-92"), ("АИ-95", "АИ-95"), ("95й", "АИ-95"),
        ("пропан", "СУГ"), ("метан", "КПГ"), ("G-95", "G-95"),
        ("1-CK237", "ДТ"), ("00000000000006", "ДТ"),
    ],
)
def test_resolve_known_words_and_ids(word: str, name: str) -> None:
    kind = resolve(word)
    assert kind is not None and kind.name == name


@pytest.mark.parametrize("word", ["газ", "", "   ", None, "кофе"])
def test_resolve_ambiguous_or_unknown_returns_none(word: str | None) -> None:
    assert resolve(word) is None


def test_catalog_ids_are_unique() -> None:
    goods = [k.goods for k in fuel_catalog.FUELS if k.goods]
    groups = [k.product_group for k in fuel_catalog.FUELS if k.product_group]
    assert len(goods) == len(set(goods)) and len(groups) == len(set(groups))


def test_check_against_reports_missing_ids_and_skips_empty_dictionary() -> None:
    goods = [k.goods for k in fuel_catalog.FUELS if k.goods and k.name != "ДТ"]
    groups = [k.product_group for k in fuel_catalog.FUELS if k.product_group]
    assert check_against(goods, groups) == ["ДТ: Goods 00000000000006"]
    assert check_against([], []) == []


def test_is_api_id() -> None:
    assert fuel_catalog.is_api_id("1-CK237") and fuel_catalog.is_api_id("00000000000006")
    assert not fuel_catalog.is_api_id("солярка") and not fuel_catalog.is_api_id("92")
