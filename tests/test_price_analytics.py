"""Аналитика цены топлива: расчёт без модели и инструмент на фейковом клиенте."""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace as NS

from agent.analysis import fuel_price_analytics
from agent.config import Mode
from tests.test_agent import make_tools, tx


def fill(i: int, poi: str, product: str, qty: float, paid: str, station: str, kind: str = "P") -> NS:
    t = tx(i, 12, "0", qty)
    t.poi_id, t.product_name, t.type = poi, product, kind
    t.sum, t.sum_no_discount = paid, station
    return t


REGIONS = {"azs-msk": "Москва", "azs-tver": "Тверская область"}


def test_client_price_includes_discount_and_station_price_does_not() -> None:
    result = fuel_price_analytics([fill(1, "azs-msk", "ДТ", 100, "6000", "6200")], REGIONS)
    product = result["by_product"][0]  # type: ignore[index]
    assert product["client_price_per_liter"] == "60,00 ₽"
    assert product["station_price_per_liter"] == "62,00 ₽"
    assert product["discount_total"] == "200,00 ₽"


def test_markup_inside_operation_shows_as_negative_discount() -> None:
    result = fuel_price_analytics([fill(1, "azs-msk", "ДТ", 10, "650", "620")], REGIONS)
    assert result["by_product"][0]["discount_total"] == "-30,00 ₽"  # type: ignore[index]


def test_liters_and_money_are_grouped_by_region() -> None:
    txs = [
        fill(1, "azs-msk", "ДТ", 100, "6000", "6000"),
        fill(2, "azs-tver", "ДТ", 50, "2900", "2900"),
        fill(3, "azs-msk", "АИ-95", 20, "1300", "1300"),
    ]
    regions = {r["region"]: r for r in fuel_price_analytics(txs, REGIONS)["by_region"]}  # type: ignore[union-attr]
    assert regions["Москва"]["liters"] == "120" and regions["Москва"]["paid"] == "7 300,00 ₽"
    assert regions["Тверская область"]["liters"] == "50"


def test_cheaper_region_is_suggested_with_saving_estimate() -> None:
    txs = [fill(i, "azs-msk", "ДТ", 100, "6200", "6300") for i in range(3)]
    txs += [fill(10 + i, "azs-tver", "ДТ", 100, "5900", "6000") for i in range(3)]
    options = fuel_price_analytics(txs, REGIONS)["cheaper_options"]
    assert options == [
        {
            "product": "ДТ",
            "expensive_region": "Москва",
            "expensive_price_per_liter": "62,00 ₽",
            "cheaper_region": "Тверская область",
            "cheaper_price_per_liter": "59,00 ₽",
            "difference_per_liter": "3,00 ₽",
            "liters_in_expensive_region": "300",
            "possible_saving": "900,00 ₽",
        }
    ]


def test_regions_with_few_operations_are_not_compared() -> None:
    txs = [fill(i, "azs-msk", "ДТ", 100, "6200", "6200") for i in range(3)]
    txs.append(fill(10, "azs-tver", "ДТ", 100, "5000", "5000"))  # одна дешёвая заправка — не вывод
    assert fuel_price_analytics(txs, REGIONS)["cheaper_options"] == []


def test_different_fuels_are_never_compared_with_each_other() -> None:
    txs = [fill(i, "azs-msk", "ДТ", 100, "6200", "6200") for i in range(3)]
    txs += [fill(10 + i, "azs-tver", "АИ-92", 100, "5200", "5200") for i in range(3)]
    assert fuel_price_analytics(txs, REGIONS)["cheaper_options"] == []


def test_refunds_storno_and_unknown_regions() -> None:
    storno = fill(2, "azs-msk", "ДТ", 100, "6000", "6000")
    storno.is_storno = True
    txs = [
        fill(1, "azs-msk", "ДТ", 100, "6000", "6000"),
        storno,
        fill(3, "azs-msk", "ДТ", 100, "6000", "6000", kind="R"),
        fill(4, "azs-unknown", "ДТ", 10, "600", "600"),
    ]
    result = fuel_price_analytics(txs, REGIONS)
    assert result["operations"] == 2 and result["unknown_region_operations"] == 1


def _with_stations(fake: NS, calls: list[str]) -> None:
    async def get_azs_list_v1(*, id: str, **_: object) -> NS:
        calls.append(id)
        codes = {"azs-msk": "77", "azs-tver": "69"}
        station = NS(regionCode=codes[id]) if id in codes else None
        return NS(data=NS(result=[station] if station else [], total_count=1 if station else 0))

    async def get_dictionary(*, name: str, **_: object) -> NS:
        assert name == "Region"
        items = [NS(id="77", code=None, name="Москва", value=None), NS(id="69", code=None, name="Тверская область", value=None)]
        return NS(data=NS(result=items, total_count=2))

    fake.dictionaries = NS(get_azs_list_v1=get_azs_list_v1, get_dictionary=get_dictionary)


def test_tool_resolves_regions_once_and_filters_product(tmp_path: Path) -> None:
    txs = [fill(1, "azs-msk", "ДТ", 100, "6000", "6100"), fill(2, "azs-tver", "АИ-95", 10, "650", "650")]
    fake, budget, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path, txs=txs)
    calls: list[str] = []
    _with_stations(fake, calls)
    args = {"date_from": "2026-09-01", "date_to": "2026-09-30", "product": "дт"}
    out = json.loads(asyncio.run(tools["fuel_price_analytics"].call(args)))
    assert [p["product"] for p in out["by_product"]] == ["ДТ"]
    assert out["by_region"][0]["region"] == "Москва" and "hint" not in out
    asyncio.run(tools["fuel_price_analytics"].call(args))
    assert calls == ["azs-msk"] and budget.spent == 1  # АЗС и страница операций — из кеша


def test_tool_spends_budget_like_spending_report_and_shares_cache(tmp_path: Path) -> None:
    txs = [fill(1, "azs-msk", "ДТ", 100, "6000", "6000")]
    fake, budget, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path, txs=txs)
    _with_stations(fake, [])
    args = {"date_from": "2026-09-01", "date_to": "2026-09-30"}
    asyncio.run(tools["spending_report"].call(args))
    asyncio.run(tools["fuel_price_analytics"].call(args))
    assert budget.spent == 1


def test_tool_rejects_long_period_without_spending(tmp_path: Path) -> None:
    _, budget, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path)
    out = json.loads(
        asyncio.run(tools["fuel_price_analytics"].call({"date_from": "2026-01-01", "date_to": "2026-06-30"}))
    )
    assert out["error"] == "bad_dates" and budget.spent == 0


def test_station_lookups_are_capped_and_reported(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import agent.tools as tools_module

    monkeypatch.setattr(tools_module, "MAX_POI_LOOKUPS", 1)
    txs = [fill(1, "azs-msk", "ДТ", 10, "600", "600"), fill(2, "azs-tver", "ДТ", 10, "600", "600")]
    fake, _, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path, txs=txs)
    calls: list[str] = []
    _with_stations(fake, calls)
    out = json.loads(
        asyncio.run(tools["fuel_price_analytics"].call({"date_from": "2026-09-01", "date_to": "2026-09-30"}))
    )
    assert len(calls) == 1 and "1 АЗС" in out["hint"] and out["unknown_region_operations"] == 1


def test_decimal_money_has_no_float_noise() -> None:
    txs = [fill(1, "azs-msk", "ДТ", 0.1, "0.1", "0.2") for _ in range(3)]
    product = fuel_price_analytics(txs, REGIONS)["by_product"][0]  # type: ignore[index]
    assert product["client_price_per_liter"] == "1,00 ₽" and Decimal("0.3") == Decimal(product["liters"])


def test_service_fee_tiers_follow_contract_table() -> None:
    from agent.pricing import service_fee_for_volume, tiers_payload

    assert [service_fee_for_volume(Decimal(v)) for v in ("0", "40000", "40001", "100000", "150001", "500001")] == [
        Decimal("3.75"), Decimal("3.75"), Decimal("3"), Decimal("3"), Decimal("1.5"), Decimal("1"),
    ]
    assert tiers_payload()[0] == {"previous_month_volume": "0–40 000 л", "percent": "3.75"}
    assert tiers_payload()[-1] == {"previous_month_volume": "более 500 000 л", "percent": "1"}


def test_service_fee_is_added_on_top_of_paid_price() -> None:
    result = fuel_price_analytics(
        [fill(1, "azs-msk", "ДТ", 100, "6000", "6200")], REGIONS, service_fee_percent=Decimal("3.75")
    )
    product = result["by_product"][0]  # type: ignore[index]
    assert product["client_price_per_liter"] == "60,00 ₽"
    assert product["service_fee"] == "225,00 ₽"
    assert product["client_price_with_fee_per_liter"] == "62,25 ₽"


def test_saving_estimate_includes_service_fee() -> None:
    txs = [fill(i, "azs-msk", "ДТ", 100, "6200", "6300") for i in range(3)]
    txs += [fill(10 + i, "azs-tver", "ДТ", 100, "5900", "6000") for i in range(3)]
    option = fuel_price_analytics(txs, REGIONS, service_fee_percent=Decimal("1"))["cheaper_options"][0]  # type: ignore[index]
    assert option["difference_per_liter"] == "3,03 ₽" and option["possible_saving"] == "909,00 ₽"


def test_tool_picks_fee_by_previous_month_or_shows_tiers(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("OPTI24_SERVICE_FEE_PERCENT", raising=False)
    txs = [fill(1, "azs-msk", "ДТ", 100, "6000", "6000")]
    fake, _, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path, txs=txs)
    _with_stations(fake, [])
    period = {"date_from": "2026-09-01", "date_to": "2026-09-30"}
    unknown = json.loads(asyncio.run(tools["fuel_price_analytics"].call(period)))
    assert unknown["service_fee_source"] == "неизвестен" and len(unknown["service_fee_tiers"]["tiers"]) == 5
    assert "service_fee" not in unknown["by_product"][0]
    by_volume = json.loads(
        asyncio.run(tools["fuel_price_analytics"].call({**period, "previous_month_liters": 120000}))
    )
    assert by_volume["by_product"][0]["service_fee"] == "150,00 ₽"  # 2,5 % от 6 000 ₽
    monkeypatch.setenv("OPTI24_SERVICE_FEE_PERCENT", "1,5")
    configured = json.loads(asyncio.run(tools["fuel_price_analytics"].call(period)))
    assert configured["service_fee_source"] == "OPTI24_SERVICE_FEE_PERCENT"
    assert configured["by_product"][0]["service_fee"] == "90,00 ₽"
