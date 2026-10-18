"""Инструменты агента: справочники API (статусы, схемы скидок, условия оплаты, услуги АЗС)."""

from __future__ import annotations

from typing import Any

from anthropic import beta_async_tool

from .common import MAX_LIST, _json
from .context import ToolContext

# Справочники метода get_dictionary (описание метода в SDK apisdkopti24). Все бесплатные.
DICTIONARIES: dict[str, str] = {
    "CardStatus": "статусы карт",
    "ContractStatus": "статусы договоров",
    "Country": "страны",
    "Currency": "валюты",
    "Goods": "виды топлива для цен на АЗС",
    "PaymentScheme": "схемы оплаты договора",
    "PaymentTerm": "условия оплаты договора",
    "ProductGroup": "группы продуктов",
    "ProductType": "типы продуктов",
    "POIType": "типы торговых точек",
    "Region": "регионы",
    "Services": "услуги на АЗС",
    "Unit": "единицы измерения",
    "Office": "офисы продаж",
    "POIPartner": "партнёры (сети АЗС)",
    "DiscountScheme": "схемы расчёта скидки",
}


def build(ctx: ToolContext) -> dict[str, Any]:
    """Инструменты темы по имени; порядок в списке агента задаёт agent/tools/__init__.py."""
    dictionary = ctx.dictionary

    @beta_async_tool(
        description=(
            "Справочник API: что значит код и какие бывают значения. Бесплатно. name: "
            + ", ".join(DICTIONARIES)
            + " (Services — услуги АЗС, Office — офисы продаж, POIPartner — сети АЗС)."
        )
    )
    async def lookup_dictionary(name: str, query: str | None = None) -> str:
        """Записи справочника.

        Args:
            name: Имя справочника из описания, например DiscountScheme.
            query: Код или часть названия (необязательно).
        """
        key = next((known for known in DICTIONARIES if known.lower() == name.strip().lower()), None)
        if key is None:
            return _json({"error": "unknown_dictionary", "message": f"Нет справочника {name!r}.", "names": list(DICTIONARIES)})
        items = await dictionary(key)
        rows = [
            {"code": str(getattr(item, "code", None) or item.id), "name": item.name or item.value or str(item.id)}
            for item in items
            if not getattr(item, "deleted", 0)
        ]
        if query:
            needle = query.strip().casefold()
            rows = [row for row in rows if needle in row["code"].casefold() or needle in row["name"].casefold()]
        result: dict[str, object] = {"dictionary": key, "title": DICTIONARIES[key], "total": len(rows), "shown": min(len(rows), MAX_LIST), "items": rows[:MAX_LIST]}
        if not items:
            result["hint"] = "Справочник пуст или недоступен."
        elif len(rows) > MAX_LIST:
            result["hint"] = "Показана часть: уточни query."
        return _json(result)

    return {"lookup_dictionary": lookup_dictionary}
