"""Денежные значения: только Decimal, никакого float в расчётах.

API присылает цены и суммы то числом, то строкой. Число сначала превращаем в
строку, и только потом в Decimal: Decimal(0.1) хранит двоичную погрешность,
а Decimal(str(0.1)) — ровно 0.1.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

KOPECK = Decimal("0.01")


def to_decimal(value: float | int | str | None) -> Decimal:
    if value is None or value == "":
        return Decimal("0")
    return Decimal(str(value))


def rub(value: Decimal) -> str:
    """Сумма для показа: '12 345,60 ₽'."""
    quantized = value.quantize(KOPECK, rounding=ROUND_HALF_UP)
    text = f"{quantized:,.2f}".replace(",", " ").replace(".", ",")
    return f"{text} ₽"
