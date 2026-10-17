"""Условия ценообразования из типового договора поставки по топливным картам.

Источник: договор ОПТИ 24 (Газпромнефть — Региональные продажи), Приложение № 1
«Порядок установления цены на Товары и условия оплаты», п. 3 и таблица № 3;
редакция для договоров, заключённых с 01.10.2026 по 31.10.2026. Продавец меняет
проценты в одностороннем порядке (там же, п. 4), а у конкретного клиента условия
могут отличаться: процент договора задаётся в OPTI24_SERVICE_FEE_PERCENT.
"""

from __future__ import annotations

import os
from decimal import Decimal

TERMS_EDITION = "типовой договор, Приложение № 1, редакция для договоров с 01.10.2026"

# Сервисный сбор: процент от стоимости каждой покупки (с НДС), списывается со счёта
# договора в момент покупки. Ставка зависит от литров нефтепродуктов за прошлый месяц.
# (верхняя граница литров включительно, процент); None — без верхней границы.
SERVICE_FEE_TIERS: tuple[tuple[int | None, Decimal], ...] = (
    (40_000, Decimal("3.75")),
    (100_000, Decimal("3")),
    (150_000, Decimal("2.5")),
    (500_000, Decimal("1.5")),
    (None, Decimal("1")),
)


def service_fee_for_volume(previous_month_liters: Decimal) -> Decimal:
    """Ставка сервисного сбора в процентах по объёму прошлого месяца."""
    for upper, percent in SERVICE_FEE_TIERS:
        if upper is None or previous_month_liters <= upper:
            return percent
    raise AssertionError("последняя ступень без верхней границы")  # pragma: no cover


def configured_service_fee() -> Decimal | None:
    """Процент сбора из договора клиента (OPTI24_SERVICE_FEE_PERCENT) или None."""
    raw = os.environ.get("OPTI24_SERVICE_FEE_PERCENT", "").strip().replace(",", ".")
    return Decimal(raw) if raw else None


def tiers_payload() -> list[dict[str, str]]:
    """Ступени сбора для ответа модели, когда процент договора неизвестен."""
    rows: list[dict[str, str]] = []
    previous_upper = 0
    for upper, percent in SERVICE_FEE_TIERS:
        if upper is None:
            volume = f"более {previous_upper:,} л"
        else:
            start = previous_upper + 1 if previous_upper else 0
            volume = f"{start:,}–{upper:,} л"
            previous_upper = upper
        rows.append({"previous_month_volume": volume.replace(",", " "), "percent": str(percent)})
    return rows
