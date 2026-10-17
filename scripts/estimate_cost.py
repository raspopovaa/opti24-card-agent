"""Расчёт стоимости одного вопроса агенту по моделям.

Запуск: python scripts/estimate_cost.py
Цены — с таблицы моделей на 2026-10-06 ($ за 1 млн токенов). Количество токенов —
оценки, заложенные в ASSUMPTIONS. Замените их своими измерениями: агент в конце
сессии печатает фактические токены (Usage), а ``client.messages.count_tokens``
считает вход точно.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal as D

USD_RUB = D("90")  # условный курс для наглядности; поставьте актуальный


@dataclass(frozen=True)
class Model:
    name: str
    input: D
    output: D
    cache_read_factor: D  # доля от цены входа за токен из кеша


MODELS = [
    Model("claude-haiku-5-5", D("0.10"), D("0.50"), D("0.1")),
    Model("claude-sonnet-5-5", D("2.00"), D("10.00"), D("0.1")),
    Model("claude-opus-5-5", D("4.00"), D("20.00"), D("0.05")),
    Model("claude-fable-5-1", D("10.00"), D("50.00"), D("0.025")),
]

ASSUMPTIONS = {
    "prefix": 2500,  # системная инструкция + схемы 8 инструментов (оценка по размеру в символах)
    "question": 120,  # вопрос пользователя с датой
    "tool_result": 700,  # одна сводка инструмента (код сжимает сотни операций до сводки)
    "requests": 3,  # план → вызов инструментов → ответ
    "output_per_request": [180, 180, 450],  # вызов, вызов, финальный текст (с мышлением)
}
CACHE_WRITE = D("1.25")
M = D(1_000_000)


def cost_usd(model: Model, *, cached: bool, warm: bool) -> D:
    """Стоимость одного вопроса. warm — кеш уже записан предыдущим вопросом."""
    a = ASSUMPTIONS
    total = D(0)
    context = a["question"]
    for i in range(a["requests"]):
        prefix_price = model.input
        if cached:
            first_read = warm or i > 0
            prefix_price = model.input * (model.cache_read_factor if first_read else CACHE_WRITE)
        total += D(a["prefix"]) * prefix_price / M
        total += D(context) * model.input / M
        total += D(a["output_per_request"][i]) * model.output / M
        context += a["output_per_request"][i] + a["tool_result"]
    return total


def main() -> None:
    print(f"{'модель':<20}{'без кеша':>12}{'кеш, холодный':>16}{'кеш, тёплый':>14}{'₽ за 1000':>12}")
    base = cost_usd(MODELS[2], cached=False, warm=False)
    for m in MODELS:
        plain = cost_usd(m, cached=False, warm=False)
        cold = cost_usd(m, cached=True, warm=False)
        warm = cost_usd(m, cached=True, warm=True)
        print(
            f"{m.name:<20}{plain:>11.4f}${cold:>15.4f}${warm:>13.4f}$"
            f"{warm * 1000 * USD_RUB:>11.0f}₽"
        )
    haiku = cost_usd(MODELS[0], cached=True, warm=True)
    print(f"\nHaiku 5.5 с кешем дешевле Opus 5.5 без кеша в {base / haiku:.0f} раз(а).")

    # Каскад: простые вопросы на Haiku, сложные на Sonnet (кеши моделей раздельные).
    mix = D("0.8") * haiku + D("0.2") * cost_usd(MODELS[1], cached=True, warm=True)
    print(f"Каскад 80% Haiku + 20% Sonnet: {mix:.4f}$ за вопрос.")


if __name__ == "__main__":
    main()
