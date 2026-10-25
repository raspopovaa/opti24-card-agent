"""Общие константы и помощники инструментов агента."""

from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from apisdkopti24 import APIClient, APIError

from ..money import rub, to_decimal
from ..safety import BillingBudget

MAX_LIST = 25
# Бесплатный пакет запросов API в месяц по текущему тарифу; после него API блокируется.
# Пакет зависит от договора, поэтому значение настраивается (OPTI24_MONTHLY_QUOTA).
DEFAULT_MONTHLY_QUOTA = 500
QUOTA_WARNING_LEFT = 50
# Регион АЗС ищется отдельным бесплатным запросом на каждую АЗС; при лимите SDK
# 1 запрос в секунду больше 40 АЗС за вопрос — слишком долгое ожидание.
MAX_POI_LOOKUPS = 40


def _json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


class TransactionsUnavailableError(Exception):
    """Стенд не отдаёт операции (на DEMO get_transactions_v2 отвечает HTTP 405)."""


def _error(exc: Exception) -> str:
    # Текст исключений SDK очищен от секретов, но всё равно ограничиваем длину.
    return _json({"error": type(exc).__name__, "message": str(exc)[:300]})


def _rub_optional(value: float | int | str | None) -> str | None:
    """Сумма или None: не показываем «0 ₽» там, где API вернул null."""
    if value is None or value == "":
        return None
    return rub(to_decimal(value))


def limit_left(part: Any) -> str | None:
    """Остаток лимита (установлено минус израсходовано) в Decimal; None, если чего-то нет.

    Остаток считает код, а не модель: проверка ответа помечает числа, которых нет в данных.
    """
    values = part.model_dump() if part else {}
    if values.get("value") is None or values.get("used") is None:
        return None
    return str(to_decimal(values["value"]) - to_decimal(values["used"]))


def describe_card(card: Any) -> str:
    """Карта для окна подтверждения: номер, ID, комментарий, статус и группа — из ответа API."""
    parts = [f"{card.number} (ID {card.id})"]
    if getattr(card, "comment", None):
        parts.append(f"«{card.comment}»")
    status = getattr(card, "status_name", None) or getattr(card, "status", None)
    if status:
        parts.append(f"сейчас: {status}")
    if getattr(card, "group_name", None):
        parts.append(f"группа {card.group_name}")
    return ", ".join(parts)


# Коды периода лимита, значение которых подтверждено; остальные показываются кодом.
PERIOD_WORDS = {3: "сутки", 7: "неделя"}


def period_words(number: object, period_type: object) -> str:
    """Период лимита словами: «сутки», «2 × неделя»; неизвестный код показывается как есть."""
    try:
        name = PERIOD_WORDS.get(int(str(period_type)))
    except ValueError:
        name = None
    unit = name or f"период с кодом {period_type} (значение кода уточните в личном кабинете)"
    return unit if str(number) == "1" else f"{number} × {unit}"


def describe_limit(item: Any) -> str:
    """Продуктовый лимит словами: размер, израсходовано и период."""
    sizes = []
    for part, unit in ((getattr(item, "amount", None), "л"), (getattr(item, "sum", None), "₽")):
        if part is not None and getattr(part, "value", None) is not None:
            used = getattr(part, "used", None)
            sizes.append(f"{part.value} {unit}" + (f" (израсходовано {used})" if used is not None else ""))
    time = getattr(item, "time", None)
    period = period_words(time.number, time.type) if time is not None else "период не указан"
    product = getattr(item, "productType", None) or getattr(item, "productGroup", None) or "все продукты"
    return f"лимит {item.id}: {', '.join(sizes) or 'размер не указан'} на {period}, продукт {product}"


def verified(matched: bool | None, message: str) -> dict[str, str]:
    """Сверка после изменения: match, mismatch или not_checked и что сказать человеку."""
    status = "not_checked" if matched is None else ("match" if matched else "mismatch")
    return {"status": status, "message": message}


def limit_matches(item: Any, liters: float | None, rubles: float | None, period: tuple[int, int]) -> bool:
    """Лимит из get_limits совпадает с подтверждённым: размер и период."""
    for part, want in ((getattr(item, "amount", None), liters), (getattr(item, "sum", None), rubles)):
        have = getattr(part, "value", None) if part is not None else None
        if want is not None and (have is None or to_decimal(have) != to_decimal(want)):
            return False
    time = getattr(item, "time", None)
    return time is not None and (int(time.number), int(time.type)) == period


def mask_email(email: str | None) -> str | None:
    """Почта для модели без лишних личных данных: «iv***@example.ru»."""
    if not email or "@" not in email:
        return None
    name, domain = email.split("@", 1)
    return f"{name[:2]}***@{domain}"


def mask_phone(phone: str | None) -> str | None:
    """Телефон для модели без лишних личных данных: последние 2 цифры, «…-00»."""
    digits = "".join(ch for ch in str(phone or "") if ch.isdigit())
    return f"…-{digits[-2:]}" if len(digits) >= 2 else None


def percent(part: int, whole: int) -> str | None:
    """Доля в процентах с одним знаком после запятой; None, если делить не на что."""
    if whole <= 0:
        return None
    return str((Decimal(part) * 100 / Decimal(whole)).quantize(Decimal("0.1")))


def _iso(value: object) -> str:
    return value.isoformat() if isinstance(value, datetime) else str(value)


def _doc_date(value: int | str) -> str:
    """Дата документа: UNIX timestamp или строка."""
    try:
        return datetime.fromtimestamp(int(value)).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        return str(value)


def _check_dates(date_from: str, date_to: str) -> str | None:
    try:
        start, end = date.fromisoformat(date_from), date.fromisoformat(date_to)
    except ValueError:
        return "Даты нужны в формате ГГГГ-ММ-ДД."
    if end < start:
        return "Дата окончания раньше даты начала."
    if (end - start).days > 31:
        return "Период не больше месяца: разбей запрос на части."
    return None


async def load_transactions(
    client: APIClient,
    budget: BillingBudget,
    date_from: str,
    date_to: str,
    card_id: str | None = None,
    max_pages: int | None = None,
) -> tuple[list[Any], int]:
    """Операции за период и их общее число; страница из 100 операций — один тарифный запрос.

    При ``max_pages`` загрузка останавливается раньше, и тогда длина списка
    меньше общего числа: вызывающий код видит, что данные неполные.
    """
    items: list[Any] = []
    offset = 0
    pages = 0
    total = 0
    while True:
        if max_pages is not None and pages >= max_pages:
            return items, total
        pages += 1
        budget.spend()
        try:
            if card_id:
                page = await client.transactions.get_card_transactions_v2(
                    card_id=card_id,
                    date_from=date_from,
                    date_to=date_to,
                    page_limit=100,
                    page_offset=offset,
                )
            else:
                page = await client.transactions.get_transactions_v2(
                    date_from=date_from, date_to=date_to, page_limit=100, page_offset=offset
                )
        except APIError as exc:
            if exc.http_status_code == 405:
                budget.spent -= 1  # отклонённый запрос не считаем тарифным
                raise TransactionsUnavailableError(
                    "Сервер ответил HTTP 405: на DEMO операции не отдаются (так отвечает "
                    "сам стенд, это видно в проверках opti24-live-check). Операции доступны "
                    "на рабочем договоре."
                ) from exc
            raise
        batch = page.data.result or []
        items.extend(batch)
        offset += len(batch)
        total = page.data.total_count
        if not batch or offset >= total:
            return items, total
