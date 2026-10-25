"""Инструменты агента: Карты и группы карт: поиск, профиль, блокировка, комментарий."""

from __future__ import annotations

from typing import Any

from anthropic import beta_async_tool
from apisdkopti24 import APIError
from apisdkopti24.errors import RequestValidationError

from ..config import Mode
from ..safety import BudgetExceededError, mask_card_number
from .common import MAX_LIST, _error, _iso, _json, _rub_optional, describe_card, mask_phone
from .context import ToolContext


def build(ctx: ToolContext) -> dict[str, Any]:
    """Инструменты темы по имени; порядок в списке агента задаёт agent/tools/__init__.py."""
    client = ctx.client
    budget = ctx.budget
    gate = ctx.gate
    limits_payload = ctx.limits_payload
    confirm_write = ctx.confirm_write
    perform = ctx.perform
    finish = ctx.finish
    cards_by_id = ctx.cards_by_id
    default_group_id = ctx.default_group_id

    @beta_async_tool
    async def list_cards(
        status: str | None = None, group_id: str | None = None, query: str | None = None
    ) -> str:
        """Список топливных карт договора: ID, номер, статус, группа, комментарий. Бесплатно.

        Нужен, чтобы найти ID карты для других инструментов.
        Показывает не больше 25 карт; если total больше shown, сузь фильтр.

        Args:
            status: Код статуса: Active, Locked(Client), Locked(Manager), Inactive
                (все — lookup_dictionary CardStatus); русские названия API не принимает.
            group_id: Показать только карты этой группы (ID из list_card_groups).
            query: Поиск по номеру карты или комментарию.
        """
        defaults: dict[str, str] = {}
        if not group_id and not query and default_group_id:  # поиск по номеру или имени — по всем картам
            group_id = default_group_id
            defaults["group_id"] = f"{group_id} (OPTI24_DEFAULT_GROUP_ID)"
        try:
            response = await client.cards.get_cards_v2(
                status=status, group_id=group_id, q=query, onpage=MAX_LIST
            )
        except (APIError, RequestValidationError) as exc:
            return _error(exc)
        cards = response.data.result or []
        total = response.data.total_count
        hint = (
            {"hint": "Показана часть карт: уточни query, status или group_id."}
            if total and total > len(cards)
            else {}
        )
        return _json(
            {
                "total": total,
                "shown": len(cards),
                **hint,
                **({"defaults": defaults} if defaults else {}),
                "cards": [
                    {
                        "id": c.id,
                        "number": c.number,
                        "status": c.status_name or c.status,
                        "group": c.group_name,
                        "comment": c.comment,
                    }
                    for c in cards
                ],
            }
        )

    @beta_async_tool
    async def get_card_profile(
        card_id: str, include_drivers: bool = True, include_limits: bool = True
    ) -> str:
        """Карта, водитель, группа и лимиты. По тарифу: 1–2 запроса.

        Вызывай после list_cards. include_drivers=False экономит запрос.
        """
        try:
            budget.spend()
            detail_response = await client.cards.get_card_detail(card_id=card_id)
            details = detail_response.data.result or []
            if not details:
                return _json({"error": "not_found", "message": "Карта не найдена."})
            card = details[0]

            # Группа берётся из бесплатного списка карт по номеру.
            group = None
            try:
                cards_response = await client.cards.get_cards_v2(q=card.number, onpage=1)
                cards = cards_response.data.result or []
                group = cards[0].group_name if cards else None
            except Exception:  # noqa: BLE001 — группа не критична для профиля
                group = None

            profile: dict[str, object] = {
                "id": card.id,
                "number": card.number,
                "status": card.status,
                "comment": card.comment,
                "group": group,
                "product": card.product,
                "carrier": card.carrier,
                "available": _rub_optional(card.available),
                "currency": card.currency,
                "last_used_at": _iso(card.date_last_usage) if card.date_last_usage else None,
                "last_station": card.servicecenter_last_usage_name,
                "mpc": card.mpc,
                "pin_reset": card.pin_reset,
                "pin_counter": card.pin_counter,
            }

            if include_drivers:
                budget.spend()
                drivers_response = await client.cards.get_card_drivers(card_id=card_id)
                profile["drivers"] = [
                    {
                        "name": " ".join(
                            part for part in (d.last_name, d.first_name, d.middle_name) if part
                        ),
                        "position": d.position,
                        "role": d.role,
                        "phone": mask_phone(d.mobile_phone),
                    }
                    for d in (drivers_response.data.result or [])
                ]

            if include_limits:
                limits_response = await client.limits.get_limits(card_id=card_id)
                profile["limits_total"] = limits_response.data.total_count
                profile["limits"] = await limits_payload(limits_response.data.result or [])
        except (APIError, RequestValidationError, BudgetExceededError) as exc:
            return _error(exc)
        return _json(profile)

    @beta_async_tool
    async def list_card_groups() -> str:
        """Группы карт (например, по подразделениям). Бесплатный запрос API."""
        try:
            response = await client.card_groups.get_card_groups()
        except APIError as exc:
            return _error(exc)
        return _json(
            [
                {"id": g.id, "name": g.name, "cards": g.cards_count}
                for g in response.data.result or []
            ]
        )

    @beta_async_tool
    async def block_cards(card_ids: list[str], reason: str, block: bool = True) -> str:
        """Заблокировать или разблокировать карты. ИЗМЕНЯЕТ ДАННЫЕ, нужно подтверждение.

        Только по прямой просьбе пользователя и по конкретным картам;
        разблокировка — отдельное решение человека.

        Args:
            card_ids: ID карт (из list_cards), не больше 10.
            reason: Причина одной фразой, для подтверждения человеку.
            block: True — заблокировать, False — разблокировать.
        """
        card_ids = list(dict.fromkeys(str(c).strip() for c in card_ids or [] if str(c).strip()))
        if not card_ids or len(card_ids) > 10:
            return _json({"error": "bad_request", "message": "Нужно от 1 до 10 карт."})
        action = "block_cards" if block else "unblock_cards"
        verb = "Заблокировать" if block else "Разблокировать"
        if gate.mode is Mode.READ_ONLY:
            _, denied = await confirm_write(action, "", len(card_ids), {"card_ids": card_ids})
            return denied
        # Человек подтверждает карты, которые показал API, а не ID, которые назвала модель.
        cards = await cards_by_id(card_ids)
        missing = [card_id for card_id in card_ids if card_id not in cards]
        if missing:
            return _json(
                {
                    "error": "not_found",
                    "message": f"Карт с ID {', '.join(missing)} нет в списке карт договора: ничего не изменено. "
                    "Найди карты через list_cards и не угадывай ID.",
                }
            )
        lines = "\n".join(f"- {describe_card(cards[card_id])}" for card_id in card_ids)
        approval, denied = await confirm_write(
            action,
            f"{verb} карты ({len(card_ids)}):\n{lines}\nПричина со слов агента: {reason}",
            len(card_ids),
            {"card_ids": card_ids, "cards": [mask_card_number(cards[c].number) for c in card_ids]},
            {"card_ids": card_ids, "block": block},
        )
        if denied:
            return denied
        response, failed = await perform(
            approval,
            lambda: client.cards.block_card(card_ids=card_ids, block=block),
            "статус карт в list_cards",
        )
        if failed:
            return failed
        done = [str(item) for item in (response.data or [])]
        not_done = [card_id for card_id in card_ids if card_id not in done]
        finish(approval, "partial" if not_done else "done", {"done": done, "not_done": not_done})
        return _json(
            {
                "status": "partial" if not_done else ("blocked" if block else "unblocked"),
                "requested": len(card_ids),
                "done": done,
                "not_done": not_done,
            }
        )

    @beta_async_tool
    async def set_card_comment(card_id: str, comment: str) -> str:
        """Установить комментарий карты (водитель, машина). ИЗМЕНЯЕТ ДАННЫЕ.

        Тарифный запрос, нужно подтверждение человека. Комментарий 1–90 символов.
        """
        card_id = str(card_id).strip()
        comment = str(comment).strip()
        if not card_id:
            return _json({"error": "bad_request", "message": "Нужен card_id."})
        if not 1 <= len(comment) <= 90:
            return _json(
                {"error": "bad_request", "message": "Комментарий должен быть от 1 до 90 символов."}
            )
        target: dict[str, object] = {"card_id": card_id}
        summary = ""
        if gate.mode is not Mode.READ_ONLY:
            card = (await cards_by_id([card_id])).get(card_id)
            if card is None:
                return _json(
                    {
                        "error": "not_found",
                        "message": f"Карты с ID {card_id} нет в списке карт договора: ничего не изменено. "
                        "Найди карту через list_cards.",
                    }
                )
            target["card"] = mask_card_number(card.number)
            was = f"«{card.comment}»" if card.comment else "пусто"
            summary = f"Изменить комментарий карты {card.number} (ID {card_id}): было {was}, станет «{comment}»."
        approval, denied = await confirm_write(
            "set_card_comment", summary, 1, target, {"card_id": card_id, "comment": comment}
        )
        if denied:
            return denied
        response, failed = await perform(
            approval,
            lambda: client.cards.set_card_comment(card_id=card_id, comment=comment),
            "комментарий карты в list_cards",
        )
        if failed:
            return failed
        finish(approval, "done" if response.data else "not_done")
        return _json(
            {
                "status": "updated",
                "card_id": card_id,
                "comment": comment,
                "saved": bool(response.data),
            }
        )

    return {
        "list_cards": list_cards,
        "get_card_profile": get_card_profile,
        "list_card_groups": list_card_groups,
        "block_cards": block_cards,
        "set_card_comment": set_card_comment,
    }
