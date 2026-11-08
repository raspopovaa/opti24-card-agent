"""Справочники: действующие записи и подписи продуктов."""

from types import SimpleNamespace as NS

from agent.tools.reference import active_rows


def test_active_rows_skip_inactive_and_duplicate_codes() -> None:
    items = [
        NS(id="Inactive", code=None, value="Неактивна", name=None, active=False),
        NS(id="Inactive", code=None, value="Неактивна", name=None, active=True),
        NS(id="Active", code=None, value="Активна", name=None, active=True),
        NS(id="00000000000006", code=None, value=None, name="ДТ"),  # без поля active
    ]
    assert active_rows(items) == [
        {"code": "Inactive", "name": "Неактивна"},
        {"code": "Active", "name": "Активна"},
        {"code": "00000000000006", "name": "ДТ"},
    ]
