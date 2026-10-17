"""Коды ответа терминала из Приложения № 4 к договору."""

from __future__ import annotations

import pytest

from agent.decline_codes import decode


@pytest.mark.parametrize(
    ("code", "check"),
    [
        ("05", "card_status"),
        ("5", "card_status"),
        ("14", "card_status"),
        ("54", "card_status"),
        ("51", "contract_balance"),
        ("55", "pin_attempts"),
        ("75", "pin_attempts"),
        ("76", "pin_attempts"),
        ("57", "restrictions"),
        ("58", "region_limits"),
        ("61", "limits"),
        ("65", "limits"),
        ("19", None),
        ("69", None),
        ("82", None),
        ("91", None),
        ("z1", None),
        ("Z3", None),
        ("34", None),
    ],
)
def test_table_codes_map_to_related_checks(code: str, check: str | None) -> None:
    result = decode(code)
    assert result["in_table"] is True and result["related_check"] == check


def test_unknown_code_is_other_code_not_a_guess() -> None:
    result = decode(" er3 ")
    assert result["code"] == "ER3" and result["in_table"] is False
    assert "прочие коды" in str(result["meaning"]) and result["related_check"] is None
    assert "Приложение № 4" in str(result["source"])
