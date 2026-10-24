"""Длинный ответ инструмента сокращается целым JSON с пометкой, а не обрезается посередине."""

import json

from agent.llm_openai import fit_tool_output


def test_short_answer_is_unchanged():
    result = {"total": 2, "items": [1, 2]}

    assert json.loads(fit_tool_output(result)) == result


def test_long_list_is_shortened_with_counts():
    items = [{"id": i, "comment": "x" * 50} for i in range(500)]

    out = fit_tool_output({"period": "2026-10", "items": items}, limit=2000)

    data = json.loads(out)
    assert len(out) <= 2000
    assert data["period"] == "2026-10"
    assert data["truncated"]["items"]["total"] == 500
    assert data["truncated"]["items"]["shown"] == len(data["items"])
    assert data["items"] == items[: len(data["items"])]
    assert "hint" in data


def test_long_text_becomes_marked_partial_json():
    out = fit_tool_output('"' * 5000, limit=1000)

    data = json.loads(out)
    assert len(out) <= 1000
    assert data["truncated"] is True
    assert data["partial"]
