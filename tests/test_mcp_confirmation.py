"""MCP-режим: изменения подтверждает человек через elicitation, а не окно «Allow» клиента."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

import pytest
from mcp.server.elicitation import AcceptedElicitation, CancelledElicitation, DeclinedElicitation

from agent import mcp_server


def _context(result: object = None, error: Exception | None = None) -> NS:
    async def elicit(*, message: str, schema: type) -> object:
        assert "Выполнить" in message and schema is mcp_server._Confirmation
        if error:
            raise error
        return result

    return NS(elicit=elicit)


@pytest.mark.parametrize(
    ("result", "error", "approved"),
    [
        (AcceptedElicitation(data=mcp_server._Confirmation(confirm=True)), None, True),
        (AcceptedElicitation(data=mcp_server._Confirmation(confirm=False)), None, False),
        (DeclinedElicitation(), None, False),
        (CancelledElicitation(), None, False),
        # Клиент без elicitation или вызов вне запроса: изменение не выполняется.
        (None, RuntimeError("elicitation not supported"), False),
    ],
)
def test_only_explicit_human_yes_allows_change(
    monkeypatch: pytest.MonkeyPatch, result: object, error: Exception | None, approved: bool
) -> None:
    monkeypatch.setattr(mcp_server.server, "get_context", lambda: _context(result, error))
    assert asyncio.run(mcp_server._ask_human_via_client("Заблокировать карт: 1\nВыполнить? [да/нет]: ")) is approved


def test_server_no_longer_auto_approves() -> None:
    assert not hasattr(mcp_server, "_human_already_approved")
