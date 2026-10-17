"""Те же инструменты как MCP-сервер: агент запускается внутри Claude Code.

Зачем: Claude Code работает по подписке Claude, а не по платному API-ключу.
Роль цикла и модели берёт на себя Claude Code (модель выбирается командой /model),
а наши инструменты, расчёты и предохранители остаются теми же.

Подтверждение изменений: окну «Allow» в Claude Code доверять нельзя — инструмент
могут разрешить заранее или запустить в автоматическом режиме, и тогда изменение
прошло бы без человека. Поэтому при ``AGENT_MODE=confirm`` сервер сам спрашивает
человека через MCP elicitation: вопрос показывает клиент, и модель ответить за
человека не может. Клиент без elicitation или любой сбой запроса — отказ.

Запуск вручную: ``python -m agent.mcp_server`` (обычно запускает сам Claude Code).
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

from apisdkopti24 import APIClient, ConnectionSettings, EnvironmentCredentialsProvider
from mcp.server.elicitation import AcceptedElicitation
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from .config import Mode, Settings
from .contract import is_production, open_session, preferred_contract_id
from .dashboard import collect
from .safety import ApprovalGate, BillingBudget
from .tools import build_tools

settings = Settings.from_env()
server = FastMCP("fuel-cards")

_client: APIClient | None = None


class _Confirmation(BaseModel):
    confirm: bool = Field(description="Выполнить это изменение")


async def _ask_human_via_client(prompt: str) -> bool:
    """Спрашивает человека через окно MCP-клиента; только явное «да» разрешает изменение."""
    try:
        result = await server.get_context().elicit(message=prompt, schema=_Confirmation)
    except Exception:  # noqa: BLE001 — граница безопасности: не смогли спросить человека — не меняем
        return False
    return isinstance(result, AcceptedElicitation) and result.data.confirm


async def _get_client() -> APIClient:
    global _client
    if _client is None:
        env_file: Path = settings.opti24_env_file
        conn = ConnectionSettings.from_env(env_file=env_file)
        client = APIClient(
            settings=conn,
            credentials_provider=EnvironmentCredentialsProvider.from_env(env_file=env_file),
        )
        await client.__aenter__()
        # Без терминала выбрать договор не у кого: нужен AGENT_CONTRACT_ID или API_CONTRACT_ID.
        await open_session(
            client,
            preferred=preferred_contract_id(env_file),
            production=is_production(conn),
            choose=None,
        )
        _client = client
    return _client


def _register() -> None:
    budget = BillingBudget(limit=settings.max_billable_calls)
    gate = ApprovalGate(mode=settings.mode, ask=_ask_human_via_client, audit_log=settings.audit_log)
    proxy = _ClientProxy()
    for tool in build_tools(proxy, budget, gate):  # type: ignore[arg-type]
        server.tool(name=tool.name)(tool.func)  # описание берётся из docstring

    @server.tool()
    async def get_dashboard() -> str:
        """Дашборд договора: баланс, расход месяца, карты, топливо, контроль заправок.

        Платные запросы API: 1 на договор и до 3 на операции с начала месяца.
        Показывай пользователю как короткую сводку, а не пересказывай все поля.
        """
        try:
            board = await collect(await _get_client(), budget, date.today())  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001 — ошибку отдаём модели, а не роняем сервер
            return json.dumps(
                {"error": type(exc).__name__, "message": str(exc)[:300]}, ensure_ascii=False
            )
        return json.dumps(asdict(board), ensure_ascii=False, default=str)


class _ServiceProxy:
    """Клиент SDK создаётся при первом вызове инструмента, а не при старте сервера."""

    def __init__(self, service: str) -> None:
        self._service = service

    def __getattr__(self, method: str):  # noqa: ANN204
        async def call(**kwargs: object) -> object:
            client = await _get_client()
            return await getattr(getattr(client, self._service), method)(**kwargs)

        return call


class _ClientProxy:
    def __getattr__(self, service: str) -> _ServiceProxy:
        return _ServiceProxy(service)


_register()

if __name__ == "__main__":
    if settings.mode is Mode.READ_ONLY:
        print("fuel-cards: режим read_only, изменяющие инструменты отключены", file=sys.stderr)
    server.run()
