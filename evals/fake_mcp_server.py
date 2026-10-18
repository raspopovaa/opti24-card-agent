"""MCP-сервер с инструментами агента на поддельном Opti24: для evals через Claude Code.

Не требует договора, сети и ключей. Режим только чтения: изменяющие инструменты
всегда отклоняются. Запускается из evals/run_evals_claude.py.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from agent.config import Mode  # noqa: E402
from agent.safety import ApprovalGate, BillingBudget  # noqa: E402
from agent.tools import build_tools  # noqa: E402
from evals.run_evals import FakeOpti24, never  # noqa: E402

server = FastMCP("fuel-cards")

_audit_log = Path(tempfile.gettempdir()) / "opti24-evals-audit.jsonl"
_gate = ApprovalGate(mode=Mode.READ_ONLY, ask=never, audit_log=_audit_log)
for _tool in build_tools(FakeOpti24(), BillingBudget(limit=20), _gate):  # type: ignore[arg-type]
    server.tool(name=_tool.name)(_tool.func)

if __name__ == "__main__":
    server.run()
