"""Оценка агента на Claude через Claude Code по подписке, без API-ключа.

Для каждого кейса запускается ``claude -p`` с системной инструкцией агента, без
встроенных инструментов Claude Code и с одним MCP-сервером (evals/fake_mcp_server.py):
инструменты агента на поддельном Opti24. Из потока событий берутся имена и
аргументы вызванных инструментов, дальше работает та же проверка, что и в
evals/run_evals.py.

    python evals/run_evals_claude.py                       # Haiku, по одному прогону на кейс
    python evals/run_evals_claude.py --model sonnet --runs 3
    python evals/run_evals_claude.py --limit 3             # только первые 3 кейса (проба)
    python evals/run_evals_claude.py --only "карта c1"      # кейсы, в вопросе которых есть текст

Ограничения: цикл ведёт Claude Code, а не наш агент (иные параметры мышления и
служебные добавки к запросу), поэтому результат приблизительный. Запросы
расходуют лимиты подписки Claude.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.prompt import SYSTEM_PROMPT  # noqa: E402
from evals.run_evals import ToolInput, _describe_calls, check_case  # noqa: E402

SERVER_NAME = "fuel-cards"
TOOL_PREFIX = f"mcp__{SERVER_NAME}__"
CASE_TIMEOUT_SECONDS = 240


def parse_stream(lines: Iterable[str]) -> tuple[list[ToolInput], dict[str, Any] | None, str | None]:
    """Вызовы инструментов агента, итоговое событие и модель из потока stream-json."""
    calls: list[ToolInput] = []
    result: dict[str, Any] | None = None
    model: str | None = None
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "system" and event.get("subtype") == "init":
            model = event.get("model")
        elif event.get("type") == "assistant":
            for block in event.get("message", {}).get("content", []):
                name = block.get("name", "")
                if block.get("type") == "tool_use" and name.startswith(TOOL_PREFIX):
                    calls.append((name.removeprefix(TOOL_PREFIX), dict(block.get("input") or {})))
        elif event.get("type") == "result":
            result = event
    return calls, result, model


def build_command(question: str, model: str, config_path: Path) -> list[str]:
    prompt = f"[Сегодня {date.today():%Y-%m-%d}] {question}"
    return [
        "claude", "-p", prompt,
        "--model", model,
        "--system-prompt", SYSTEM_PROMPT,
        "--tools", "",
        "--strict-mcp-config", "--mcp-config", str(config_path),
        "--allowedTools", f"{TOOL_PREFIX}*",
        "--setting-sources", "project",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--output-format", "stream-json", "--verbose",
    ]  # fmt: skip


async def run_once(question: str, model: str, config_path: Path, workdir: Path) -> tuple[list[ToolInput], str | None, str | None]:
    """Один прогон: вызовы, текст ошибки (или None) и модель из события init."""
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    process = await asyncio.create_subprocess_exec(
        *build_command(question, model, config_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=workdir,
        env=env,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), CASE_TIMEOUT_SECONDS)
    except TimeoutError:
        process.kill()
        return [], f"таймаут {CASE_TIMEOUT_SECONDS} с", None
    calls, result, used_model = parse_stream(stdout.decode("utf-8", "replace").splitlines())
    if result is None or result.get("is_error"):
        reason = (result or {}).get("result") or stderr.decode("utf-8", "replace")
        return calls, " ".join(str(reason).split())[:200] or f"код выхода {process.returncode}", used_model
    return calls, None, used_model


async def main(model: str, runs: int, limit: int | None, only: str | None = None, cases_file: Path | None = None) -> int:
    cases = [json.loads(line) for line in (cases_file or Path(__file__).with_name("cases.jsonl")).open()]
    cases = [c for c in cases if only.casefold() in c["q"].casefold()] if only else cases
    cases = cases[:limit] if limit else cases
    with tempfile.TemporaryDirectory(prefix="opti24-evals-") as tmp:
        workdir = Path(tmp)
        config_path = workdir / "mcp.json"
        server = Path(__file__).with_name("fake_mcp_server.py")
        config_path.write_text(
            json.dumps({"mcpServers": {SERVER_NAME: {"command": sys.executable, "args": [str(server)]}}}),
            encoding="utf-8",
        )
        failed_cases = 0
        shown_model = False
        for case in cases:
            failures: list[tuple[list[ToolInput], list[str]]] = []
            for _ in range(runs):
                calls, error, used_model = await run_once(case["q"], model, config_path, workdir)
                if used_model and not shown_model:
                    print(f"Модель Claude Code: {used_model}, прогонов на кейс: {runs}")
                    shown_model = True
                problems = [f"ошибка запуска: {error}"] if error else check_case(case, calls, date.today())
                if problems:
                    failures.append((calls, problems))
            failed_cases += bool(failures)
            print(f"{'FAIL' if failures else 'OK  '} {runs - len(failures)}/{runs}  {case['q']}")
            for calls, problems in failures:
                print(f"       вызовы: {_describe_calls(calls)}")
                print(f"       ошибки: {'; '.join(problems)}")
            if failures and all("ошибка запуска" in p[0] for _, p in failures) and "limit" in failures[0][1][0].lower():
                print("Остановлено: похоже, исчерпан лимит подписки.")
                return 2
    print(f"\nКейсов с ошибками: {failed_cases} из {len(cases)}")
    return 1 if failed_cases else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evals агента через Claude Code (подписка, без API-ключа)")
    parser.add_argument("--model", default="haiku", help="алиас или ID модели Claude Code (по умолчанию haiku)")
    parser.add_argument("--runs", type=int, default=1, help="сколько раз прогнать каждый кейс")
    parser.add_argument("--limit", type=int, default=None, help="взять только первые N кейсов")
    parser.add_argument("--only", default=None, help="только кейсы, в вопросе которых есть этот текст")
    parser.add_argument("--cases", type=Path, default=None, help="файл кейсов, например evals/paraphrases.jsonl")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs должно быть не меньше 1")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    raise SystemExit(asyncio.run(main(arguments.model, arguments.runs, arguments.limit, arguments.only, arguments.cases)))
