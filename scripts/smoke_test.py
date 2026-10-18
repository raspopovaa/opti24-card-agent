"""Проверка инструментов и дашборда на настоящем API Opti24 (без модели ИИ).

Моки в tests/ проверяют логику, но форму ответов сервера они только предполагают.
Этот скрипт вызывает те же инструменты на DEMO или на рабочем договоре и сообщает,
какие из них отработали, сколько платных запросов потрачено и где форма ответа
не совпала с ожиданиями.

    # DEMO: бесплатно
    python scripts/smoke_test.py


Блокировка карт никогда не вызывается. Результаты печатаются в сокращённом виде,
номера карт маскируются. Комментарии к картам могут содержать имена и названия
машин, поэтому перед отправкой вывода кому-либо просмотрите его.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apisdkopti24 import APIClient, ConnectionSettings, EnvironmentCredentialsProvider  # noqa: E402

from agent.config import Mode, default_env_file  # noqa: E402
from agent.contract import (  # noqa: E402
    choose_in_terminal,
    is_production,
    open_session,
    preferred_contract_id,
)
from agent.dashboard import collect  # noqa: E402
from agent.safety import ApprovalGate, BillingBudget  # noqa: E402
from agent.tools import build_tools  # noqa: E402

DEMO_ENV = Path(default_env_file("demo")).expanduser()
PROD_ENV = Path(default_env_file("production")).expanduser()


async def never(_: str) -> bool:
    return False


def short(text: str, limit: int = 160) -> str:
    """Начало ответа для печати: длинные числа (номера карт) маскируются."""
    text = re.sub(r"\d{12,}", lambda m: m.group()[:4] + "…" + m.group()[-2:], text)
    return text if len(text) <= limit else text[:limit] + "…"


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", action="store_true", help="рабочий договор (платно)")
    parser.add_argument("--contract-id", help="ID договора (иначе из .env или выбор из списка)")
    parser.add_argument("--budget", type=int, default=8, help="предел платных запросов")
    args = parser.parse_args()

    env_file = PROD_ENV if args.production else DEMO_ENV
    if args.production:
        answer = input(f"Рабочий API: платные запросы, до {args.budget} шт. Продолжить? [да/нет]: ")
        if answer.strip().lower() not in {"да", "y", "yes"}:
            return 1
    print(f"Окружение: {'РАБОЧЕЕ' if args.production else 'DEMO'}, файл {env_file}")

    failures = 0
    conn = ConnectionSettings.from_env(env_file=env_file)
    if is_production(conn) != args.production:
        print("Адрес в .env не соответствует флагу --production: остановка.")
        return 1
    async with APIClient(
        settings=conn,
        credentials_provider=EnvironmentCredentialsProvider.from_env(env_file=env_file),
    ) as client:
        try:
            await open_session(
                client,
                preferred=args.contract_id or preferred_contract_id(env_file),
                production=args.production,
                choose=choose_in_terminal,
            )
            budget = BillingBudget(limit=args.budget)
            gate = ApprovalGate(Mode.READ_ONLY, never, Path("./smoke-audit.jsonl"))
            tools = {t.name: t for t in build_tools(client, budget, gate)}
            today = date.today()
            month_start = today.replace(day=1).isoformat()

            checks: list[tuple[str, dict[str, object]]] = [
                ("get_api_usage", {}),
                ("list_card_groups", {}),
                ("list_cards", {}),
                ("get_contract_summary", {}),
                ("spending_report", {"date_from": month_start, "date_to": today.isoformat()}),
                ("find_suspicious_fills", {"date_from": month_start, "date_to": today.isoformat()}),
            ]
            for name, tool_args in checks:
                try:
                    output = await tools[name].call(tool_args)
                except Exception as exc:  # noqa: BLE001
                    print(f"FAIL {name}: {type(exc).__name__}: {short(str(exc))}")
                    failures += 1
                    continue
                payload = json.loads(output)
                demo_405 = (
                    isinstance(payload, dict)
                    and payload.get("error") == "TransactionsUnavailableError"
                    and not args.production
                )
                if demo_405:
                    print(f"SKIP {name}: на DEMO операции недоступны (HTTP 405 стенда)")
                elif isinstance(payload, dict) and "error" in payload:
                    print(f"FAIL {name}: {payload['error']}: {short(str(payload.get('message')))}")
                    failures += 1
                else:
                    print(f"OK   {name}: {short(output)}")

            board = await collect(client, budget, today)
            for note in board.notes:
                demo_note = note.startswith("Операции на DEMO") and not args.production
                print(f"{'SKIP' if demo_note else 'WARN'} дашборд: {note}")
                failures += 0 if demo_note else 1
            print(f"{'OK  ' if board.contract else 'FAIL'} дашборд: блоков с данными "
                  f"{sum(bool(x) for x in (board.contract, board.cards, board.month, board.control, board.api))} из 5")
            print(f"\nПлатных запросов потрачено: {budget.spent} из {budget.limit}")
        finally:
            await client.auth.logoff()
    print("Итог:", "всё работает" if not failures else f"проблем: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
