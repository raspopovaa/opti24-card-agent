"""Прогон кейсов на настоящем API Opti24 (по умолчанию DEMO): настоящая модель, только чтение.

Запуск (ключ модели и доступ к API Opti24 — как у самого агента; файл .env Opti24
берётся из OPTI24_ENV_FILE, по умолчанию DEMO):
    python evals/run_live.py --month 2026-09
    python evals/run_live.py --month 2026-09 --max-billable 60 --runs 1

Зачем, если есть run_evals.py: фейковый API не покажет, как агент ведёт себя на
ответах настоящего сервера — пустых периодах, балансе, настоящих картах и ошибках
API. Поэтому здесь кейсы без заранее известных ID: номер карты берётся из
list_cards (бесплатно), месяц с заправками задаёт --month.

Безопасность:
- режим всегда read_only: инструменты изменений агенту не видны, блокировки и
  лимиты не трогаются, на любое подтверждение ответ «нет»;
- тарифицируемые запросы считаются по всему прогону, и бюджет каждого вопроса не
  больше остатка: дойдя до --max-billable, прогон останавливается. До старта
  показывается остаток пакета договора, и при остатке меньше лимита прогона он не
  начинается.

Кейсы — evals/cases_live.jsonl, формат как у cases.jsonl. Плейсхолдеры:
{month_label} («сентябрь 2026»), {month_first}, {month_last}, {card_number}, {card_id},
{prev_month_first}, {prev_month_last}.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from apisdkopti24 import APIClient, ConnectionSettings, EnvironmentCredentialsProvider

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import Mode, Settings  # noqa: E402
from agent.contract import is_production, open_session, preferred_contract_id  # noqa: E402
from agent.core import FuelCardAgent  # noqa: E402
from agent.providers import get_provider  # noqa: E402
from evals.run_evals import _describe_calls, check_case, questions  # noqa: E402

MONTHS = (
    "январь", "февраль", "март", "апрель", "май", "июнь",
    "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь",
)


def month_placeholders(month: str) -> dict[str, str]:
    """Границы и название месяца из ГГГГ-ММ."""
    first = date.fromisoformat(f"{month}-01")
    last = (first.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    return {
        "month_label": f"{MONTHS[first.month - 1]} {first.year}",
        "month_first": first.isoformat(),
        "month_last": last.isoformat(),
    }


def fill(case: dict[str, Any], values: dict[str, str]) -> dict[str, Any]:
    """Подставляет плейсхолдеры в вопросы кейса; аргументы сверяет check_case."""
    filled = dict(case)
    if "q" in case:
        filled["q"] = case["q"].format_map(values)
    if "turns" in case:
        filled["turns"] = [turn.format_map(values) for turn in case["turns"]]
    return filled


async def _never(_: str) -> bool:
    return False  # read_only: подтверждений быть не должно, а если будут — «нет»


def _settings_or_exit() -> Settings:
    settings = dataclasses.replace(Settings.from_env(), mode=Mode.READ_ONLY)
    if not settings.api_key:
        key_env = get_provider(settings.provider).key_env
        raise SystemExit(f"Не задан ключ провайдера {settings.provider}: переменная {key_env}.")
    return settings


def _trace(agent: FuelCardAgent) -> list[str]:
    """Короткие ответы инструментов: без них не понять, почему модель повторила вызов."""
    seen: list[str] = []
    for tool in agent.tools:
        original = tool.call

        async def call(tool_input: object, _original: Any = original, _name: str = tool.name) -> Any:
            result = await _original(tool_input)
            seen.append(f"{_name} → {' '.join(str(result).split())[:160]}")
            return result

        tool.call = call
    return seen


async def _tool(agent: FuelCardAgent, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(await agent.tools_by_name[name].call(args or {}))
    return result


async def main(month: str, max_billable: int, runs: int, cases_file: Path | None = None) -> int:
    settings = _settings_or_exit()
    cases = [json.loads(line) for line in (cases_file or Path(__file__).with_name("cases_live.jsonl")).open()]
    env_file = settings.opti24_env_file
    conn = ConnectionSettings.from_env(env_file=env_file)
    credentials = EnvironmentCredentialsProvider.from_env(env_file=env_file)
    async with APIClient(settings=conn, credentials_provider=credentials) as opti24:
        contract_id = await open_session(
            opti24, preferred=preferred_contract_id(env_file), production=is_production(conn), choose=None
        )
        probe = FuelCardAgent(settings=settings, opti24=opti24, ask_human=_never)
        usage = await _tool(probe, "get_api_usage")
        left = usage.get("left_in_period")
        print(f"API: {conn.base_url} · договор {contract_id} · модель {settings.provider}/{settings.model}")
        print(f"Пакет: использовано {usage.get('calls_in_period')} из {usage.get('monthly_quota')}, осталось {left}")
        if isinstance(left, int) and left < max_billable:
            raise SystemExit(
                f"В пакете осталось {left} запросов, а прогону разрешено до {max_billable}. "
                "Уменьшите --max-billable или запустите в следующем месяце."
            )
        cards = (await _tool(probe, "list_cards")).get("cards") or []
        if not cards:
            raise SystemExit("На договоре нет карт: кейсам с картой нечего проверять.")
        values = {
            **month_placeholders(month),
            "card_number": str(cards[0]["number"]),
            "card_id": str(cards[0]["id"]),
        }
        print(f"Месяц: {values['month_label']} · карта {values['card_number']} (ID {values['card_id']})\n")

        spent = 0
        failed_cases = 0
        for raw in cases:
            case = fill(raw, values)
            failures: list[tuple[str, list[str]]] = []
            for _ in range(runs):
                if spent >= max_billable:
                    print(f"\nОстановлено: потрачено {spent} тарифицируемых запросов (предел {max_billable}).")
                    return failed_cases + 1
                agent = FuelCardAgent(settings=settings, opti24=opti24, ask_human=_never)
                agent.budget.limit = min(agent.budget.limit, max_billable - spent)  # предел на весь прогон
                trace = _trace(agent)
                answer = ""
                for question in questions(case):
                    answer = await agent.ask(question)
                spent += agent.budget.spent
                problems = check_case(case, agent.tool_inputs, date.today(), answer, [], values)
                problems += agent.last_check.problems()
                if problems:
                    problems.append("ответ: " + " ".join(answer.split())[:400])
                    failures.append((_describe_calls(agent.tool_inputs), problems + [f"\n         {t}" for t in trace]))
            status = "OK  " if not failures else "FAIL"
            print(f"{status} {runs - len(failures)}/{runs}  {' → '.join(questions(case))}")
            for calls, problems in failures:
                print(f"       вызовы: {calls}")
                print(f"       ошибки: {'; '.join(problems)}")
            failed_cases += bool(failures)
        print(f"\nКейсов с ошибками: {failed_cases} из {len(cases)} · тарифицируемых запросов: {spent}")
        return failed_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Кейсы агента на настоящем API Opti24 (по умолчанию DEMO), только чтение")
    parser.add_argument("--month", required=True, help="месяц с заправками, ГГГГ-ММ")
    parser.add_argument("--max-billable", type=int, default=80, help="предел тарифицируемых запросов на прогон")
    parser.add_argument("--runs", type=int, default=1, help="прогонов на кейс")
    parser.add_argument("--cases", type=Path, help="файл кейсов (по умолчанию cases_live.jsonl)")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    raise SystemExit(
        asyncio.run(main(arguments.month, arguments.max_billable, arguments.runs, arguments.cases)) and 1
    )
