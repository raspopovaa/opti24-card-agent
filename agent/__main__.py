"""Запуск: ``python -m agent`` — дашборд договора и диалог в терминале.

Примеры:
    python -m agent                              # DeepSeek-V4-Flash в Cloud.ru (ключ в AGENT_API_KEY)
    python -m agent --provider claude            # Claude Haiku 5.5 (ключ в ANTHROPIC_API_KEY)
    python -m agent --provider deepseek          # ключ в DEEPSEEK_API_KEY
    python -m agent --provider yandex            # YANDEX_API_KEY и YANDEX_FOLDER_ID
    python -m agent --provider ollama --model qwen3:8b   # локально, бесплатно
    python -m agent --no-dashboard
    python -m agent --demo                       # синтетический договор, без ключа Opti24
"""

from __future__ import annotations

import argparse
import asyncio
import os
from contextlib import AbstractAsyncContextManager
from datetime import date

from apisdkopti24 import APIClient, ConnectionSettings, EnvironmentCredentialsProvider

from .config import Settings
from .contract import choose_in_terminal, is_production, open_session, preferred_contract_id
from .core import FuelCardAgent
from .dashboard import collect, render
from .demo import demo_banner, demo_client
from .providers import PROVIDERS


async def _confirm(question: str) -> bool:
    answer = await asyncio.to_thread(input, f"\n⚠ {question}")
    return answer.strip().lower() in {"да", "д", "y", "yes"}


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m agent", description=__doc__.split("\n")[0])
    parser.add_argument("--provider", choices=sorted(PROVIDERS), help="провайдер модели")
    parser.add_argument("--model", help="имя модели (по умолчанию — дешёвая модель провайдера)")
    parser.add_argument("--contract-id", help="ID договора (иначе из AGENT_CONTRACT_ID, .env или выбор)")
    parser.add_argument("--no-dashboard", action="store_true", help="не показывать дашборд")
    parser.add_argument("--demo", action="store_true", help="демо-режим: синтетический договор, без API Opti24")
    return parser.parse_args()


async def main() -> None:
    args = _parse()
    # Флаги важнее переменных окружения; Settings читает только окружение.
    if args.provider:
        os.environ["AGENT_PROVIDER"] = args.provider
    if args.model:
        os.environ["AGENT_MODEL"] = args.model
    settings = Settings.from_env()
    env_file = settings.opti24_env_file
    client_cm: AbstractAsyncContextManager[APIClient]
    if args.demo:
        client_cm = demo_client()
    else:
        conn = ConnectionSettings.from_env(env_file=env_file)
        credentials = EnvironmentCredentialsProvider.from_env(env_file=env_file)
        client_cm = APIClient(settings=conn, credentials_provider=credentials)

    async with client_cm as opti24:
        if args.demo:
            print(demo_banner())
        await open_session(
            opti24,
            preferred=None if args.demo else (args.contract_id or preferred_contract_id(env_file)),
            production=False if args.demo else is_production(conn),
            choose=choose_in_terminal,
        )
        agent = FuelCardAgent(settings=settings, opti24=opti24, ask_human=_confirm)
        if not args.no_dashboard:
            render(await collect(opti24, agent.budget, date.today()))
        print(
            f"\nПровайдер: {settings.provider} · модель: {settings.model} · "
            f"режим: {settings.mode} · пустая строка — выход"
        )
        try:
            while question := (await asyncio.to_thread(input, "\nВы: ")).strip():
                print(f"\nАгент: {await agent.ask(question)}")
        finally:
            u = agent.usage
            print(
                f"\nТокены: вход {u.input_tokens}, из кеша {u.cache_read_tokens}, "
                f"выход {u.output_tokens}; тарифицируемых вызовов API: {agent.budget.spent}"
            )
            await opti24.auth.logoff()


if __name__ == "__main__":
    asyncio.run(main())
