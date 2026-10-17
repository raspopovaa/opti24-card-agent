"""Выбор договора и открытие сессии.

У одной учётной записи может быть несколько договоров (на DEMO их так и есть).
SDK не выбирает договор молча: без явного выбора он поднимает
``ContractSelectionError``. Порядок здесь такой же, как в проверках SDK
(``opti24-live-check``): ``auth_user()`` → если договоров несколько, берём
список из ошибки и выбираем один → ``auth_user(contract_id=...)``.

Договор берётся в таком порядке: ``AGENT_CONTRACT_ID`` → ``API_CONTRACT_ID`` из
окружения или из .env-файла Opti24 → интерактивный выбор. На рабочем API
договор нужно задавать явно, никакого выбора «по умолчанию».
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from urllib.parse import urlsplit

from apisdkopti24 import APIClient, ConnectionSettings
from apisdkopti24.errors import ContractSelectionError

PRODUCTION_HOST = "api.opti-24.ru"

ChooseContract = Callable[[list[tuple[str, str]]], Awaitable[str]]


def read_env_value(path: Path, key: str) -> str | None:
    """Значение ключа из .env-файла (без загрузки файла в окружение процесса)."""
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        name, sep, value = line.strip().partition("=")
        if sep and name.strip() == key:
            return value.strip().strip("'\"") or None
    return None


def preferred_contract_id(env_file: Path) -> str | None:
    return (
        os.environ.get("AGENT_CONTRACT_ID")
        or os.environ.get("API_CONTRACT_ID")
        or read_env_value(env_file, "API_CONTRACT_ID")
    )


def is_production(settings: ConnectionSettings) -> bool:
    return urlsplit(str(settings.base_url)).hostname == PRODUCTION_HOST


async def open_session(
    client: APIClient,
    *,
    preferred: str | None,
    production: bool,
    choose: ChooseContract | None,
) -> str:
    """Авторизуется и возвращает ID выбранного договора."""
    if preferred:
        await client.auth.auth_user(contract_id=preferred)
        return preferred
    if production:
        raise ContractSelectionError(
            "На рабочем API договор нужно задать явно: AGENT_CONTRACT_ID=<ID договора> "
            "или API_CONTRACT_ID в .env-файле Opti24."
        )
    try:
        response = await client.auth.auth_user()
    except ContractSelectionError as error:
        contracts = list(error.available_contracts)
        if choose is None or not contracts:
            names = ", ".join(f"{number} ({cid})" for cid, number in contracts) or "список пуст"
            raise ContractSelectionError(
                f"Несколько договоров: {names}. Задайте AGENT_CONTRACT_ID.",
                available_contracts=error.available_contracts,
            ) from error
        contract_id = await choose(contracts)
        await client.auth.auth_user(contract_id=contract_id)
        return contract_id
    return response.data.contracts[0].id


async def choose_in_terminal(contracts: list[tuple[str, str]]) -> str:
    """Интерактивный выбор: показываем номера договоров, а не только ID."""
    import asyncio

    print("Доступно несколько договоров:")
    for index, (_, number) in enumerate(contracts, start=1):
        print(f"  {index}. {number}")
    while True:
        answer = (await asyncio.to_thread(input, "Номер договора из списка: ")).strip()
        if answer.isdigit() and 1 <= int(answer) <= len(contracts):
            return contracts[int(answer) - 1][0]
        print("Введите число из списка.")
