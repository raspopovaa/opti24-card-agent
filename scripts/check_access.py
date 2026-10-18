"""Быстрая проверка доступа к API Opti24: вход и один запрос статистики, без модели.

Оба запроса бесплатные (authUser и info не тарифицируются), поэтому пакет запросов не
тратится. Печатает только роль, режим и счётчики: без номеров карт и персональных данных.

    python scripts/check_access.py                 # DEMO
    python scripts/check_access.py --production    # рабочий договор
    python scripts/check_access.py --timeout 20    # короче ждать ответа сервера
    python scripts/check_access.py --contract-id ID  # если в аккаунте несколько договоров
    python scripts/check_access.py --production --list  # показать номера и ID договоров
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apisdkopti24 import (  # noqa: E402
    APIClient,
    APIError,
    ConnectionSettings,
    EnvironmentCredentialsProvider,
)
from apisdkopti24.errors import ContractSelectionError  # noqa: E402

from agent.config import default_env_file  # noqa: E402
from agent.contract import is_production, preferred_contract_id  # noqa: E402


async def check(production: bool, timeout: float, contract_id: str | None, show_list: bool = False) -> int:
    env_file = Path(default_env_file("production" if production else "demo")).expanduser()
    print(f"Окружение: {'РАБОЧЕЕ' if production else 'DEMO'}, файл {env_file}")
    conn = ConnectionSettings.from_env(env_file=env_file)
    if is_production(conn) != production:
        print("Адрес в .env не соответствует выбранному окружению: остановка.")
        return 1
    try:
        async with asyncio.timeout(timeout):
            async with APIClient(
                settings=conn,
                credentials_provider=EnvironmentCredentialsProvider.from_env(env_file=env_file),
            ) as client:
                try:
                    auth = (
                        await client.auth.auth_user(contract_id=contract_id or preferred_contract_id(env_file))
                    ).data
                except ContractSelectionError as exc:
                    # Сервер вход принял и вернул договоры: доступ есть, нужно лишь выбрать договор.
                    print(
                        f"OK   вход принят сервером, доступно договоров: {len(exc.available_contracts)}. "
                        "Чтобы продолжить, задайте --contract-id или API_CONTRACT_ID в .env."
                    )
                    if show_list:
                        for index, (contract_key, number) in enumerate(exc.available_contracts, start=1):
                            print(f"  {index}. {number}  (ID {contract_key})")
                    return 0
                print(f"OK   вход: роль {auth.role_name}, только чтение: {auth.read_only}, договоров: {len(auth.contracts)}")
                info = (await client.auth.get_info()).data
                print(f"OK   статистика: запросов в периоде {info.methods.all}, пакет {info.client_info.Queries}")
    except TimeoutError:
        print(f"FAIL нет ответа за {timeout:.0f} с: проверьте VPN и адрес (API принимает только RU, BY, KZ и ряд других стран)")
        return 1
    except APIError as exc:
        print(f"FAIL {type(exc).__name__}: {str(exc)[:300]}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Быстрая проверка доступа к API Opti24")
    parser.add_argument("--production", action="store_true", help="рабочий договор")
    parser.add_argument("--timeout", type=float, default=40, help="сколько секунд ждать ответа")
    parser.add_argument("--contract-id", help="ID договора, если их несколько")
    parser.add_argument("--list", action="store_true", help="показать номера и ID договоров")
    args = parser.parse_args()
    return asyncio.run(check(args.production, args.timeout, args.contract_id, args.list))


if __name__ == "__main__":
    raise SystemExit(main())
