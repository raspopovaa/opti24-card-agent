"""Дашборд договора в терминале: баланс, расход месяца, карты, операции, контроль.

Строится кодом без модели: не тратит токены и работает без ключей ИИ-провайдера.
Запуск отдельно: ``python -m agent.dashboard`` (нужны только учётные данные Opti24).

Расход запросов API за один показ:
* 1 платный запрос — данные договора (баланс, расход, число карт);
* до ``max_operation_pages`` платных запросов — операции с начала месяца
  (страница = 100 операций), если у договора их больше, дашборд честно пишет,
  что показаны первые N;
* бесплатные запросы: число запросов в пакете, группы карт.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from datetime import date

from apisdkopti24 import APIClient, APIError, ConnectionSettings, EnvironmentCredentialsProvider
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table

from .analysis import AnomalyRules, charts, find_anomalies, summarize
from .config import Settings
from .contract import choose_in_terminal, is_production, open_session, preferred_contract_id
from .money import rub, to_decimal
from .safety import BillingBudget, BudgetExceededError
from .tools import TransactionsUnavailableError, load_transactions

DEFAULT_PAGES = 3


@dataclass(slots=True)
class Dashboard:
    today: date
    contract: dict[str, str] = field(default_factory=dict)
    cards: dict[str, str] = field(default_factory=dict)
    month: dict[str, object] = field(default_factory=dict)
    control: dict[str, object] = field(default_factory=dict)
    api: dict[str, object] = field(default_factory=dict)
    charts: dict[str, object] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _liters_change(now: str, before: str) -> str:
    current, previous = to_decimal(now), to_decimal(before)
    if previous == 0:
        return "—"
    percent = (current - previous) / previous * 100
    return f"{percent:+.0f}% к прошлому месяцу"


async def collect(
    client: APIClient, budget: BillingBudget, today: date, max_operation_pages: int = DEFAULT_PAGES
) -> Dashboard:
    board = Dashboard(today=today)

    try:
        budget.spend()
        contract = (await client.contracts.get_contract_data()).data
    except (APIError, BudgetExceededError) as exc:
        board.notes.append(f"Договор не получен: {type(exc).__name__}")
    else:
        info, balance, cards = contract.contractData, contract.balanceData, contract.cardsData
        board.contract = {
            "Договор": info.contract_number,
            "Статус": info.contract_status_name,
            "Доступный остаток": rub(to_decimal(balance.available_amount)),
            "Собственные средства": rub(to_decimal(balance.balance)),
            "Последний платёж": (
                f"{rub(to_decimal(balance.last_payment_sum))} от {balance.last_payment_date}"
                if balance.last_payment_sum
                else "—"
            ),
        }
        board.cards = {
            "Карт всего": cards.cards_quantity_all,
            "Карт активно": cards.cards_quantity_active,
            "Групп карт": cards.card_groups_quantity_all or "—",
        }
        board.month = {
            "Расход за месяц": rub(to_decimal(balance.consumption_for_month)),
            "Литров за месяц": balance.consumption_for_month_volume,
            "Динамика": _liters_change(
                balance.consumption_for_month_volume, balance.consumption_for_prev_month_volume
            ),
        }

    first_day = today.replace(day=1).isoformat()
    try:
        items, total = await load_transactions(
            client, budget, first_day, today.isoformat(), max_pages=max_operation_pages
        )
    except TransactionsUnavailableError:
        board.notes.append("Операции на DEMO недоступны (HTTP 405 стенда): проверьте на рабочем договоре.")
    except (APIError, BudgetExceededError) as exc:
        board.notes.append(f"Операции не получены: {type(exc).__name__}")
    else:
        summary = summarize(items)
        board.charts = charts(items)
        board.month["Операций (покупок)"] = str(summary["operations"])
        board.month["Топливо"] = summary["by_product"]
        board.month["Топ карт"] = summary["top_cards"]
        anomalies = find_anomalies(items, AnomalyRules(), top=5)
        board.control = {
            "Проверено операций": anomalies["checked_operations"],
            "Подозрительных": anomalies["suspicious_total"],
            "Примеры": anomalies["shown"],
        }
        if total > len(items):
            board.notes.append(
                f"Показаны первые {len(items)} операций из {total}: лимит запросов дашборда. "
                "Остальное доступно вопросом агенту."
            )

    try:
        info_data = (await client.auth.get_info()).data
    except APIError as exc:
        board.notes.append(f"Статистика API не получена: {type(exc).__name__}")
    else:
        board.api = {
            "Запросов в пакете": info_data.client_info.Queries,
            "Вызовов API за месяц": info_data.methods.all,
            "Платных вызовов агента в сессии": budget.spent,
        }
    return board


def _kv(title: str, data: dict[str, object], style: str) -> Panel:
    table = Table.grid(padding=(0, 2))
    for key, value in data.items():
        if isinstance(value, (list, dict)):
            continue
        table.add_row(f"[dim]{key}[/]", str(value))
    return Panel(table, title=f"[bold]{title}[/]", border_style=style, expand=True)


def render(board: Dashboard, console: Console | None = None) -> None:
    console = console or Console()
    console.print(
        f"[bold #153f34]Дашборд договора[/]  [dim]{board.today:%d.%m.%Y}[/]  "
        "[dim](данные с начала месяца)[/]"
    )
    console.print(
        Group(
            _kv("Договор и баланс", board.contract, "#153f34"),
            _kv("Карты", board.cards, "#153f34"),
            _kv("Месяц", board.month, "#153f34"),
        )
    )

    fuel = board.month.get("Топливо")
    if isinstance(fuel, list) and fuel:
        table = Table(title="Топливо", title_justify="left")
        for column in ("Продукт", "Литры", "Сумма", "Цена литра"):
            table.add_column(column, justify="right" if column != "Продукт" else "left")
        for row in fuel:
            table.add_row(row["product"], row["liters"], row["sum"], row["avg_price_per_liter"] or "—")
        console.print(table)

    top = board.month.get("Топ карт")
    if isinstance(top, list) and top:
        table = Table(title="Топ-5 карт по расходу", title_justify="left")
        for column in ("Карта", "Операций", "Сумма"):
            table.add_column(column, justify="right" if column != "Карта" else "left")
        for row in top:
            table.add_row(row["card_id"], str(row["operations"]), row["sum"])
        console.print(table)

    if board.control:
        suspicious = board.control["Подозрительных"]
        style = "red" if suspicious else "green"
        lines = [f"Проверено операций: {board.control['Проверено операций']}, "
                 f"подозрительных: [{style}]{suspicious}[/]"]
        for item in board.control["Примеры"]:  # type: ignore[union-attr]
            lines.append(f"• {item['date']}  {item['sum']}  " + "; ".join(item["reasons"]))
        console.print(Panel("\n".join(lines), title="[bold]Контроль заправок[/]", border_style=style))

    if board.api:
        console.print(_kv("Запросы API", board.api, "grey50"))
    for note in board.notes:
        console.print(f"[yellow]⚠ {note}[/]")


async def main() -> None:
    settings = Settings.from_env()
    env_file = settings.opti24_env_file
    conn = ConnectionSettings.from_env(env_file=env_file)
    async with APIClient(
        settings=conn,
        credentials_provider=EnvironmentCredentialsProvider.from_env(env_file=env_file),
    ) as client:
        await open_session(
            client,
            preferred=preferred_contract_id(env_file),
            production=is_production(conn),
            choose=choose_in_terminal,
        )
        budget = BillingBudget(limit=settings.max_billable_calls)
        try:
            render(await collect(client, budget, date.today()))
        finally:
            await client.auth.logoff()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
