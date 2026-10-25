"""Локальная база операций: период загружается из API один раз и читается из базы."""

import asyncio
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace as NS

from apisdkopti24.models.transactions import TransactionItemV2

from agent.db import OperationStore
from agent.safety import ApprovalGate, BillingBudget, Mode
from agent.tools.context import make_context

CONTRACT = "syn-contract-1"
LOADED = datetime(2026, 10, 10, 12, 0)


def _item(op_id: int, day: str, card_id: str = "syn-c-1", amount: str = "1000.00") -> TransactionItemV2:
    return TransactionItemV2.model_validate(
        {
            "id": op_id,
            "timestamp": f"{day}T08:00:00Z",
            "utc_time": f"{day}T05:00:00Z",
            "card_id": card_id,
            "poi_id": "syn-poi-1",
            "terminal_id": "syn-t-1",
            "type": "Purchase",
            "product_id": "syn-p-1",
            "product_name": "ДТ",
            "product_category_id": "syn-pc-1",
            "currency": "810",
            "check_id": op_id,
            "stor_transaction_id": None,
            "is_storno": False,
            "is_manual_corrention": False,
            "qty": 20,
            "price": "50.00",
            "price_no_discount": "52.00",
            "sum": amount,
            "sum_no_discount": "1040.00",
            "discount": "40.00",
            "exchange_rate": "1",
            "card_number": "9999000000000001",
            "payment_type": "Card",
        }
    )


def _raw(*items: TransactionItemV2) -> list[dict[str, object]]:
    return [item.model_dump(mode="json", by_alias=True) for item in items]


def _store(tmp_path: Path, ttl: timedelta = timedelta(hours=1)) -> OperationStore:
    return OperationStore(tmp_path / "agent.db", CONTRACT, ttl)


def test_saved_period_round_trips_sdk_objects(tmp_path: Path) -> None:
    store = _store(tmp_path)
    items = [_item(1, "2026-10-02"), _item(2, "2026-10-03", amount="-500.00")]
    store.save(date(2026, 10, 1), date(2026, 10, 9), None, _raw(*items), LOADED)

    saved = store.lookup(date(2026, 10, 1), date(2026, 10, 9), None, LOADED)

    assert saved is not None
    assert [TransactionItemV2.model_validate(raw) for raw in saved.items] == items
    assert saved.loaded_at == LOADED


def test_period_of_all_cards_covers_one_card_and_shorter_period(tmp_path: Path) -> None:
    store = _store(tmp_path)
    items = [_item(1, "2026-10-02"), _item(2, "2026-10-05", card_id="syn-c-2")]
    store.save(date(2026, 10, 1), date(2026, 10, 9), None, _raw(*items), LOADED)

    one_card = store.lookup(date(2026, 10, 1), date(2026, 10, 9), "syn-c-2", LOADED)
    shorter = store.lookup(date(2026, 10, 2), date(2026, 10, 3), None, LOADED)

    assert one_card is not None and [raw["id"] for raw in one_card.items] == [2]
    assert shorter is not None and [raw["id"] for raw in shorter.items] == [1]


def test_one_card_does_not_cover_all_cards_or_longer_period(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save(date(2026, 10, 1), date(2026, 10, 9), "syn-c-1", _raw(_item(1, "2026-10-02")), LOADED)

    assert store.lookup(date(2026, 10, 1), date(2026, 10, 9), None, LOADED) is None
    assert store.lookup(date(2026, 10, 1), date(2026, 10, 10), "syn-c-1", LOADED) is None


def test_open_period_expires_and_settled_period_does_not(tmp_path: Path) -> None:
    store = _store(tmp_path)
    # Октябрь загружен 10.10: период ещё открыт, копия живёт час.
    store.save(date(2026, 10, 1), date(2026, 10, 9), None, _raw(_item(1, "2026-10-02")), LOADED)
    # Сентябрь загружен 10.10: прошло больше трёх дней, сторно уже пришли.
    store.save(date(2026, 9, 1), date(2026, 9, 30), None, _raw(_item(2, "2026-09-15")), LOADED)
    later = LOADED + timedelta(hours=2)

    assert store.lookup(date(2026, 10, 1), date(2026, 10, 9), None, later) is None
    assert store.lookup(date(2026, 9, 1), date(2026, 9, 30), None, later) is not None


def test_reload_updates_rows_instead_of_duplicating(tmp_path: Path) -> None:
    store = _store(tmp_path)
    period = (date(2026, 10, 1), date(2026, 10, 9), None)
    store.save(*period, _raw(_item(1, "2026-10-02")), LOADED)
    store.save(*period, _raw(_item(1, "2026-10-02", amount="900.00")), LOADED + timedelta(minutes=5))

    saved = store.lookup(*period, LOADED + timedelta(minutes=5))

    assert saved is not None and [raw["sum"] for raw in saved.items] == ["900.00"]


def test_contract_id_is_not_written_to_the_database(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save(date(2026, 10, 1), date(2026, 10, 9), None, _raw(_item(1, "2026-10-02")), LOADED)
    store.close()

    assert CONTRACT.encode() not in (tmp_path / "agent.db").read_bytes()


class _Client:
    def __init__(self, items: list[TransactionItemV2], total: int | None = None) -> None:
        self.contract_id = CONTRACT
        self.calls = 0

        async def get_transactions_v2(**_: object) -> NS:
            self.calls += 1
            return NS(data=NS(result=items, total_count=len(items) if total is None else total))

        self.transactions = NS(get_transactions_v2=get_transactions_v2)


def _context(client: _Client, tmp_path: Path, budget: BillingBudget) -> object:
    gate = ApprovalGate(mode=Mode.READ_ONLY, ask=lambda _: False, audit_log=tmp_path / "audit.jsonl")
    return make_context(client, budget, gate, reports_dir=tmp_path / "reports")  # type: ignore[arg-type]


def test_next_session_reads_period_from_database(tmp_path: Path) -> None:
    client = _Client([_item(1, "2026-10-02"), _item(2, "2026-10-03")])
    first, second = BillingBudget(limit=10), BillingBudget(limit=10)

    loaded = asyncio.run(_context(client, tmp_path, first).transactions("2026-10-01", "2026-10-09", None))
    ctx = _context(client, tmp_path, second)
    again = asyncio.run(ctx.transactions("2026-10-01", "2026-10-09", None))

    assert (client.calls, first.spent, second.spent) == (1, 1, 0)
    assert again == loaded
    assert ctx.transactions_source("2026-10-01", "2026-10-09", None)["data_source"] == "локальная база"


def test_spending_report_names_the_data_source(tmp_path: Path) -> None:
    from agent.tools.operations import build

    client = _Client([_item(1, "2026-10-02")])
    tools = build(_context(client, tmp_path, BillingBudget(limit=10)))  # type: ignore[arg-type]

    answer = json.loads(asyncio.run(tools["spending_report"].call({"date_from": "2026-10-01", "date_to": "2026-10-09"})))

    assert answer["data_source"] == "API"
    assert "data_as_of" in answer


def test_partial_load_is_not_saved(tmp_path: Path) -> None:
    client = _Client([_item(1, "2026-10-02")], total=250)  # страниц больше, чем загружено

    store = _store(tmp_path)
    ctx = make_context(
        client,  # type: ignore[arg-type]
        BillingBudget(limit=1),
        ApprovalGate(mode=Mode.READ_ONLY, ask=lambda _: False, audit_log=tmp_path / "audit.jsonl"),
        reports_dir=tmp_path / "reports",
        store=store,
    )
    try:
        asyncio.run(ctx.transactions("2026-10-01", "2026-10-09", None))
    except Exception:  # noqa: BLE001 — бюджет кончился на второй странице
        pass

    assert store.lookup(date(2026, 10, 1), date(2026, 10, 9), None, LOADED) is None
