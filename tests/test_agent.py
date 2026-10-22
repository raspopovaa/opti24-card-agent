"""Тесты без сети и без ключей: фейковый клиент Opti24, настоящий код инструментов."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from agent.analysis import find_anomalies, summarize
from agent.config import Mode
from agent.money import rub, to_decimal
from agent.safety import ApprovalGate, BillingBudget
from agent.tools import build_tools


def tx(i: int, hour: int, price: str, qty: float, card: str = "c1", day: int = 1) -> NS:
    return NS(
        id=i,
        timestamp=datetime(2026, 9, day, hour, 0),
        card_id=card,
        card_number=f"782601000000{i:04d}",
        poi_id="p",
        terminal_id="t1",
        check_id=i,
        type="P",
        product_name="АИ-95",
        qty=qty,
        price=price,
        sum=str(Decimal(price) * Decimal(str(qty))),
        discount="0",
        is_storno=False,
        is_manual_correction=False,
    )


class Dump:
    """Мини-замена Pydantic-модели: поля и model_dump()."""

    def __init__(self, data: dict[str, object]) -> None:
        self._data = data

    def __getattr__(self, name: str) -> object:
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(name) from None

    def model_dump(self) -> dict[str, object]:
        return self._data


def test_money_has_no_float_error() -> None:
    assert to_decimal(0.1) + to_decimal(0.2) == Decimal("0.3")
    assert rub(Decimal("12345.6")) == "12 345,60 ₽"


def test_anomalies_find_expensive_night_and_big_fills() -> None:
    items = [tx(i, 12, "60.00", 40, day=i) for i in range(1, 8)]
    items += [tx(100, 2, "70.00", 40), tx(101, 12, "60.00", 200, day=9)]
    result = find_anomalies(items)
    reasons = {f["transaction_id"]: " ".join(f["reasons"]) for f in result["shown"]}
    assert "ночная" in reasons["100"] and "выше медианы" in reasons["100"]
    assert "объём" in reasons["101"]
    assert result["suspicious_total"] == 2


def test_summary_ignores_storno_and_refunds() -> None:
    good = tx(1, 12, "60.00", 10)
    storno = tx(2, 12, "60.00", 10)
    storno.is_storno = True
    summary = summarize([good, storno])
    assert summary["operations"] == 1
    assert summary["total"] == "600,00 ₽"


def test_budget_blocks_overspend() -> None:
    budget = BillingBudget(limit=2)
    budget.spend(2)
    with pytest.raises(RuntimeError):
        budget.spend()


class FakeOpti24:
    def __init__(self, transactions: list[NS]) -> None:
        self.blocked: list[str] = []
        self.contract_id = "c1"
        self.comments: list[tuple[str, str]] = []
        self.saved_limits: list[NS] = []
        self.saved_regions: list[NS] = []
        self.saved_restrictions: list[NS] = []
        self.ordered_reports: list[dict[str, object]] = []
        page = NS(data=NS(result=transactions, total_count=len(transactions)))

        async def get_transactions_v2(**_: object) -> NS:
            return page

        async def get_card_transactions_v2(**_: object) -> NS:
            return page

        async def get_transaction_detail(**_: object) -> NS:
            item = transactions[0] if transactions else tx(999, 12, "60.00", 10)
            return NS(data=NS(result=[item], total_count=1))

        async def block_card(*, card_ids: list[str], block: bool) -> NS:
            self.blocked += card_ids if block else []
            return NS(data=list(card_ids))

        card = NS(
            id="c1",
            number="7826010000000001",
            status="Active",
            status_name="Активна",
            group_name="Грузовые",
            comment="Газель, Иванов",
        )
        detail = NS(
            id="c1",
            number="7826010000000001",
            status="Active",
            comment="Газель, Иванов",
            product="limit",
            carrier="Plastic",
            available="9000.00",
            currency="RUB",
            date_last_usage=None,
            servicecenter_last_usage_name="АЗС-1",
            mpc=False,
            pin_reset=3,
            pin_counter=0,
        )
        driver = NS(
            last_name="Иванов",
            first_name="Иван",
            middle_name=None,
            position="Водитель",
            role="Driver",
            mobile_phone="+7 900 000-00-00",
        )
        limit = NS(
            id="l1",
            card_id="c1",
            group_id=None,
            productType="1-CK231",
            productGroup=None,
            amount=None,
            sum=Dump({"value": 9000.0, "used": 231.46, "currency": "810"}),
            term=None,
            transactions=None,
            time=NS(number=1, type=5),
        )

        async def get_card_detail(**_: object) -> NS:
            return NS(data=NS(result=[detail], total_count=1))

        async def get_card_drivers(**_: object) -> NS:
            return NS(data=NS(result=[driver], total_count=1))

        async def get_cards_v2(**_: object) -> NS:
            return NS(data=NS(result=[card], total_count=1))

        async def get_limits(**_: object) -> NS:
            return NS(data=NS(result=[limit], total_count=1))

        async def get_dictionary(**_: object) -> NS:
            item = NS(id="1-CK231", name="Топливо", value=None)
            return NS(data=NS(result=[item], total_count=1))

        async def get_contract_data(**_: object) -> NS:
            contract_data = NS(
                contract_number="D-1",
                contract_status_name="Действует",
                discount_scheme=None,
                balance_amount_limiter=None,
                current_amount_limiter="14277644.68",
                credit_limit=None,
            )
            balance = NS(
                available_amount="123139.19",
                own_balance="1150502.89",
                consumption_for_month="1202214.25",
                consumption_for_month_volume="200",
                consumption_for_prev_month_volume="120000",
            )
            cards = NS(cards_quantity_all="3", cards_quantity_active="2")
            return NS(data=NS(payment_term_id=None, payment_scheme_id=None, contractData=contract_data, balanceData=balance, cardsData=cards))

        async def get_payments(**_: object) -> NS:
            item = NS(
                id="p1",
                date="2026-10-02T10:00:00",
                amount="1000.00",
                amount_client="1000.00",
                currency="810;RUR",
                description="Оплата по счёту",
                payment_name="Платёж",
                payment_number="1",
            )
            return NS(data=NS(result=[item], total_count=1))

        async def get_invoices(**_: object) -> NS:
            item = NS(
                id="i1",
                ref_number="СЧ-1",
                date_start="2026-10-01",
                date_end="2026-10-31",
                amount="5000.00",
                paid_amount="0.00",
                status="OPEN",
                comment=None,
            )
            return NS(data=NS(result=[item], total_count=1))

        async def get_documents(**_: object) -> NS:
            item = NS(
                id="d1",
                name="УПД",
                number="1",
                date=1759363200,
                total=1200.0,
                vat=200.0,
                sum=1000.0,
                currency="руб.",
                consignee="ООО Ромашка",
            )
            return NS(data=NS(result=[item], total_count=1))

        async def auth_user(**_: object) -> NS:
            contract = NS(id="c1", number="D-1", cards_count=3)
            data = NS(
                contracts=[contract],
                org_name="ООО Ромашка",
                role_name="Администратор",
                read_only=False,
                email="user@example.ru",
            )
            return NS(data=data)

        async def get_info(**_: object) -> NS:
            data = NS(
                from_=datetime(2026, 10, 1),
                to=datetime(2026, 11, 1),
                client_info=NS(Queries=None),
                methods=NS(all=10),
            )
            return NS(data=data)

        async def set_card_comment(**kwargs: object) -> NS:
            self.comments.append((str(kwargs["card_id"]), str(kwargs["comment"])))
            return NS(data=True)

        async def set_limit(**kwargs: object) -> NS:
            self.saved_limits.extend(kwargs["limits"])
            return NS(data=["l1"])

        async def get_region_limits(**_: object) -> NS:
            item = NS(
                id="r1", card_id="c1", group_id=None, country="RUS", region="MOS",
                service_center=None, partner=None, limit_type=2, date="2026-10-01",
            )
            return NS(data=NS(result=[item], total_count=1))

        async def set_region_limit(**kwargs: object) -> NS:
            self.saved_regions.extend(kwargs["region_limits"])
            return NS(data=["r1"])

        async def get_restrictions(**_: object) -> NS:
            item = NS(
                id="s1", card_id="c1", group_id=None, productType="1-CK231",
                productTypeName="Топливо", productGroup=None, productGroupName=None,
                restriction_type=2, date="2026-10-01",
            )
            return NS(data=NS(result=[item], total_count=1))

        async def set_restriction(**kwargs: object) -> NS:
            self.saved_restrictions.extend(kwargs["restrictions"])
            return NS(data=["s1"])

        async def order_report_v1(**kwargs: object) -> NS:
            self.ordered_reports.append(dict(kwargs))
            return NS(data=["job1"])

        async def download_report_file_v1_to(*, job_id: str, destination: Path) -> Path:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"xlsx")
            return destination

        async def remove_limit(**_: object) -> NS:
            return NS(data=True)

        async def remove_region_limit(**_: object) -> NS:
            return NS(data=True)

        async def remove_restriction(**_: object) -> NS:
            return NS(data=True)

        async def get_report_job_list_v1(**_: object) -> NS:
            item = NS(
                job_id="job1", date="2026-10-01", report_name="Транзакции",
                report_format="xlsx", contract_id="c1", client_id="cl1", user_id="u1",
            )
            return NS(data=[item])

        self.transactions = NS(
            get_transactions_v2=get_transactions_v2,
            get_card_transactions_v2=get_card_transactions_v2,
            get_transaction_detail=get_transaction_detail,
        )
        self.cards = NS(
            block_card=block_card,
            get_card_detail=get_card_detail,
            get_card_drivers=get_card_drivers,
            get_cards_v2=get_cards_v2,
            set_card_comment=set_card_comment,
        )
        self.limits = NS(
            get_limits=get_limits, set_limit=set_limit, remove_limit=remove_limit
        )
        self.region_limits = NS(
            get_region_limits=get_region_limits,
            set_region_limit=set_region_limit,
            remove_region_limit=remove_region_limit,
        )
        self.restrictions = NS(
            get_restrictions=get_restrictions,
            set_restriction=set_restriction,
            remove_restriction=remove_restriction,
        )
        self.reports = NS(
            order_report_v1=order_report_v1,
            get_report_job_list_v1=get_report_job_list_v1,
            download_report_file_v1_to=download_report_file_v1_to,
        )
        self.dictionaries = NS(get_dictionary=get_dictionary)
        self.contracts = NS(
            get_contract_data=get_contract_data,
            get_payments=get_payments,
            get_invoices=get_invoices,
            get_documents=get_documents,
        )
        self.auth = NS(auth_user=auth_user, get_info=get_info)


def make_tools(
    mode: Mode, answer: bool, tmp: Path, txs: list[NS] | None = None, budget_limit: int = 3
):
    fake = FakeOpti24(txs or [])
    budget = BillingBudget(limit=budget_limit)

    async def ask(_: str) -> bool:
        return answer

    gate = ApprovalGate(mode=mode, ask=ask, audit_log=tmp / "audit.jsonl")
    tools = {
        t.name: t
        for t in build_tools(fake, budget, gate, reports_dir=tmp / "reports")  # type: ignore[arg-type]
    }
    return fake, budget, tools


def test_block_runs_without_question_because_it_is_reversible(tmp_path: Path) -> None:
    fake, _, tools = make_tools(Mode.CONFIRM, answer=False, tmp=tmp_path)
    out = json.loads(asyncio.run(tools["block_cards"].call({"card_ids": ["a"], "reason": "тест"})))
    assert out["status"] == "blocked" and fake.blocked == ["a"]
    log = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    assert '"asked": false' in log and '"approved": true' in log


def test_irreversible_actions_still_need_a_yes(tmp_path: Path) -> None:
    fake, budget, tools = make_tools(Mode.CONFIRM, answer=False, tmp=tmp_path, budget_limit=5)
    removed = json.loads(asyncio.run(tools["remove_limit"].call({"limit_id": "l1"})))
    ordered = json.loads(
        asyncio.run(
            tools["order_fuel_report"].call({"date_from": "2026-09-01", "date_to": "2026-09-30", "card_ids": ["c1"]})
        )
    )
    assert removed["status"] == "declined" and ordered["status"] == "declined"
    assert fake.ordered_reports == [] and budget.spent == 0
    log = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    assert '"asked": true' in log and '"approved": false' in log


def test_block_works_after_yes_and_is_audited(tmp_path: Path) -> None:
    fake, _, tools = make_tools(Mode.CONFIRM, answer=True, tmp=tmp_path)
    asyncio.run(tools["block_cards"].call({"card_ids": ["a", "b"], "reason": "тест"}))
    assert fake.blocked == ["a", "b"]
    log = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    assert '"approved": true' in log and "тест" not in log  # причина и ID в журнал не попадают


def test_unblock_uses_same_tool(tmp_path: Path) -> None:
    fake, _, tools = make_tools(Mode.CONFIRM, answer=True, tmp=tmp_path)
    out = json.loads(
        asyncio.run(
            tools["block_cards"].call(
                {"card_ids": ["a"], "reason": "карта нашлась", "block": False}
            )
        )
    )
    assert out["status"] == "unblocked" and out["done"] == ["a"]


def test_read_only_mode_never_blocks_even_if_human_says_yes(tmp_path: Path) -> None:
    fake, _, tools = make_tools(Mode.READ_ONLY, answer=True, tmp=tmp_path)
    out = json.loads(asyncio.run(tools["block_cards"].call({"card_ids": ["a"], "reason": "x"})))
    assert out["status"] == "not_allowed" and fake.blocked == []


def test_report_rejects_long_period_without_spending_budget(tmp_path: Path) -> None:
    _, budget, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path)
    args = {"date_from": "2026-01-01", "date_to": "2026-06-30"}
    out = json.loads(asyncio.run(tools["spending_report"].call(args)))
    assert out["error"] == "bad_dates" and budget.spent == 0


def test_report_spends_one_call_per_page(tmp_path: Path) -> None:
    _, budget, tools = make_tools(
        Mode.READ_ONLY, answer=False, tmp=tmp_path, txs=[tx(1, 12, "60.00", 10)]
    )
    args = {"date_from": "2026-09-01", "date_to": "2026-09-30"}
    out = json.loads(asyncio.run(tools["spending_report"].call(args)))
    assert out["total"] == "600,00 ₽" and budget.spent == 1


def test_same_period_is_loaded_once_for_report_and_anomalies(tmp_path: Path) -> None:
    _, budget, tools = make_tools(
        Mode.READ_ONLY, answer=False, tmp=tmp_path, txs=[tx(1, 12, "60.00", 10)]
    )
    args = {"date_from": "2026-09-01", "date_to": "2026-09-30"}
    asyncio.run(tools["spending_report"].call(args))
    out = json.loads(asyncio.run(tools["find_suspicious_fills"].call(args)))
    asyncio.run(tools["spending_report"].call(args))
    assert out["checked_operations"] == 1 and budget.spent == 1


def test_block_dedupes_ids_and_shows_them_to_human(tmp_path: Path) -> None:
    fake = FakeOpti24([])
    asked: list[str] = []

    async def ask(text: str) -> bool:
        asked.append(text)
        return True

    gate = ApprovalGate(mode=Mode.CONFIRM, ask=ask, audit_log=tmp_path / "audit.jsonl")
    tools = {t.name: t for t in build_tools(fake, BillingBudget(limit=3), gate)}  # type: ignore[arg-type]
    out = json.loads(
        asyncio.run(tools["block_cards"].call({"card_ids": ["c7", "c7", " "], "reason": "x"}))
    )
    assert out["status"] == "blocked" and out["done"] == ["c7"] and out["not_done"] == []
    assert fake.blocked == ["c7"]
    assert asked == []  # блокировка обратима: вопроса нет, результат с ID возвращается агенту
    assert "c7" not in (tmp_path / "audit.jsonl").read_text(encoding="utf-8")


def test_limits_allow_contract_and_reject_two_targets(tmp_path: Path) -> None:
    _, _, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path)
    both = json.loads(
        asyncio.run(tools["get_limits"].call({"card_id": "c1", "group_id": "g1"}))
    )
    assert both["error"] == "bad_request"
    contract = json.loads(asyncio.run(tools["get_limits"].call({})))
    assert contract["total"] == 1 and contract["limits"][0]["product"] == "Топливо"


def test_account_context_has_role_and_read_only(tmp_path: Path) -> None:
    _, budget, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path)
    out = json.loads(asyncio.run(tools["get_account_context"].call({})))
    assert out["role"] == "Администратор" and out["read_only"] is False
    assert out["period"]["from"].startswith("2026-10-01")
    assert out["contract"]["number"] == "D-1" and budget.spent == 0


def test_contract_summary_uses_balance_data_and_none_limits(tmp_path: Path) -> None:
    _, budget, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path)
    out = json.loads(asyncio.run(tools["get_contract_summary"].call({})))
    assert out["available"] == "123 139,19 ₽"
    assert out["own_funds"] == "1 150 502,89 ₽"
    assert out["month_spent"] == "1 202 214,25 ₽"
    assert out["limit_available"] is None and out["credit_limit"] is None
    assert budget.spent == 1


def test_limits_include_product_period_and_usage(tmp_path: Path) -> None:
    _, budget, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path)
    out = json.loads(asyncio.run(tools["get_limits"].call({"card_id": "c1"})))
    item = out["limits"][0]
    assert item["product"] == "Топливо"
    assert item["period"] == {"number": 1, "type": 5}
    assert item["sum"]["used"] == 231.46
    assert item["sum_left"] == "8768.54" and item["amount_left"] is None  # остаток считает код
    assert out["total"] == 1 and budget.spent == 0


def test_card_profile_includes_driver_and_limits(tmp_path: Path) -> None:
    _, budget, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path)
    out = json.loads(asyncio.run(tools["get_card_profile"].call({"card_id": "c1"})))
    assert out["number"] == "7826010000000001"
    assert out["group"] == "Грузовые"
    assert out["drivers"][0]["name"] == "Иванов Иван"
    assert out["limits"][0]["product"] == "Топливо"
    assert budget.spent == 2


def test_list_transactions_filters_and_reuses_cache(tmp_path: Path) -> None:
    fuel = tx(1, 12, "60.00", 10)
    diesel = tx(2, 13, "70.00", 40)
    diesel.product_name = "ДТ"
    _, budget, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path, txs=[fuel, diesel])
    args = {"date_from": "2026-09-01", "date_to": "2026-09-30"}
    out = json.loads(asyncio.run(tools["list_transactions"].call({**args, "product": "ДТ"})))
    assert out["total"] == 1 and out["transactions"][0]["product"] == "ДТ"
    asyncio.run(tools["spending_report"].call(args))
    assert budget.spent == 1  # операции за один период загружены один раз


def test_transaction_detail_is_free(tmp_path: Path) -> None:
    _, budget, tools = make_tools(
        Mode.READ_ONLY, answer=False, tmp=tmp_path, txs=[tx(1, 12, "60.00", 10)]
    )
    out = json.loads(asyncio.run(tools["get_transaction_detail"].call({"transaction_id": "1"})))
    assert out["sum"] == "600,00 ₽" and out["card_number"].startswith("7826")
    assert budget.spent == 0


def test_finance_payments_are_paid_and_documents_free(tmp_path: Path) -> None:
    _, budget, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path)
    payments = json.loads(
        asyncio.run(
            tools["get_finance"].call(
                {"kind": "payments", "date_from": "2026-10-01", "date_to": "2026-10-31"}
            )
        )
    )
    assert payments["total"] == 1 and budget.spent == 1
    documents = json.loads(
        asyncio.run(
            tools["get_finance"].call(
                {"kind": "documents", "date_from": "2026-10-01", "date_to": "2026-10-31"}
            )
        )
    )
    assert documents["total"] == 1 and budget.spent == 1  # документы бесплатны


def test_block_reports_partial_result(tmp_path: Path) -> None:
    fake = FakeOpti24([])

    async def partial(*, card_ids: list[str], block: bool) -> NS:
        fake.blocked += card_ids
        return NS(data=card_ids[:1])

    fake.cards.block_card = partial  # type: ignore[attr-defined]

    async def ask(_: str) -> bool:
        return True

    gate = ApprovalGate(mode=Mode.CONFIRM, ask=ask, audit_log=tmp_path / "audit.jsonl")
    tools = {t.name: t for t in build_tools(fake, BillingBudget(limit=3), gate)}  # type: ignore[arg-type]
    out = json.loads(
        asyncio.run(tools["block_cards"].call({"card_ids": ["a", "b"], "reason": "тест"}))
    )
    assert out["status"] == "partial" and out["done"] == ["a"] and out["not_done"] == ["b"]


def test_set_limit_sends_model_after_confirmation(tmp_path: Path) -> None:
    fake, budget, tools = make_tools(Mode.CONFIRM, answer=True, tmp=tmp_path)
    out = json.loads(
        asyncio.run(
            tools["set_limit"].call(
                {
                    "card_id": "c1",
                    "product_type": "1-CK231",
                    "liters": 100,
                    "period_number": 1,
                    "period_type": 3,
                }
            )
        )
    )
    assert out["status"] == "saved" and out["limit_ids"] == ["l1"]
    item = fake.saved_limits[0]
    assert item.amount.value == 100 and item.time.type == 3 and budget.spent == 1


def test_set_limit_is_blocked_in_read_only(tmp_path: Path) -> None:
    fake, _, tools = make_tools(Mode.READ_ONLY, answer=True, tmp=tmp_path)
    out = json.loads(
        asyncio.run(
            tools["set_limit"].call(
                {"card_id": "c1", "rubles": 5000, "period_type": 5}
            )
        )
    )
    assert out["status"] == "not_allowed" and fake.saved_limits == []


def test_set_card_comment_validates_length_and_saves(tmp_path: Path) -> None:
    fake, budget, tools = make_tools(Mode.CONFIRM, answer=True, tmp=tmp_path)
    bad = json.loads(
        asyncio.run(tools["set_card_comment"].call({"card_id": "c1", "comment": "x" * 91}))
    )
    assert bad["error"] == "bad_request" and fake.comments == []
    good = json.loads(
        asyncio.run(tools["set_card_comment"].call({"card_id": "c1", "comment": "Газель, Иванов"}))
    )
    assert good["status"] == "updated" and fake.comments == [("c1", "Газель, Иванов")]
    assert budget.spent == 1


def test_set_region_limit_and_restriction(tmp_path: Path) -> None:
    fake, budget, tools = make_tools(Mode.CONFIRM, answer=True, tmp=tmp_path)
    region = json.loads(
        asyncio.run(
            tools["set_region_limit"].call(
                {"card_id": "c1", "country": "RUS", "region": "MOS", "limit_type": 2}
            )
        )
    )
    assert region["status"] == "saved" and fake.saved_regions[0].limit_type == 2
    restriction = json.loads(
        asyncio.run(
            tools["set_restriction"].call(
                {"card_id": "c1", "product_type": "1-CK231", "restriction_type": 2}
            )
        )
    )
    assert restriction["status"] == "saved"
    assert fake.saved_restrictions[0].product_type == "1-CK231"
    assert budget.spent == 2


def test_order_fuel_report_returns_job_ids(tmp_path: Path) -> None:
    fake, budget, tools = make_tools(Mode.CONFIRM, answer=True, tmp=tmp_path)
    out = json.loads(
        asyncio.run(
            tools["order_fuel_report"].call(
                {
                    "date_from": "2026-09-01",
                    "date_to": "2026-09-30",
                    "report_format": "xlsx",
                    "card_ids": ["c1"],
                }
            )
        )
    )
    assert out["status"] == "ordered" and out["job_ids"] == ["job1"]
    assert fake.ordered_reports[0]["start"] == "2026-09-01"
    assert budget.spent == 1


def test_get_report_status_and_access_rules(tmp_path: Path) -> None:
    _, budget, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path)
    status = json.loads(asyncio.run(tools["get_report_status"].call({})))
    assert status["reports"][0]["job_id"] == "job1"
    rules = json.loads(asyncio.run(tools["get_access_rules"].call({"card_id": "c1"})))
    assert rules["region_limits"][0]["access"] == "deny"
    assert rules["restrictions"][0]["product"] == "Топливо"
    assert budget.spent == 2  # регионы и ограничители тарифицируемые, лимиты бесплатны


def test_diagnose_fueling_returns_checks_and_causes(tmp_path: Path) -> None:
    _, budget, tools = make_tools(
        Mode.READ_ONLY, answer=False, tmp=tmp_path, budget_limit=10
    )
    out = json.loads(asyncio.run(tools["diagnose_fueling"].call({"card_id": "c1"})))
    assert out["checks"]["card_status"]["ok"] is True
    # Запрет на регион есть, но без АЗС из чека нельзя сказать, мешает ли он.
    assert out["checks"]["region_limits"]["ok"] is None
    assert out["checks"]["restrictions"]["ok"] is False
    assert out["checks"]["contract_balance"]["ok"] is True
    assert out["checks"]["check_purchase"]["allowed"] is None
    # Счётчик PIN берётся из get_card_detail: в тестовых данных 0 — неверных вводов не было.
    assert out["checks"]["pin_attempts"] == {"ok": True, "counter": 0, "resets_left": 3}
    assert any("регион" in cause for cause in out["likely_causes"])
    assert budget.spent == 4


def _station(fake: NS, region: str = "MOS", allowed: bool = True) -> list[dict[str, object]]:
    purchases: list[dict[str, object]] = []

    async def get_azs_list_v1(*, id: str, **_: object) -> NS:
        prices = [NS(GoodsCode="DT", Price="65.50")]
        return NS(data=NS(result=[NS(regionCode=region, prices=prices)], total_count=1))

    async def get_dictionary(*, name: str, **_: object) -> NS:
        return NS(data=NS(result=[NS(id="DT", code=None, name="ДТ", value=None)], total_count=1))

    async def check_purchase(**kwargs: object) -> NS:
        purchases.append(kwargs)
        return NS(data=allowed)

    fake.dictionaries = NS(get_azs_list_v1=get_azs_list_v1, get_dictionary=get_dictionary)
    fake.final_prices = NS(check_purchase=check_purchase)
    return purchases


def test_diagnose_follows_decline_receipt_and_station(tmp_path: Path) -> None:
    fake, budget, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path, budget_limit=10)
    purchases = _station(fake, region="MOS", allowed=False)
    args = {
        "card_id": "c1", "decline_code": "ER3", "decline_text": "Операция прервана",
        "poi_id": "azs-1", "product": "дт", "liters": 50,
    }
    out = json.loads(asyncio.run(tools["diagnose_fueling"].call(args)))
    receipt = out["receipt"]
    assert (receipt["decline_code"], receipt["decline_text"]) == ("ER3", "Операция прервана")
    # ER3 нет в таблице Приложения № 4: «прочие коды ответа» — к персоналу АЗС.
    assert receipt["decoded"]["in_table"] is False and "персонал" in receipt["decoded"]["terminal_message"]
    assert out["likely_causes"][0].startswith("по коду ответа ER3")
    assert purchases == [
        {"card_id": "c1", "poi_id": "azs-1", "goods": [{"code": "DT", "quantity": 50.0, "price": 65.5}]}
    ]
    assert out["checks"]["check_purchase"]["allowed"] is False
    assert out["checks"]["region_limits"]["ok"] is False  # запрет на регион MOS, АЗС в MOS
    causes = out["likely_causes"]
    # Порядок проверок: статус → check_purchase → лимиты → баланс → регион и ограничитель → PIN.
    assert causes.index(next(c for c in causes if "покупка" in c)) < causes.index(
        next(c for c in causes if "региональный" in c)
    )
    assert budget.spent == 4  # АЗС, справочник и check_purchase бесплатны


def test_diagnose_station_outside_denied_region_is_ok(tmp_path: Path) -> None:
    fake, _, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path, budget_limit=10)
    _station(fake, region="SPB", allowed=True)
    args = {"card_id": "c1", "poi_id": "azs-1", "product": "ДТ", "liters": 30}
    out = json.loads(asyncio.run(tools["diagnose_fueling"].call(args)))
    assert out["checks"]["region_limits"]["ok"] is True
    assert out["checks"]["check_purchase"]["allowed"] is True


def test_diagnose_reports_goods_missing_from_station_price_list(tmp_path: Path) -> None:
    fake, _, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path, budget_limit=10)
    purchases = _station(fake)
    args = {"card_id": "c1", "poi_id": "azs-1", "product": "АИ-100", "liters": 30}
    out = json.loads(asyncio.run(tools["diagnose_fueling"].call(args)))
    assert out["checks"]["check_purchase"]["skipped"] == "товара нет в прайсе АЗС"
    assert out["checks"]["check_purchase"]["station_goods"] == ["ДТ"] and purchases == []


def test_diagnose_finds_station_by_number_from_receipt(tmp_path: Path) -> None:
    fake, _, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path, budget_limit=10)
    purchases = _station(fake, region="SPB")
    searches: list[dict[str, object]] = []
    station = NS(id="poi-137", regionCode="SPB", prices=[NS(GoodsCode="DT", Price="65.50")])

    async def get_azs_list_v1(**params: object) -> NS:
        searches.append(params)
        found = [station] if params.get("q") == "АЗС 137" else []
        return NS(data=NS(result=found, total_count=len(found)))

    fake.dictionaries.get_azs_list_v1 = get_azs_list_v1
    args = {"card_id": "c1", "poi_id": "АЗС 137", "product": "ДТ", "liters": 40}
    out = json.loads(asyncio.run(tools["diagnose_fueling"].call(args)))
    assert searches == [{"id": "АЗС 137"}, {"q": "АЗС 137", "onpage": 2}]
    assert purchases[0]["poi_id"] == "poi-137" and out["checks"]["check_purchase"]["allowed"] is True


def test_diagnose_decodes_known_code_and_flags_mismatch_with_checks(tmp_path: Path) -> None:
    _, _, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path, budget_limit=10)
    out = json.loads(asyncio.run(tools["diagnose_fueling"].call({"card_id": "c1", "decline_code": "51"})))
    decoded = out["receipt"]["decoded"]
    assert decoded["in_table"] is True and decoded["related_check"] == "contract_balance"
    assert out["likely_causes"][0] == "по коду ответа 51 из чека: не хватает денег на счёте договора"
    assert out["next_steps"][0] == "Пополнить счёт договора."
    # Сейчас баланс положительный: агент должен сказать, что состояние могло измениться.
    assert "пополнили" in decoded["note"]


def test_diagnose_pin_in_receipt_text_becomes_a_cause(tmp_path: Path) -> None:
    _, _, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path, budget_limit=10)
    args = {"card_id": "c1", "decline_text": "Неверный PIN"}
    out = json.loads(asyncio.run(tools["diagnose_fueling"].call(args)))
    assert any("PIN" in cause for cause in out["likely_causes"])


def test_download_fuel_report_saves_file(tmp_path: Path) -> None:
    _, budget, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path)
    out = json.loads(
        asyncio.run(
            tools["download_fuel_report"].call({"job_id": "job1", "report_format": "xlsx"})
        )
    )
    assert out["status"] == "downloaded" and out["bytes"] == 4
    assert Path(out["path"]).is_file() and Path(out["path"]).parent == tmp_path / "reports"
    assert budget.spent == 1


def test_remove_tools_confirm_and_call(tmp_path: Path) -> None:
    _, budget, tools = make_tools(Mode.CONFIRM, answer=True, tmp=tmp_path, budget_limit=5)
    for name, args in (
        ("remove_limit", {"limit_id": "l1"}),
        ("remove_region_limit", {"region_limit_id": "r1"}),
        ("remove_restriction", {"restriction_id": "s1"}),
    ):
        out = json.loads(asyncio.run(tools[name].call(args)))
        assert out["status"] == "removed" and out["removed"] is True
    assert budget.spent == 3


def test_remove_tools_are_blocked_in_read_only(tmp_path: Path) -> None:
    _, _, tools = make_tools(Mode.READ_ONLY, answer=True, tmp=tmp_path)
    out = json.loads(asyncio.run(tools["remove_limit"].call({"limit_id": "l1"})))
    assert out["status"] == "not_allowed"


def test_claude_md_matches_system_prompt() -> None:
    from agent.prompt import SYSTEM_PROMPT

    claude_md = (Path(__file__).resolve().parent.parent / "CLAUDE.md").read_text(encoding="utf-8")
    assert claude_md.strip() == SYSTEM_PROMPT.strip()  # инструкция для Claude Code не отстала


def test_tool_schemas_are_compact() -> None:
    _, _, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=Path("/tmp"))
    size = sum(len(json.dumps(t.to_dict(), ensure_ascii=False)) for t in tools.values())
    assert size < 18000  # ~5–6 тыс. токенов на 25 схем: описания должны оставаться короткими


def test_mcp_server_exposes_all_tools() -> None:
    from agent import mcp_server

    names = {t.name for t in asyncio.run(mcp_server.server.list_tools())}
    assert names == {
        "get_account_context", "get_contract_summary", "list_cards", "list_card_groups",
        "get_card_profile", "get_limits", "get_access_rules", "diagnose_fueling",
        "spending_report", "list_transactions", "get_transaction_detail",
        "find_suspicious_fills", "fuel_price_analytics", "get_finance", "get_report_status",
        "download_fuel_report", "lookup_knowledge", "lookup_dictionary", "get_api_usage", "block_cards",
        "set_card_comment", "set_limit", "set_region_limit", "set_restriction",
        "order_fuel_report", "remove_limit", "remove_region_limit",
        "remove_restriction", "get_dashboard",
    }


# ---------- дашборд, провайдеры, цикл OpenAI-совместимых моделей ----------


def fake_full_client(txs: list[NS]) -> NS:
    async def contract() -> NS:
        info = NS(contract_number="D-1", contract_status_name="Действует")
        balance = NS(
            available_amount="100000.50",
            balance="50000",
            last_payment_sum="20000",
            last_payment_date="2026-10-01",
            consumption_for_month="12345.6",
            consumption_for_month_volume="200",
            consumption_for_prev_month_volume="160",
        )
        cards = NS(cards_quantity_all="10", cards_quantity_active="8", card_groups_quantity_all="2")
        return NS(data=NS(payment_term_id=None, payment_scheme_id=None, contractData=info, balanceData=balance, cardsData=cards))

    async def get_info() -> NS:
        return NS(data=NS(client_info=NS(Queries=500), methods=NS(all=7)))

    async def page(**kwargs: object) -> NS:
        offset = kwargs["page_offset"]
        return NS(data=NS(result=txs[offset : offset + 2], total_count=len(txs)))

    return NS(
        contracts=NS(get_contract_data=contract),
        auth=NS(get_info=get_info),
        transactions=NS(get_transactions_v2=page),
    )


def test_dashboard_collects_kpis_and_reports_truncation() -> None:
    from datetime import date

    from agent.dashboard import collect

    # page_limit=100 в реальном API, в фейке страница = 2 операции: хватит проверить обрезку
    txs = [tx(i, 12, "60.00", 10, day=i) for i in range(1, 6)]
    budget = BillingBudget(limit=10)
    board = asyncio.run(collect(fake_full_client(txs), budget, date(2026, 9, 20), max_operation_pages=2))
    assert board.contract["Доступный остаток"] == "100 000,50 ₽"
    assert board.month["Динамика"] == "+25% к прошлому месяцу"
    assert board.month["Операций (покупок)"] == "4"
    assert any("первые 4 операций из 5" in n for n in board.notes)
    assert budget.spent == 3  # договор + 2 страницы операций
    assert board.api["Запросов в пакете"] == 500


def test_dashboard_survives_api_error() -> None:
    from datetime import date

    from apisdkopti24 import APIError

    from agent.dashboard import collect

    client = fake_full_client([])

    async def broken() -> NS:
        raise APIError("boom")

    client.contracts = NS(get_contract_data=broken)
    board = asyncio.run(collect(client, BillingBudget(limit=5), date(2026, 9, 20)))
    assert any("Договор не получен" in n for n in board.notes)
    assert board.api  # остальные блоки собрались


def test_providers_and_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent.config import Settings
    from agent.providers import PROVIDERS, get_provider

    assert {"claude", "deepseek", "qwen", "kimi", "glm", "ollama"} <= set(PROVIDERS)
    with pytest.raises(ValueError):
        get_provider("nope")
    monkeypatch.setenv("AGENT_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    s = Settings.from_env()
    assert (s.model, s.base_url, s.api_key) == ("deepseek-chat", "https://api.deepseek.com", "k")
    monkeypatch.setenv("AGENT_PROVIDER", "ollama")
    assert Settings.from_env().api_key == "ollama"


def test_us_chinese_and_russian_providers_are_available() -> None:
    from agent.providers import PROVIDERS

    assert {"claude", "openai", "deepseek", "qwen", "yandex", "gigachat", "vllm", "custom"} <= set(PROVIDERS)


def test_yandex_model_uri_and_project_come_from_folder(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent.config import Settings

    monkeypatch.setenv("AGENT_PROVIDER", "yandex")
    monkeypatch.setenv("YANDEX_API_KEY", "k")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "folder-id")
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    s = Settings.from_env()
    assert (s.model, s.project) == ("gpt://folder-id/yandexgpt/latest", "folder-id")
    monkeypatch.setenv("AGENT_MODEL", "gpt://other-folder/aliceai-llm")
    assert Settings.from_env().model == "gpt://other-folder/aliceai-llm"


def test_yandex_without_folder_fails_before_any_request(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent.config import Settings
    from agent.core import make_llm

    monkeypatch.setenv("AGENT_PROVIDER", "yandex")
    monkeypatch.setenv("YANDEX_API_KEY", "k")
    monkeypatch.delenv("YANDEX_FOLDER_ID", raising=False)
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="YANDEX_FOLDER_ID"):
        make_llm(Settings.from_env())


def test_vllm_runs_locally_without_key_and_needs_a_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent.config import Settings
    from agent.core import make_llm

    monkeypatch.setenv("AGENT_PROVIDER", "vllm")
    for name in ("AGENT_MODEL", "AGENT_BASE_URL", "AGENT_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="AGENT_MODEL"):
        make_llm(Settings.from_env())
    monkeypatch.setenv("AGENT_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    s = Settings.from_env()
    assert (s.base_url, s.model, s.api_key) == ("http://localhost:8000/v1", "Qwen/Qwen2.5-7B-Instruct", "ollama")
    assert str(make_llm(s).base_url).startswith("http://localhost:8000/v1")


def test_custom_provider_needs_base_url_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent.config import Settings
    from agent.core import make_llm

    monkeypatch.setenv("AGENT_PROVIDER", "custom")
    monkeypatch.setenv("AGENT_API_KEY", "k")
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="AGENT_BASE_URL"):
        make_llm(Settings.from_env())
    monkeypatch.setenv("AGENT_BASE_URL", "https://llm.example.ru/v1")
    monkeypatch.setenv("AGENT_MODEL", "some-model")
    client = make_llm(Settings.from_env())
    assert str(client.base_url).startswith("https://llm.example.ru/v1")


def test_openai_loop_calls_tool_then_answers(tmp_path: Path) -> None:
    from agent.core_types import Usage
    from agent.llm_openai import run_openai_loop

    fake, _, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path)
    steps = []

    def msg(content: str | None, calls: list[NS] | None) -> NS:
        m = NS(content=content, tool_calls=calls)
        m.model_dump = lambda **_: {"role": "assistant", "content": content}  # type: ignore[attr-defined]
        usage = NS(prompt_tokens=100, completion_tokens=20)
        return NS(choices=[NS(message=m)], usage=usage)

    call = NS(id="c1", function=NS(name="get_api_usage", arguments="{}"))
    responses = iter([msg(None, [call]), msg("Готово", None)])

    async def create(**kwargs: object) -> NS:
        steps.append(kwargs["messages"][-1]["role"])  # type: ignore[index]
        return next(responses)

    async def get_info() -> NS:
        return NS(data=NS(client_info=NS(Queries=500), methods=NS(all=3)))

    fake.auth = NS(get_info=get_info)
    tools = {t.name: t for t in build_tools(fake, BillingBudget(5), ApprovalGate(Mode.READ_ONLY, lambda _: None, tmp_path / "a")) }  # type: ignore[arg-type]
    llm = NS(chat=NS(completions=NS(create=create)))
    usage = Usage()
    text, called = asyncio.run(
        run_openai_loop(llm, "m", "sys", [{"role": "user", "content": "q"}], list(tools.values()), 5, usage)
    )
    assert text == "Готово" and called == [("get_api_usage", {})]
    assert steps == ["user", "tool"] and usage.requests == 2 and usage.input_tokens == 200


def test_openai_loop_stops_at_iteration_limit(tmp_path: Path) -> None:
    from agent.core_types import Usage
    from agent.llm_openai import run_openai_loop

    call = NS(id="c", function=NS(name="nope", arguments="{}"))

    async def create(**_: object) -> NS:
        m = NS(content=None, tool_calls=[call])
        m.model_dump = lambda **_: {"role": "assistant", "content": None}  # type: ignore[attr-defined]
        return NS(choices=[NS(message=m)], usage=None)

    llm = NS(chat=NS(completions=NS(create=create)))
    text, _ = asyncio.run(run_openai_loop(llm, "m", "s", [], [], 3, Usage()))
    assert "предел" in text


# ---------- выбор договора ----------


def test_open_session_picks_contract_from_error(tmp_path: Path) -> None:
    from apisdkopti24.errors import ContractSelectionError

    from agent.contract import open_session

    calls: list[str | None] = []

    async def auth_user(*, contract_id: str | None = None) -> NS:
        calls.append(contract_id)
        if contract_id is None:
            raise ContractSelectionError("несколько", available_contracts=(("id1", "№1"), ("id2", "№2")))
        return NS()

    async def choose(contracts: list[tuple[str, str]]) -> str:
        assert [n for _, n in contracts] == ["№1", "№2"]
        return "id2"

    client = NS(auth=NS(auth_user=auth_user))
    chosen = asyncio.run(open_session(client, preferred=None, production=False, choose=choose))  # type: ignore[arg-type]
    assert chosen == "id2" and calls == [None, "id2"]


def test_open_session_requires_explicit_contract_on_production() -> None:
    from apisdkopti24.errors import ContractSelectionError

    from agent.contract import open_session

    with pytest.raises(ContractSelectionError):
        asyncio.run(open_session(NS(), preferred=None, production=True, choose=None))  # type: ignore[arg-type]


def test_open_session_without_terminal_lists_contracts() -> None:
    from apisdkopti24.errors import ContractSelectionError

    from agent.contract import open_session

    async def auth_user(**_: object) -> NS:
        raise ContractSelectionError("x", available_contracts=(("id1", "№1"),))

    with pytest.raises(ContractSelectionError, match="AGENT_CONTRACT_ID"):
        asyncio.run(open_session(NS(auth=NS(auth_user=auth_user)), preferred=None, production=False, choose=None))  # type: ignore[arg-type]


def test_preferred_contract_from_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agent.contract import preferred_contract_id

    for var in ("AGENT_CONTRACT_ID", "API_CONTRACT_ID"):
        monkeypatch.delenv(var, raising=False)
    env = tmp_path / ".env"
    env.write_text("API_KEY=x\nAPI_CONTRACT_ID='abc-123'\n", encoding="utf-8")
    assert preferred_contract_id(env) == "abc-123"
    monkeypatch.setenv("AGENT_CONTRACT_ID", "zzz")
    assert preferred_contract_id(env) == "zzz"


def test_405_on_transactions_is_unavailable_and_refunds_budget():
    import asyncio
    from types import SimpleNamespace

    from apisdkopti24 import APIError

    from agent.safety import BillingBudget
    from agent.tools import TransactionsUnavailableError, load_transactions

    async def boom(**_):
        raise APIError(405, "x", http_status_code=405)

    client = SimpleNamespace(transactions=SimpleNamespace(get_transactions_v2=boom))
    budget = BillingBudget(limit=5)
    try:
        asyncio.run(load_transactions(client, budget, "2026-10-01", "2026-10-17"))
    except TransactionsUnavailableError:
        pass
    else:
        raise AssertionError("ожидалась TransactionsUnavailableError")
    assert budget.spent == 0


def test_web_dashboard_renders_and_escapes() -> None:
    from datetime import date

    from agent.dashboard import Dashboard
    from agent.web import render_page

    board = Dashboard(today=date(2026, 10, 17))
    board.contract = {"Договор": "<b>X</b>", "Статус": "Действует", "Доступный остаток": "1 ₽"}
    board.month = {
        "Топливо": [{"product": "ДТ", "liters": "100", "sum": "5 000,00 ₽", "avg_price_per_liter": None}],
        "Топ карт": [{"card_id": "700", "operations": 3, "sum": "5 000,00 ₽"}],
    }
    board.control = {"Проверено операций": 3, "Подозрительных": 1,
                     "Примеры": [{"date": "2026-10-01", "sum": "1 ₽", "reasons": ["ночь"]}]}
    board.charts = {
        "daily": [{"date": "2026-10-01", "sum": 5000.0, "liters": 100.0}],
        "heatmap": [[1] + [0] * 23] + [[0] * 24] * 6,
        "recent": [{"time": "01.10 03:00", "card": "700", "product": "ДТ", "liters": "100", "price": "50 ₽", "sum": "5 000 ₽"}],
    }
    html = render_page(board, production=False, fetched_at=0.0)
    assert "&lt;b&gt;X&lt;/b&gt;" in html and "<b>X</b>" not in html
    for title in ("Последние операции", "Расход по дням", "Контроль заправок", "Когда заправляются", "Чат с агентом"):
        assert title in html


def test_web_server_pages_chat_and_origin_checks(monkeypatch) -> None:
    import asyncio
    import json as _json
    from datetime import date

    import agent.web as web
    from agent.dashboard import Dashboard

    async def fake_collect(*_a, **_k):
        return Dashboard(today=date(2026, 10, 17))

    monkeypatch.setattr(web, "collect", fake_collect)

    class StubAgent:
        async def ask(self, q: str) -> str:
            return "ответ: " + q

    async def run():
        app = web.Server(client=None, production=False, settings=object(), port=0)  # type: ignore[arg-type]
        app._agent = StubAgent()  # type: ignore[assignment]
        server = await asyncio.start_server(app.handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        app.port = port

        async def send(raw: str) -> bytes:
            r, w = await asyncio.open_connection("127.0.0.1", port)
            w.write(raw.encode())
            await w.drain()
            data = await r.read()
            w.close()
            return data

        def post(host: str, origin: str | None = None) -> str:
            body = _json.dumps({"question": "привет"}, ensure_ascii=False).encode()
            extra = f"Origin: {origin}\r\n" if origin else ""
            return (f"POST /chat HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\n{extra}"
                    f"Content-Length: {len(body)}\r\n\r\n") + body.decode()

        async with server:
            return (
                await send(f"GET / HTTP/1.1\r\nHost: localhost:{port}\r\n\r\n"),
                await send("GET /etc/passwd HTTP/1.1\r\nHost: x\r\n\r\n"),
                await send(post(f"localhost:{port}")),
                await send(post(f"localhost:{port}", "http://evil.example")),
                await send(post("evil.example")),
            )

    page, missing, chat_ok, bad_origin, bad_host = asyncio.run(run())
    assert b"200 OK" in page and "Пульт топливных карт".encode() in page
    assert b"404" in missing
    assert "ответ: привет".encode() in chat_ok or b"\\u043e" in chat_ok
    assert b"404" in bad_origin and b"404" in bad_host


def test_openai_tool_args_tolerate_broken_json() -> None:
    from agent.llm_openai import parse_tool_args

    assert parse_tool_args('{"card_id": "c1"}') == {"card_id": "c1"}
    assert parse_tool_args("{oops") == {} and parse_tool_args("[1]") == {} and parse_tool_args(None) == {}


def _usage(fake: NS, package: object, used: int) -> None:
    async def get_info(**_: object) -> NS:
        return NS(
            data=NS(
                from_=datetime(2026, 10, 1),
                to=datetime(2026, 11, 1),
                client_info=NS(Queries=package),
                methods=NS(all=used),
            )
        )

    fake.auth.get_info = get_info


def test_api_usage_shows_left_in_monthly_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPTI24_MONTHLY_QUOTA", raising=False)
    fake, budget, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path, budget_limit=10)
    _usage(fake, None, 120)
    out = json.loads(asyncio.run(tools["get_api_usage"].call({})))
    assert out["monthly_quota"] == 500 and out["quota_source"].startswith("по умолчанию")
    assert out["quota_enforced"] is False
    assert out["left_in_period"] == 380 and "warning" not in out and budget.limit == 10


def test_nearly_exhausted_package_warns_and_caps_session_budget(tmp_path: Path) -> None:
    fake, budget, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path, budget_limit=10)
    _usage(fake, 500, 497)
    out = json.loads(asyncio.run(tools["get_api_usage"].call({})))
    assert out["left_in_period"] == 3 and "заблокируется" in out["warning"]
    assert budget.limit == 3  # агент сам не израсходует пакет до блокировки


def test_exhausted_package_stops_paid_calls(tmp_path: Path) -> None:
    fake, budget, tools = make_tools(
        Mode.READ_ONLY, answer=False, tmp=tmp_path, txs=[tx(1, 12, "60.00", 10)], budget_limit=10
    )
    _usage(fake, 500, 500)
    asyncio.run(tools["get_api_usage"].call({}))
    out = json.loads(
        asyncio.run(tools["spending_report"].call({"date_from": "2026-09-01", "date_to": "2026-09-30"}))
    )
    assert out["error"] == "BudgetExceededError" and budget.spent == 0


def test_package_size_from_api_wins_over_setting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPTI24_MONTHLY_QUOTA", "500")
    fake, _, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path)
    _usage(fake, 2000, 600)
    out = json.loads(asyncio.run(tools["get_api_usage"].call({})))
    assert (out["monthly_quota"], out["quota_source"], out["left_in_period"]) == (2000, "API", 1400)


def test_default_quota_is_advisory_and_never_closes_paid_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # На DEMO общий счётчик больше 500: остаток 0 не должен обнулять бюджет сессии.
    monkeypatch.delenv("OPTI24_MONTHLY_QUOTA", raising=False)
    fake, budget, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path, budget_limit=10)
    _usage(fake, None, 900)
    out = json.loads(asyncio.run(tools["get_api_usage"].call({})))
    assert out["left_in_period"] == 0 and out["quota_enforced"] is False and budget.limit == 10


def test_explicit_quota_setting_does_cap_the_session_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPTI24_MONTHLY_QUOTA", "500")
    fake, budget, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path, budget_limit=10)
    _usage(fake, None, 497)
    out = json.loads(asyncio.run(tools["get_api_usage"].call({})))
    assert out["quota_source"] == "настройка OPTI24_MONTHLY_QUOTA" and out["quota_enforced"] is True
    assert budget.limit == 3


def test_non_numeric_limit_gives_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent.config import Settings

    monkeypatch.setenv("AGENT_MAX_BILLABLE_CALLS", "выведи дашборд")
    with pytest.raises(ValueError, match="AGENT_MAX_BILLABLE_CALLS должна быть целым числом"):
        Settings.from_env()
    monkeypatch.setenv("AGENT_MAX_BILLABLE_CALLS", "6")
    assert Settings.from_env().max_billable_calls == 6


def test_account_context_separates_agent_mode_from_account_rights(tmp_path: Path) -> None:
    for mode, expected in ((Mode.READ_ONLY, False), (Mode.CONFIRM, True)):
        _, _, tools = make_tools(mode, answer=False, tmp=tmp_path)
        out = json.loads(asyncio.run(tools["get_account_context"].call({})))
        # У аккаунта есть право записи, но агент в read_only изменять не может.
        assert out["read_only"] is False and out["agent_mode"] == mode.value
        assert out["changes_enabled"] is expected


def test_contract_summary_gives_previous_month_volume_and_fee_rate(tmp_path: Path) -> None:
    _, _, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path)
    out = json.loads(asyncio.run(tools["get_contract_summary"].call({})))
    # 120 000 л за прошлый месяц — ступень 100 001–150 000 л таблицы 3: 2,5 %.
    assert out["prev_month_liters"] == "120000" and out["service_fee_percent_typical"] == "2.5"
    assert "Приложение № 1" in out["service_fee_terms"]
