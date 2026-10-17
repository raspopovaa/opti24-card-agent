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
        poi_id="p",
        type="P",
        product_name="АИ-95",
        qty=qty,
        price=price,
        sum=str(Decimal(price) * Decimal(str(qty))),
        is_storno=False,
    )


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
        page = NS(data=NS(result=transactions, total_count=len(transactions)))

        async def get_transactions_v2(**_: object) -> NS:
            return page

        async def block_card(*, card_ids: list[str], block: bool) -> None:
            self.blocked += card_ids

        self.transactions = NS(get_transactions_v2=get_transactions_v2)
        self.cards = NS(block_card=block_card)


def make_tools(mode: Mode, answer: bool, tmp: Path, txs: list[NS] | None = None):
    fake = FakeOpti24(txs or [])
    budget = BillingBudget(limit=3)

    async def ask(_: str) -> bool:
        return answer

    gate = ApprovalGate(mode=mode, ask=ask, audit_log=tmp / "audit.jsonl")
    tools = {t.name: t for t in build_tools(fake, budget, gate)}  # type: ignore[arg-type]
    return fake, budget, tools


def test_block_requires_confirmation(tmp_path: Path) -> None:
    fake, _, tools = make_tools(Mode.CONFIRM, answer=False, tmp=tmp_path)
    out = json.loads(asyncio.run(tools["block_cards"].call({"card_ids": ["a"], "reason": "тест"})))
    assert out["status"] == "declined" and fake.blocked == []


def test_block_works_after_yes_and_is_audited(tmp_path: Path) -> None:
    fake, _, tools = make_tools(Mode.CONFIRM, answer=True, tmp=tmp_path)
    asyncio.run(tools["block_cards"].call({"card_ids": ["a", "b"], "reason": "тест"}))
    assert fake.blocked == ["a", "b"]
    log = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    assert '"approved": true' in log and "тест" not in log  # причина и ID в журнал не попадают


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
    assert out == {"status": "blocked", "cards": 1} and fake.blocked == ["c7"]
    assert "c7" in asked[0]
    assert "c7" not in (tmp_path / "audit.jsonl").read_text(encoding="utf-8")


def test_limits_need_exactly_one_target(tmp_path: Path) -> None:
    _, _, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path)
    for args in ({}, {"card_id": "c1", "group_id": "g1"}):
        out = json.loads(asyncio.run(tools["get_limits"].call(args)))
        assert out["error"] == "bad_request"


def test_claude_md_matches_system_prompt() -> None:
    from agent.prompt import SYSTEM_PROMPT

    claude_md = (Path(__file__).resolve().parent.parent / "CLAUDE.md").read_text(encoding="utf-8")
    assert claude_md.strip() == SYSTEM_PROMPT.strip()  # инструкция для Claude Code не отстала


def test_tool_schemas_are_compact() -> None:
    _, _, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=Path("/tmp"))
    size = sum(len(json.dumps(t.to_dict(), ensure_ascii=False)) for t in tools.values())
    assert size < 6000  # около 2 тыс. токенов на все схемы: короткие инструменты экономят вход


def test_mcp_server_exposes_all_tools() -> None:
    from agent import mcp_server

    names = {t.name for t in asyncio.run(mcp_server.server.list_tools())}
    assert names == {
        "get_contract_summary", "list_cards", "list_card_groups", "get_limits",
        "spending_report", "find_suspicious_fills", "get_api_usage", "block_cards", "get_dashboard",
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
        return NS(data=NS(contractData=info, balanceData=balance, cardsData=cards))

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

    from agent.dashboard import collect
    from apisdkopti24 import APIError

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
    assert text == "Готово" and called == ["get_api_usage"]
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
    from agent.contract import open_session
    from apisdkopti24.errors import ContractSelectionError

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
    from agent.contract import open_session
    from apisdkopti24.errors import ContractSelectionError

    with pytest.raises(ContractSelectionError):
        asyncio.run(open_session(NS(), preferred=None, production=True, choose=None))  # type: ignore[arg-type]


def test_open_session_without_terminal_lists_contracts() -> None:
    from agent.contract import open_session
    from apisdkopti24.errors import ContractSelectionError

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
