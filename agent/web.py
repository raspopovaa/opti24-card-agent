"""Веб-дашборд с чатом на своём компьютере: ``python -m agent.web`` → http://localhost:8765

Данные те же, что в терминальном дашборде (``dashboard.collect``). Сервер слушает
только 127.0.0.1, внешних библиотек и внешних адресов в странице нет. Кнопка
«Обновить» делает новые платные запросы (до 4), поэтому данные кешируются на минуту.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import re
import time
from datetime import date
from html import escape

from apisdkopti24 import APIClient, ConnectionSettings, EnvironmentCredentialsProvider

from .config import Mode, Settings
from .core import FuelCardAgent
from .contract import is_production, open_session, preferred_contract_id
from .dashboard import Dashboard, collect
from .safety import BillingBudget

CACHE_SECONDS = 60
PAGE_BUDGET = 10  # платных запросов на весь запуск сервера

CSS = """
:root{--bg:#f4f6f4;--card:#fff;--ink:#1b2420;--dim:#66736c;--accent:#153f34;--bar:#2f7d63;--warn:#b3261e;--line:#dfe4e0;--heat:21,63,52}
@media (prefers-color-scheme: dark){:root{--bg:#101614;--card:#19211e;--ink:#e8eeea;--dim:#9aa8a0;--accent:#7fd1b0;--bar:#4fae8c;--warn:#ff8a80;--line:#2a3530;--heat:127,209,176}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}
header{background:var(--accent);color:#fff;padding:14px 20px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
header h1{font-size:20px;margin:0}.badge{background:#ffffff26;border-radius:6px;padding:2px 8px;font-size:12px}.age{margin-left:auto;font-size:13px;opacity:.8}
main{max-width:1100px;margin:0 auto;padding:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin-bottom:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;overflow-x:auto}
.card h2{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--dim);margin:0 0 8px}
.big{font-size:26px;font-weight:600;font-variant-numeric:tabular-nums}.hint{color:var(--dim);font-size:13px}
dl{margin:0;display:grid;grid-template-columns:auto 1fr;gap:4px 12px}dt{color:var(--dim)}dd{margin:0;text-align:right;font-variant-numeric:tabular-nums}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}td,th{padding:5px 6px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--dim);font-weight:500}td:first-child,th:first-child{text-align:left}
.bar{height:8px;background:var(--bar);border-radius:4px;min-width:2px}.wide{grid-column:1/-1}
.warn{color:var(--warn)}.note{color:var(--dim)}ul{margin:6px 0 0;padding-left:20px}
.heat td{padding:0;border:0;text-align:center;font-size:11px;height:22px;min-width:18px}.heat td:first-child{text-align:left;padding-right:8px;color:var(--dim)}
.stack{display:flex;height:18px;border-radius:5px;overflow:hidden;margin:6px 0}.legend span{margin-right:12px;white-space:nowrap}
a.btn,button{font:inherit;padding:7px 14px;border:1px solid var(--accent);border-radius:8px;color:var(--accent);background:transparent;text-decoration:none;cursor:pointer}
button.primary{background:var(--accent);color:#fff}button:disabled{opacity:.5}
#log{max-height:340px;overflow-y:auto;display:flex;flex-direction:column;gap:8px;margin-bottom:10px}
.msg{padding:8px 12px;border-radius:10px;white-space:pre-wrap;max-width:90%}.me{align-self:flex-end;background:var(--accent);color:#fff}.bot{background:var(--bg);border:1px solid var(--line)}
form{display:flex;gap:8px}input[type=text]{flex:1;font:inherit;padding:8px 10px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--ink)}
"""

PALETTE = ["#2f7d63", "#d9a441", "#4b7bb5", "#b3563b", "#8a6bb0", "#7a8a84"]

CHAT_JS = """
const log=document.getElementById('log'),form=document.getElementById('chat'),box=document.getElementById('q'),go=document.getElementById('go');
function add(t,c){const d=document.createElement('div');d.className='msg '+c;d.textContent=t;log.appendChild(d);log.scrollTop=log.scrollHeight;return d}
form.addEventListener('submit',async e=>{e.preventDefault();const q=box.value.trim();if(!q)return;box.value='';add(q,'me');go.disabled=true;const w=add('Думаю…','bot');
try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});const j=await r.json();w.textContent=j.answer||j.error||'Пустой ответ'}
catch(err){w.textContent='Сервер недоступен: '+err}finally{go.disabled=false;box.focus()}});
"""


def _number(text: object) -> float:
    cleaned = re.sub(r"[^\d,.-]", "", str(text)).replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def _kv(title: str, data: dict[str, object]) -> str:
    rows = "".join(
        f"<dt>{escape(k)}</dt><dd>{escape(str(v))}</dd>"
        for k, v in data.items()
        if not isinstance(v, (list, dict))
    )
    return f'<section class="card"><h2>{escape(title)}</h2><dl>{rows}</dl></section>' if rows else ""


def _kpi(title: str, value: object, hint: str = "") -> str:
    return (
        f'<section class="card"><h2>{escape(title)}</h2><div class="big">{escape(str(value))}</div>'
        f'<div class="hint">{escape(hint)}</div></section>'
    )


def _kpis(board: Dashboard) -> str:
    c, m = board.contract, board.month
    if not c:
        return ""
    payment, _, paid_on = str(c.get("Последний платёж", "—")).partition(" от ")
    return (
        '<div class="grid">'
        + _kpi("Баланс договора", c.get("Доступный остаток", "—"), f"Собственные средства: {c.get('Собственные средства', '—')}")
        + _kpi("Последний платёж", payment, f"от {paid_on}" if paid_on else "")
        + _kpi("Израсходовано за месяц", m.get("Расход за месяц", "—"), f"{m.get('Литров за месяц', '—')} л · {m.get('Динамика', '')}")
        + "</div>"
    )


def _recent(rows: object) -> str:
    if not isinstance(rows, list) or not rows:
        return ""
    body = "".join(
        f"<tr><td>{escape(r['time'])}</td><td>{escape(str(r['card']))}</td><td>{escape(r['product'])}</td>"
        f"<td>{escape(str(r['liters']))}</td><td>{escape(r['price'])}</td><td>{escape(r['sum'])}</td></tr>"
        for r in rows
    )
    return (
        '<section class="card wide"><h2>Последние операции</h2><table><tr><th>Время</th><th>Карта</th>'
        f"<th>Товар</th><th>Литры</th><th>Цена</th><th>Сумма</th></tr>{body}</table></section>"
    )


def _daily(rows: object) -> str:
    if not isinstance(rows, list) or not rows:
        return ""
    top = max(r["sum"] for r in rows) or 1.0
    width, height, gap = 640, 150, 3
    step = width / len(rows)
    bars = "".join(
        f'<rect x="{i * step + gap / 2:.1f}" y="{height - r["sum"] / top * (height - 6):.1f}" '
        f'width="{max(step - gap, 1):.1f}" height="{r["sum"] / top * (height - 6):.1f}" rx="2" fill="var(--bar)">'
        f'<title>{escape(r["date"])}: {r["sum"]:,.0f} ₽, {r["liters"]:,.0f} л</title></rect>'.replace(",", " ")
        for i, r in enumerate(rows)
    )
    return (
        '<section class="card wide"><h2>Расход по дням, ₽</h2>'
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Расход по дням">{bars}</svg>'
        f'<div class="hint">{escape(rows[0]["date"])} — {escape(rows[-1]["date"])} · максимум за день {top:,.0f} ₽</div></section>'.replace(",", " ")
    )


def _structure(rows: object) -> str:
    if not isinstance(rows, list) or not rows:
        return ""
    total = sum(_number(r["sum"]) for r in rows) or 1.0
    parts, legend = [], []
    for i, r in enumerate(rows):
        share = _number(r["sum"]) / total * 100
        color = PALETTE[i % len(PALETTE)]
        parts.append(f'<div style="width:{share:.1f}%;background:{color}" title="{escape(r["product"])}"></div>')
        legend.append(
            f'<span><i style="display:inline-block;width:10px;height:10px;border-radius:2px;background:{color}"></i> '
            f"{escape(r['product'])} {share:.0f}% · {escape(str(r['liters']))} л · {escape(str(r['avg_price_per_liter'] or '—'))}/л</span>"
        )
    return (
        '<section class="card wide"><h2>Структура по продуктам (доля в ₽)</h2>'
        f'<div class="stack">{"".join(parts)}</div><div class="legend">{"".join(legend)}</div></section>'
    )


def _heatmap(grid: object) -> str:
    if not isinstance(grid, list) or not any(any(row) for row in grid):
        return ""
    peak = max(max(row) for row in grid) or 1
    days = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    head = "<tr><td></td>" + "".join(f"<td>{h}</td>" for h in range(24)) + "</tr>"
    rows = "".join(
        f"<tr><td>{days[d]}</td>"
        + "".join(
            f'<td style="background:rgba(var(--heat),{v / peak:.2f})" title="{days[d]} {h}:00 — {v}"></td>'
            for h, v in enumerate(row)
        )
        + "</tr>"
        for d, row in enumerate(grid)
    )
    return (
        f'<section class="card wide"><h2>Когда заправляются (день недели × час, число заправок)</h2>'
        f'<table class="heat">{head}{rows}</table></section>'
    )


def _top_cards(rows: object) -> str:
    if not isinstance(rows, list) or not rows:
        return ""
    biggest = max(_number(r["sum"]) for r in rows) or 1.0
    body = "".join(
        f"<tr><td>{escape(str(r['card_id']))}</td><td>{r['operations']}</td><td>{escape(str(r['sum']))}</td>"
        f'<td style="width:30%"><div class="bar" style="width:{_number(r["sum"]) / biggest * 100:.0f}%"></div></td></tr>'
        for r in rows
    )
    return (
        '<section class="card wide"><h2>Топ-5 карт по расходу</h2><table><tr><th>Карта</th>'
        f"<th>Операций</th><th>Сумма</th><th></th></tr>{body}</table></section>"
    )


def _control(control: dict[str, object]) -> str:
    if not control:
        return ""
    suspicious = control["Подозрительных"]
    cls = ' class="warn"' if suspicious else ""
    items = "".join(
        f"<li>{escape(str(i['date']))} · {escape(str(i['sum']))} · {escape('; '.join(i['reasons']))}</li>"
        for i in control["Примеры"]  # type: ignore[union-attr]
    )
    return (
        '<section class="card wide"><h2>Контроль заправок</h2>'
        f"<p>Проверено операций: {control['Проверено операций']}, подозрительных: <b{cls}>{suspicious}</b></p>"
        f"<ul>{items}</ul></section>"
    )


CHAT_HTML = (
    '<section class="card wide"><h2>Чат с агентом (только чтение)</h2><div id="log"></div>'
    '<form id="chat"><input id="q" type="text" autocomplete="off" '
    'placeholder="Например: сколько потратили на топливо в этом месяце?"><button id="go" class="primary">Спросить</button></form>'
    '<div class="hint">Агент отвечает по данным договора. Блокировка карт здесь отключена; ответы на вопросы с операциями могут тратить платные запросы.</div></section>'
)


def render_page(board: Dashboard, production: bool, fetched_at: float) -> str:
    notes = "".join(f'<p class="note">⚠ {escape(n)}</p>' for n in board.notes)
    badge = "РАБОЧИЙ ДОГОВОР" if production else "DEMO"
    age = int(time.time() - fetched_at)
    ch = board.charts
    return (
        '<!doctype html><html lang="ru"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Пульт топливных карт</title><style>{CSS}</style></head><body>"
        f'<header><h1>Пульт топливных карт</h1><span class="badge">{badge}</span>'
        f'<span class="badge">{escape(str(board.contract.get("Договор", "")))}</span>'
        f'<span class="age">{board.today:%d.%m.%Y} · данные получены {age} с назад · '
        '<a class="btn" style="color:#fff;border-color:#fff7;padding:3px 10px" href="/?refresh=1">Обновить</a></span></header><main>'
        f"{_kpis(board)}"
        f'<div class="grid">{_kv("Карты", board.cards)}{_kv("Запросы API", board.api)}</div>'
        f'<div class="grid">{_recent(ch.get("recent"))}{_daily(ch.get("daily"))}{_structure(board.month.get("Топливо"))}'
        f'{_top_cards(board.month.get("Топ карт"))}{_control(board.control)}{_heatmap(ch.get("heatmap"))}</div>'
        f"{notes}{CHAT_HTML}</main><script>{CHAT_JS}</script></body></html>"
    )


async def _no(_: str) -> bool:
    return False  # в вебе подтвердить изменение нельзя: чат только для чтения


class Server:
    def __init__(self, client: APIClient, production: bool, settings: Settings | None = None, port: int = 0) -> None:
        self.client = client
        self.production = production
        self.settings = settings
        self.port = port
        self.budget = BillingBudget(limit=PAGE_BUDGET)
        self.board: Dashboard | None = None
        self.fetched_at = 0.0
        self.lock = asyncio.Lock()
        self.chat_lock = asyncio.Lock()
        self._agent: FuelCardAgent | None = None

    async def page(self, refresh: bool) -> str:
        async with self.lock:
            stale = time.time() - self.fetched_at > CACHE_SECONDS
            if self.board is None or (refresh and stale):
                self.board = await collect(self.client, self.budget, date.today())
                self.fetched_at = time.time()
            return render_page(self.board, self.production, self.fetched_at)

    async def chat(self, question: str) -> dict[str, str]:
        if self.settings is None:
            return {"error": "Чат не настроен."}
        async with self.chat_lock:
            if self._agent is None:
                read_only = dataclasses.replace(self.settings, mode=Mode.READ_ONLY)
                self._agent = FuelCardAgent(settings=read_only, opti24=self.client, ask_human=_no)
            try:
                return {"answer": await self._agent.ask(question[:1000])}
            except Exception as exc:  # noqa: BLE001 — в страницу уходит короткая причина, а не трейсбек
                return {"error": f"Агент не ответил: {type(exc).__name__}: {str(exc)[:200]}"}

    def _local_origin(self, headers: dict[str, str]) -> bool:
        """Защита от чужих сайтов: принимаем только запросы со своей страницы."""
        allowed_hosts = {f"localhost:{self.port}", f"127.0.0.1:{self.port}"}
        if headers.get("host", "") not in allowed_hosts:
            return False
        origin = headers.get("origin")
        return origin is None or origin in {f"http://{h}" for h in allowed_hosts}

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request = (await reader.readline()).decode("latin-1")
            headers: dict[str, str] = {}
            while (line := (await reader.readline()).decode("latin-1").strip()):
                name, _, value = line.partition(":")
                headers[name.strip().lower()] = value.strip()
            method, _, rest = request.partition(" ")
            path = rest.split(" ")[0]
            content_type = "text/html; charset=utf-8"
            if method == "GET" and path.split("?")[0] == "/" and (self.port == 0 or self._local_origin(headers)):
                status, body = "200 OK", await self.page("refresh=1" in path)
            elif method == "POST" and path == "/chat" and self._local_origin(headers):
                content_type = "application/json; charset=utf-8"
                try:
                    length = int(headers.get("content-length", "0"))
                    if length > 4096 or not headers.get("content-type", "").startswith("application/json"):
                        raise ValueError
                    question = str(json.loads(await reader.readexactly(length))["question"]).strip()
                    reply = await self.chat(question) if question else {"error": "Пустой вопрос."}
                except (ValueError, KeyError, TypeError, asyncio.IncompleteReadError):
                    reply = {"error": "Некорректный запрос."}
                status, body = "200 OK", json.dumps(reply, ensure_ascii=False)
            else:
                status, body = "404 Not Found", "Не найдено"
            data = body.encode("utf-8")
            writer.write(
                f"HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\n"
                f"Content-Length: {len(data)}\r\nCache-Control: no-store\r\n"
                "Connection: close\r\n\r\n".encode("ascii") + data
            )
            await writer.drain()
        finally:
            writer.close()


async def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m agent.web")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--contract-id")
    args = parser.parse_args()

    settings = Settings.from_env()
    env_file = settings.opti24_env_file
    conn = ConnectionSettings.from_env(env_file=env_file)
    production = is_production(conn)
    async with APIClient(
        settings=conn, credentials_provider=EnvironmentCredentialsProvider.from_env(env_file=env_file)
    ) as client:
        await open_session(
            client,
            preferred=args.contract_id or preferred_contract_id(env_file),
            production=production,
            choose=None,
        )
        app = Server(client, production, settings, args.port)
        server = await asyncio.start_server(app.handle, "127.0.0.1", args.port)
        print(f"Дашборд: http://localhost:{args.port}  (Ctrl+C — остановить)")
        if os.environ.get("AGENT_OPEN_BROWSER", "1") == "1":
            import webbrowser

            webbrowser.open(f"http://localhost:{args.port}")
        try:
            async with server:
                await server.serve_forever()
        finally:
            await client.auth.logoff()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
