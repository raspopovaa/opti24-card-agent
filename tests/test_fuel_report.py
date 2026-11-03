"""Разбор транзакционного отчёта: только выдуманные строки, без данных клиентов."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from agent.config import Mode
from agent.fuel_report import ReportFormatError, read_rows, summarize_report
from tests.test_agent import make_tools

HEADER = (
    "Номер карты,Время транзакции,Тип транзакции,АЗС,Принадлежность АЗС,Товар,Количество,"
    "Единица измерения,Сумма без скидки,Сумма скидки,Сумма сервисного сбора,Сумма к оплате,"
    "Регион АЗС,Адрес АЗС,Комментарий клиента,Группа карт"
)
ROWS = (
    # Обычная покупка: 100 л по 60 ₽, скидка 100 ₽, сбор 1 %.
    '7826010000000001,2026-09-01 10:00:00,Покупка,001,Газпромнефть,ДТ,100,Литры,"6000","100","-59","5900",'
    'Регион А,"г. Тест, ул. Первая, 1",Машина 1,Группа 1',
    # Наценка: скидка отрицательная, клиент заплатил больше стелы.
    '7826010000000002,2026-09-02 02:30:00,Покупка,002,Партнер,ДТ,700,Литры,"42000","-700","-427","42700",'
    'Не определено,"г. Тест, ул. Вторая, 2",Машина 2,Группа 2',
    # Возврат части первой заправки.
    '7826010000000001,2026-09-03 11:00:00,Возврат,001,Газпромнефть,ДТ,-10,Литры,"-600","-10","5,9","-590",'
    'Регион А,"г. Тест, ул. Первая, 1",Машина 1,Группа 1',
    # Нетопливный товар.
    '7826010000000001,2026-09-03 11:05:00,Покупка,001,Газпромнефть,Салфетки,1,Штуки,"100","0","-1","100",'
    'Регион А,"г. Тест, ул. Первая, 1",Машина 1,Группа 1',
)


def report(*rows: str) -> bytes:
    return ("﻿" + "\n".join((HEADER, *rows)) + "\n").encode("utf-8")


def test_fee_and_price_with_fee_come_from_report() -> None:
    out = summarize_report(read_rows(report(*ROWS)))
    total = out["total"]
    assert isinstance(total, dict)
    # Литры: 100 + 700 − 10; штуки в литры не входят.
    assert total["liters"] == "790"
    assert total["paid"] == "48 110,00 ₽"  # 5900 + 42700 − 590 + 100
    assert total["service_fee"] == "481,10 ₽"  # 59 + 427 − 5,9 + 1
    assert total["paid_with_fee"] == "48 591,10 ₽"
    assert total["service_fee_percent"] == "1.00"
    assert total["price_with_fee_per_liter"] == "61,51 ₽"  # 48 591,10 / 790
    assert out["period"] == "2026-09-01…2026-09-03"


def test_markup_returns_and_signals() -> None:
    out = summarize_report(read_rows(report(*ROWS)))
    markup = out["markup_over_station_price"]
    assert isinstance(markup, dict)
    assert markup["operations"] == 1 and markup["overpaid"] == "700,00 ₽"
    assert markup["top_stations"][0]["owner_type"] == "Партнер"
    returns = out["returns_and_storno"]
    assert isinstance(returns, dict) and returns["operations"] == 1 and returns["paid"] == "-590,00 ₽"
    signals = out["signals"]
    assert isinstance(signals, dict)
    assert signals["large_fills"]["count"] == 1  # 700 л больше порога 650
    assert signals["night_fills"]["count"] == 1
    assert signals["non_fuel_goods"]["count"] == 1
    assert "Не определено" in str(out["hint"])


def test_cp1251_and_missing_columns() -> None:
    data = "\n".join((HEADER, ROWS[0])).encode("cp1251")
    assert len(read_rows(data)) == 1
    with pytest.raises(ReportFormatError, match="Сумма сервисного сбора"):
        read_rows("Номер карты,Время транзакции\n1,2026-09-01 10:00:00\n".encode("utf-8"))


def test_download_csv_returns_summary(tmp_path: Path) -> None:
    fake, budget, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path)

    async def download(*, job_id: str, destination: Path) -> Path:
        assert job_id == "job1"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(report(*ROWS))
        return destination

    fake.reports.download_report_file_v1_to = download
    out = json.loads(asyncio.run(tools["download_fuel_report"].call({"job_id": "job1", "report_format": "csv"})))
    assert out["status"] == "downloaded" and budget.spent == 1
    assert out["summary"]["total"]["service_fee"] == "481,10 ₽"


def test_download_xlsx_says_summary_needs_csv(tmp_path: Path) -> None:
    _, _, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path)
    out = json.loads(asyncio.run(tools["download_fuel_report"].call({"job_id": "job1", "report_format": "xlsx"})))
    assert "csv" in out["summary_unavailable"] and "summary" not in out


def test_download_broken_csv_keeps_file(tmp_path: Path) -> None:
    _, _, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path)
    out = json.loads(asyncio.run(tools["download_fuel_report"].call({"job_id": "job1", "report_format": "csv"})))
    assert out["status"] == "downloaded" and "не разобран" in out["summary_unavailable"]
