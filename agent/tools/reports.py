"""Инструменты агента: Отчёты: статус, заказ и скачивание."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from anthropic import beta_async_tool
from apisdkopti24 import APIError
from apisdkopti24.errors import ContractSelectionError, RequestValidationError

from ..safety import BudgetExceededError
from .common import _check_dates, _error, _json, mask_email
from .context import ToolContext


def build(ctx: ToolContext) -> dict[str, Any]:
    """Инструменты темы по имени; порядок в списке агента задаёт agent/tools/__init__.py."""
    client = ctx.client
    budget = ctx.budget
    reports_root = ctx.reports_root
    confirm_write = ctx.confirm_write
    perform = ctx.perform
    profile_email = ctx.profile_email

    @beta_async_tool
    async def get_report_status() -> str:
        """Последние заказанные отчёты: job_id, название, формат, дата. Бесплатно."""
        try:
            response = await client.reports.get_report_job_list_v1()
            items = list(response.data or [])
        except (APIError, RequestValidationError) as exc:
            return _error(exc)
        rows = [
            {
                "job_id": item.job_id,
                "date": item.date,
                "report_name": item.report_name,
                "report_format": item.report_format,
                "contract_id": item.contract_id,
            }
            for item in items
        ]
        rows.sort(key=lambda row: row["date"], reverse=True)
        return _json(
            {
                "total": len(rows),
                "shown": len(rows[:10]),
                **(
                    {"hint": "Показаны последние 10 отчётов."}
                    if len(rows) > 10
                    else {}
                ),
                "reports": rows[:10],
            }
        )

    @beta_async_tool
    async def order_fuel_report(
        date_from: str,
        date_to: str,
        report_format: str = "xlsx",
        email: str | None = None,
        card_ids: list[str] | None = None,
        group_ids: list[str] | None = None,
    ) -> str:
        """Заказать транзакционный отчёт за период. ИЗМЕНЯЕТ ДАННЫЕ, по тарифу.

        Асинхронно: вернёт job_id для get_report_status. format: xlsx, xml, pdf
        или csv. С email отчёт уйдёт на почту, без него — по ссылке в кабинете.
        """
        if problem := _check_dates(date_from, date_to):
            return _json({"error": "bad_dates", "message": problem})
        report_format = (report_format or "").strip().lower()
        if report_format not in {"xlsx", "xml", "pdf", "csv"}:
            return _json(
                {"error": "bad_request", "message": "format: xlsx, xml, pdf или csv."}
            )
        email = (email or "").strip() or None
        if email and "@" not in email:
            return _json({"error": "bad_request", "message": "Похоже, email указан неверно."})
        email_source = "указан пользователем" if email else None
        if email is None:
            try:
                email = await profile_email()
            except (APIError, RequestValidationError, ContractSelectionError) as exc:
                return _error(exc)
            email_source = "по умолчанию: почта из профиля пользователя" if email else None
        cards = [str(item).strip() for item in (card_ids or []) if str(item).strip()] or None
        groups = [str(item).strip() for item in (group_ids or []) if str(item).strip()] or None
        summary = (
            f"Заказать отчёт {report_format} за {date_from}…{date_to}"
            + (f" по картам {len(cards)}" if cards else "")
            + (f" по группам {len(groups)}" if groups else "")
            + (f" и отправить на {email}" if email else "")
            + "? Отчёт тарифный и формируется асинхронно."
        )
        approval, denied = await confirm_write(
            "order_fuel_report",
            summary,
            len(cards or groups or [1]),
            {"period": f"{date_from}…{date_to}", "cards": len(cards or []), "groups": len(groups or [])},
            {"date_from": date_from, "date_to": date_to, "format": report_format, "email": email,
             "card_ids": cards, "group_ids": groups},
        )
        if denied:
            return denied
        response, failed = await perform(
            approval,
            lambda: client.reports.order_report_v1(
                start=date_from,
                end=date_to,
                report_format=report_format,
                email=email,
                cards_list=cards,
                group_id=groups,
            ),
            "заказанные отчёты в get_report_status",
        )
        if failed:
            return failed
        ctx.finish(approval, "done" if response.data else "not_done", {"job_ids": list(response.data or [])})
        return _json(
            {
                "status": "ordered",
                "period": f"{date_from}…{date_to}",
                "format": report_format,
                # Модели — маскированная почта: полный адрес человек видел в подтверждении.
                "email": mask_email(email),
                "email_source": email_source,
                "job_ids": list(response.data or []),
            }
        )

    @beta_async_tool
    async def download_fuel_report(
        job_id: str, report_format: str = "xlsx", destination: str | None = None
    ) -> str:
        """Скачать готовый отчёт в папку reports. Тарифный запрос API.

        Сначала проверь get_report_status: job_id должен быть готов. Файл
        сохраняется локально и не вставляется в чат. report_format нужен только
        для расширения; destination — имя файла без пути.
        """
        job_id = str(job_id).strip()
        if not job_id:
            return _json({"error": "bad_request", "message": "Нужен job_id."})
        fmt = (report_format or "xlsx").strip().lower()
        if fmt not in {"xlsx", "xml", "pdf", "csv"}:
            return _json(
                {"error": "bad_request", "message": "report_format: xlsx, xml, pdf или csv."}
            )
        if destination:
            name = Path(destination).name
            if name != destination or name in {"", ".", ".."}:
                return _json(
                    {"error": "bad_request", "message": "destination — только имя файла без пути."}
                )
        else:
            safe = re.sub(r"[^A-Za-z0-9._-]", "_", job_id)[:80] or "report"
            name = f"fuel-report-{safe}.{fmt}"
        root = reports_root.resolve()
        path = (root / name).resolve()
        if path.parent != root:
            return _json({"error": "bad_request", "message": "Недопустимый путь файла."})
        try:
            root.mkdir(parents=True, exist_ok=True)
            budget.spend()
            saved = Path(
                await client.reports.download_report_file_v1_to(
                    job_id=job_id, destination=path
                )
            )
        except (APIError, BudgetExceededError, RequestValidationError, OSError) as exc:
            return _error(exc)
        return _json(
            {
                "status": "downloaded",
                "job_id": job_id,
                "path": str(saved),
                "bytes": saved.stat().st_size if saved.exists() else None,
            }
        )

    return {
        "get_report_status": get_report_status,
        "order_fuel_report": order_fuel_report,
        "download_fuel_report": download_fuel_report,
    }
