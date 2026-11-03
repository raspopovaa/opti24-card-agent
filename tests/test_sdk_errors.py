"""Распознанные ответы API из apisdkopti24 0.0.7: что агент отдаёт модели."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from apisdkopti24.errors import ServerError, build_api_error

from agent.config import Mode
from agent.tools.common import _error
from tests.test_agent import _station, make_tools
from tests.test_write_controls import audit, call, setup


def api_error(code: int, error_type: str, message: str, method: str, endpoint: str = "/vip/v1/x") -> Any:
    body = {"status": {"code": code, "errors": [{"type": error_type, "message": message}]}}
    return build_api_error(status_code=code, body=body, endpoint=endpoint, method_name=method)


def test_error_carries_reason_advice_and_no_retry() -> None:
    exc = api_error(403, "accessDenied", "Нет доступа (Не найден контракт, карта или группа)", "get_card_detail")
    out = json.loads(_error(exc))
    assert out["error"] == "ObjectNotAccessibleError" and out["reason"] == "object_not_accessible"
    assert "list_cards" in out["advice"] and out["hint"]
    assert "не поможет" in out["retry"]


def test_check_purchase_refusal_is_answer_with_decoded_code(tmp_path: Path) -> None:
    fake, _, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path, budget_limit=10)
    _station(fake, region="MOS", allowed=True)

    async def refused(**_: object) -> Any:
        raise api_error(403, "accessDenied", "ОТКАЗ. Недостаточно средств. [Код ошибки: 51]", "check_purchase")

    fake.final_prices.check_purchase = refused
    args = {"card_id": "c1", "poi_id": "azs-1", "product": "дт", "liters": 50}
    out = json.loads(asyncio.run(tools["diagnose_fueling"].call(args)))
    purchase = out["checks"]["check_purchase"]
    assert purchase["allowed"] is False and purchase["processing_code"] == 51
    assert purchase["decoded"]["in_table"] is True
    assert out["likely_causes"][0].startswith("процессинг отказал в покупке с кодом 51")


def test_region_limit_not_created_is_error_not_unknown(tmp_path: Path) -> None:
    fake, tools, _ = setup(tmp_path)

    async def not_created(**_: object) -> Any:
        # Так SDK размечает этот ответ после 0.0.7 (правило region_limit_point_id ещё не на PyPI).
        raise ServerError(500, "технические проблемы", method_name="set_region_limit", reason="region_limit_point_id")

    fake.region_limits.set_region_limit = not_created
    out = call(tools, "set_region_limit", {"card_id": "c1", "region": "MOS", "limit_type": 2})
    assert out["reason"] == "region_limit_point_id" and "status" not in out and "ID АЗС" in out["advice"]
    assert [r["status"] for r in audit(tmp_path) if r["event"] == "outcome"] == ["error"]


def test_gateway_reply_on_change_stays_unknown(tmp_path: Path) -> None:
    fake, tools, _ = setup(tmp_path)

    async def gateway(**_: object) -> Any:
        body = {"status": {"code": 504, "errors": [{"type": "internalError", "message": "Gateway Timeout"}]}}
        raise build_api_error(
            status_code=504, body=body, endpoint="/vip/v1/setRegionLimit", method_name="set_region_limit", mutation=True
        )

    fake.region_limits.set_region_limit = gateway
    assert call(tools, "set_region_limit", {"card_id": "c1", "region": "MOS", "limit_type": 2})["status"] == "unknown"
