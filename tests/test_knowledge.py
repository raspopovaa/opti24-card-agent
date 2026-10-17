"""База знаний по договору: выжимки, поиск, инструмент и проверка актуальности документов."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from agent.config import Mode
from agent.knowledge import DEFAULT_DIR, load_sections, search
from scripts.sync_knowledge import accept, check_sources
from scripts.sync_knowledge import main as sync_main
from tests.test_agent import make_tools

QUESTIONS = {
    "как заблокировать потерянную карту": "Как заблокировать или разблокировать карту",
    "вернуть аванс при расторжении": "Срок действия, расторжение и возврат аванса",
    "какой сервисный сбор": "Сервисный сбор",
    "почему не отпустили топливо": "Почему по карте могут не отпустить топливо",
    "до какого числа вернуть документы": "Закрывающие документы",
    "скидка на дизель в Тверской области": "Скидка на дизельное топливо по регионам",
    "что писать в платёжке при пополнении счёта договора": "Пополнение счёта договора",
    "что будет когда закончится пакет запросов API": "Что будет, когда пакет закончится",
    "что значит код отказа 58 в чеке": "Коды отказа терминала",
}


@pytest.mark.parametrize(("question", "section"), QUESTIONS.items())
def test_typical_client_questions_find_the_right_section(question: str, section: str) -> None:
    assert search(load_sections(), question)[0]["section"] == section


def test_unrelated_question_finds_nothing() -> None:
    assert search(load_sections(), "погода в Москве") == []


def test_every_summary_names_its_source_and_edition() -> None:
    registry = json.loads((DEFAULT_DIR / "sources.json").read_text(encoding="utf-8"))
    known_ids = {source["id"] for source in registry["sources"]}
    summaries = [p for p in DEFAULT_DIR.glob("*.md") if p.name.lower() != "readme.md"]
    assert summaries
    for path in summaries:
        head = path.read_text(encoding="utf-8").split("\n---\n", 1)[0]
        assert "edition:" in head and "checked:" in head, path.name
        ids = head.split("sources:", 1)[1].split("\n", 1)[0].strip(" []").split(",")
        assert {i.strip() for i in ids} <= known_ids, path.name
    topics = {topic for source in registry["sources"] for topic in source["topics"]}
    assert topics <= {p.name for p in summaries}


def test_lookup_knowledge_tool_is_free_and_says_when_nothing_found(tmp_path: Path) -> None:
    _, budget, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path)
    found = json.loads(asyncio.run(tools["lookup_knowledge"].call({"question": "как вернуть аванс"})))
    assert found["found"] >= 1 and found["sections"][0]["edition"]
    missing = json.loads(asyncio.run(tools["lookup_knowledge"].call({"question": "погода в Москве"})))
    assert missing["found"] == 0 and "менеджеру" in missing["message"]
    assert budget.spent == 0


def _registry(tmp_path: Path) -> Path:
    path = tmp_path / "sources.json"
    sources = [
        {"id": "offer", "title": "Оферта", "url": "u", "files": ["dogovor*.pdf"], "topics": ["a.md"], "sha256": None},
        {"id": "tariffs", "title": "Тарифы", "url": None, "site_section": "раздел API",
         "files": ["api_prices*.docx"], "topics": ["b.md"], "sha256": None},
    ]
    path.write_text(json.dumps({"sources": sources}, ensure_ascii=False), encoding="utf-8")
    return path


def test_sync_detects_new_changed_and_missing_documents(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox" / "Telegram"
    inbox.mkdir(parents=True)
    (inbox / "dogovor (1).pdf").write_bytes(b"v1")
    registry_path = _registry(tmp_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    states = {s.source_id: s.state for s in check_sources(registry, [tmp_path / "inbox"])}
    assert states == {"offer": "new", "tariffs": "missing"}

    statuses = check_sources(registry, [tmp_path / "inbox"])
    assert accept(registry, statuses, ["offer", "tariffs"]) == ["offer"]
    assert check_sources(registry, [tmp_path / "inbox"])[0].state == "unchanged"

    (inbox / "dogovor (1).pdf").write_bytes(b"v2")
    assert check_sources(registry, [tmp_path / "inbox"])[0].state == "changed"


def test_sync_cli_exit_code_and_accept_writes_registry(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "dogovor.pdf").write_bytes(b"v1")
    registry_path = _registry(tmp_path)
    args = ["--inbox", str(tmp_path), "--registry", str(registry_path)]
    assert sync_main(args) == 1  # новый документ: выжимки нужно пересмотреть
    assert "пересмотреть: a.md" in capsys.readouterr().out
    assert sync_main([*args, "--accept", "offer"]) == 0
    saved = json.loads(registry_path.read_text(encoding="utf-8"))["sources"][0]
    assert saved["sha256"] and saved["accepted"]
    assert sync_main(args) == 0
