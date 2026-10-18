"""Загрузка документов в базу знаний: форматы, версии, грифы, поиск."""

from __future__ import annotations

import asyncio
import json
import shutil
import zipfile
from pathlib import Path

import pytest

from agent.config import Mode
from agent.ingest import IngestError, Store, extract_text, split_chunks
from agent.knowledge import load_store_sections, search
from scripts.ingest_knowledge import main as ingest_main
from tests.test_agent import make_tools

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
S = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"


def make_zip(path: Path, files: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, body in files.items():
            archive.writestr(name, body)
    return path


def docx(path: Path) -> Path:
    body = (
        f'<w:document xmlns:w="{W}"><w:body>'
        '<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Блокировка карты</w:t></w:r></w:p>'
        "<w:p><w:r><w:t>Карту блокируют через личный кабинет в течение двух часов.</w:t></w:r></w:p>"
        "</w:body></w:document>"
    )
    return make_zip(path, {"word/document.xml": body})


def xlsx(path: Path) -> Path:
    shared = f'<sst xmlns="{S}"><si><t>Тариф</t></si><si><t>Цена</t></si><si><t>Free</t></si></sst>'
    sheet = (
        f'<worksheet xmlns="{S}"><sheetData>'
        '<row><c t="s"><v>0</v></c><c t="s"><v>1</v></c></row>'
        '<row><c t="s"><v>2</v></c><c><v>0</v></c></row>'
        "</sheetData></worksheet>"
    )
    return make_zip(path, {"xl/sharedStrings.xml": shared, "xl/worksheets/sheet1.xml": sheet})


def pptx(path: Path) -> Path:
    slide = f'<p:sld xmlns:p="p" xmlns:a="{A}"><a:p><a:r><a:t>Порядок выдачи карт водителям</a:t></a:r></a:p></p:sld>'
    return make_zip(path, {"ppt/slides/slide1.xml": slide})


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Store:
    monkeypatch.setenv("OPTI24_KNOWLEDGE_STORE", str(tmp_path / "store"))
    return Store(tmp_path / "store")


def test_extracts_text_from_common_formats(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# Заголовок\nТекст", encoding="utf-8")
    (tmp_path / "b.html").write_text(
        "<nav>меню</nav><h2>Тарифы</h2><p>Free бесплатно</p><script>x()</script>", encoding="utf-8"
    )
    (tmp_path / "c.csv").write_text("Тариф;Цена\nFree;0\nPro;990\n", encoding="cp1251")
    (tmp_path / "d.json").write_text('{"faq": "вопрос"}', encoding="utf-8")
    (tmp_path / "e.eml").write_bytes(
        "Subject: Новые правила\nContent-Type: text/plain; charset=utf-8\n\nС первого числа меняется порядок.".encode()
    )
    assert "Заголовок" in extract_text(tmp_path / "a.md")
    html = extract_text(tmp_path / "b.html")
    assert "## Тарифы" in html and "Free бесплатно" in html and "меню" not in html and "x()" not in html
    assert "Тариф: Free; Цена: 0" in extract_text(tmp_path / "c.csv")  # шапка повторяется в каждой строке
    assert "Тариф: Pro; Цена: 990" in extract_text(tmp_path / "c.csv")
    assert "вопрос" in extract_text(tmp_path / "d.json")
    assert "Новые правила" in extract_text(tmp_path / "e.eml")
    assert "## Блокировка карты" in extract_text(docx(tmp_path / "f.docx"))
    assert "Цена: 0" in extract_text(xlsx(tmp_path / "g.xlsx"))
    assert "Порядок выдачи карт" in extract_text(pptx(tmp_path / "h.pptx"))


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="нет poppler")
def test_pdf_without_text_layer_is_rejected(tmp_path: Path) -> None:
    broken = tmp_path / "scan.pdf"
    broken.write_bytes(b"%PDF-1.4 not really")
    with pytest.raises(IngestError):
        extract_text(broken)


@pytest.mark.parametrize(("name", "word"), [("p.jpg", "OCR"), ("a.mp3", "ASR"), ("x.doc", "docx"), ("z.zip", "по одному")])
def test_unsupported_formats_explain_what_to_do(tmp_path: Path, name: str, word: str) -> None:
    (tmp_path / name).write_bytes(b"x")
    with pytest.raises(IngestError, match=word):
        extract_text(tmp_path / name)


def test_broken_office_file_is_a_clear_error(tmp_path: Path) -> None:
    (tmp_path / "bad.docx").write_bytes(b"not a zip")
    with pytest.raises(IngestError, match="повреждён"):
        extract_text(tmp_path / "bad.docx")


def test_chunks_follow_headings_and_clauses() -> None:
    text = "## Срок претензии\nПретензию подают в течение суток.\n3.2. Штрафы\nНеустойка составляет 0,1 процента в день."
    headings = [h for h, _ in split_chunks(text, "Документ")]
    assert headings == ["Срок претензии", "п. 3.2 Штрафы"]
    long = "## Длинный\n" + "слово " * 600
    assert all(len(t) <= 900 for _, t in split_chunks(long, "Д"))


def test_added_document_is_found_with_source_and_version(store: Store, tmp_path: Path) -> None:
    doc = store.add(docx(tmp_path / "r.docx"), title="Регламент блокировки", kind="regulation", version="2026-10")
    found = search(load_store_sections(), "как заблокировать карту через личный кабинет")
    assert found and "Регламент блокировки" in found[0]["source"] and found[0]["edition"] == "2026-10"
    assert doc.chunks >= 1 and search(load_store_sections(), "погода в Москве") == []


def test_new_version_archives_old_one(store: Store, tmp_path: Path) -> None:
    first = tmp_path / "v1.md"
    first.write_text("## Срок\nСрок ответа на претензию десять дней.", encoding="utf-8")
    second = tmp_path / "v2.md"
    second.write_text("## Срок\nСрок ответа на претензию пять дней.", encoding="utf-8")
    store.add(first, title="Претензии", version="v1")
    store.add(second, title="Претензии", version="v2")
    texts = [s.text for s in load_store_sections()]
    assert texts == ["Срок ответа на претензию пять дней."]
    assert [d.status for d in store.documents()] == ["archived", "active"]


def test_duplicate_and_bad_metadata_are_rejected(store: Store, tmp_path: Path) -> None:
    file = tmp_path / "a.md"
    file.write_text("## Раздел\nДостаточно длинный текст раздела для базы знаний.", encoding="utf-8")
    store.add(file, title="Док", version="v1")
    with pytest.raises(IngestError, match="уже загружен"):
        store.add(file, title="Док", version="v1")
    with pytest.raises(IngestError, match="ГГГГ-ММ-ДД"):
        store.add(file, title="Другой", valid_from="1 октября")
    with pytest.raises(IngestError, match="Раздел"):
        store.add(file, title="Другой", kind="memo")


def test_archive_removes_document_from_search(store: Store, tmp_path: Path) -> None:
    file = tmp_path / "a.md"
    file.write_text("## Скидка\nСкидка на дизельное топливо зависит от объёма.", encoding="utf-8")
    doc = store.add(file, title="Скидки")
    assert search(load_store_sections(), "скидка на дизельное топливо")
    store.archive(doc.id)
    assert search(load_store_sections(), "скидка на дизельное топливо") == []


def test_internal_documents_can_be_hidden(store: Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    file = tmp_path / "a.md"
    file.write_text("## Секрет\nВнутренний порядок согласования скидок руководителем.", encoding="utf-8")
    store.add(file, title="Внутренний", classification="internal")
    monkeypatch.setenv("OPTI24_KNOWLEDGE_INTERNAL", "0")
    assert load_store_sections() == []


def test_lookup_knowledge_tool_answers_from_ingested_document(
    store: Store, tmp_path: Path
) -> None:
    file = tmp_path / "a.md"
    file.write_text("## Спецодежда\nСпецодежду выдают водителям один раз в год под подпись.", encoding="utf-8")
    store.add(file, title="Регламент водителей", kind="regulation", version="v1")
    _, budget, tools = make_tools(Mode.READ_ONLY, answer=False, tmp=tmp_path)
    out = json.loads(asyncio.run(tools["lookup_knowledge"].call({"question": "когда выдают спецодежду водителям"})))
    assert out["found"] >= 1 and "Регламент водителей" in out["sections"][0]["source"]
    assert budget.spent == 0


def test_cli_add_list_archive(store: Store, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    file = tmp_path / "a.md"
    file.write_text("## Раздел\nДостаточно длинный текст раздела для базы знаний.", encoding="utf-8")
    base = ["--store", str(tmp_path / "cli")]
    assert ingest_main([*base, "add", str(file), "--title", "Док", "--kind", "faq"]) == 0
    assert ingest_main([*base, "list"]) == 0
    doc_id = Store(tmp_path / "cli").documents()[0].id
    assert ingest_main([*base, "archive", doc_id]) == 0
    assert ingest_main([*base, "add", str(tmp_path / "nope.md"), "--title", "Х"]) == 1
    assert "Ошибка" in capsys.readouterr().err
