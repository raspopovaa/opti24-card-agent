"""Скачивание публичных страниц сайта в базу знаний: без сети, на поддельном сайте."""

from __future__ import annotations

import json
from pathlib import Path

from agent.ingest import Store
from scripts.fetch_site import crawl, guess_kind, ingest, main

HOME = "https://www.opti-24.ru/"
PAGE = (
    "<html><head><title>{title}</title></head><body><nav>Меню</nav>"
    "<h2>{title}</h2><p>{text}</p>{links}</body></html>"
)


def page(title: str, text: str, *links: str) -> tuple[int, str, bytes]:
    anchors = "".join(f'<a href="{href}">{label}</a>' for href, label in links)
    return 200, "text/html", PAGE.format(title=title, text=text, links=anchors).encode()


def make_site(**overrides: tuple[int, str, bytes]) -> tuple[dict[str, tuple[int, str, bytes]], list[str]]:
    site = {
        "https://www.opti-24.ru/robots.txt": (200, "text/plain", b"User-agent: *\nDisallow: /private/\n"),
        HOME: page(
            "Топливные карты", "Топливные карты для юридических лиц и автопарков по всей стране.",
            ("/tariffs/", "Тарифы"), ("/faq/#top", "Вопросы"), ("/lk/", "Вход"), ("/private/x/", "Закрыто"),
            ("https://example.com/", "Чужой сайт"), ("https://cdn.opti-24.ru/upload/dogovor.pdf", "Договор"),
            ("/logo.png", "Логотип"),
        ),
        "https://www.opti-24.ru/tariffs/": page(
            "Тарифы", "Тариф Drive даёт скидку на дизельное топливо на АЗС партнёров сети.", ("/", "Главная"),
        ),
        "https://www.opti-24.ru/faq/": page(
            "Вопросы и ответы", "Карту можно заблокировать в личном кабинете в любое время суток.",
        ),
        "https://cdn.opti-24.ru/upload/dogovor.pdf": (200, "text/html", b"<html>captcha</html>"),
    }
    site.update(overrides)
    calls: list[str] = []
    return site, calls


def fake(site: dict[str, tuple[int, str, bytes]], calls: list[str]):
    def fetch(url: str) -> tuple[int, str, bytes]:
        calls.append(url)
        return site.get(url, (404, "", b""))
    return fetch


def test_crawl_stays_on_public_site_pages(tmp_path: Path) -> None:
    site, calls = make_site()
    items = crawl(tmp_path, delay=0, fetch=fake(site, calls))
    fetched = set(calls)
    assert "https://www.opti-24.ru/tariffs/" in fetched
    assert "https://www.opti-24.ru/faq/" in fetched  # якорь отброшен
    assert not any("/lk/" in u or "/private/" in u or "example.com" in u or ".png" in u for u in fetched)
    assert calls.count(HOME) == 1  # ссылка «Главная» не скачивается второй раз
    saved = {i.url: i for i in items if i.status == "saved"}
    assert saved["https://www.opti-24.ru/tariffs/"].kind == "tariff"
    assert saved["https://www.opti-24.ru/faq/"].kind == "faq"
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == len(items)


def test_document_served_as_html_is_reported_not_saved(tmp_path: Path) -> None:
    site, calls = make_site()
    items = crawl(tmp_path, delay=0, fetch=fake(site, calls))
    pdf = next(i for i in items if i.url.endswith("dogovor.pdf"))
    assert pdf.status == "not_a_file"
    assert not pdf.file


def test_max_pages_limits_html_pages(tmp_path: Path) -> None:
    site, calls = make_site()
    items = crawl(tmp_path, max_pages=1, delay=0, fetch=fake(site, calls))
    assert [i.url for i in items if i.status == "saved"] == [HOME]


def test_same_titles_get_path(tmp_path: Path) -> None:
    site, calls = make_site(**{
        "https://www.opti-24.ru/faq/": page("Топливные карты", "Другая страница с тем же заголовком и своим текстом."),
    })
    items = crawl(tmp_path, delay=0, fetch=fake(site, calls))
    titles = [i.title for i in items if i.status == "saved"]
    assert "Топливные карты (/faq/)" in titles
    assert len(set(titles)) == len(titles)


def test_ingest_adds_public_documents_and_skips_unchanged(tmp_path: Path) -> None:
    site, calls = make_site()
    out, store = tmp_path / "site", Store(tmp_path / "store")
    crawl(out, delay=0, fetch=fake(site, calls))
    added, skipped = ingest(out, store)
    assert len(added) == 3 and not skipped
    docs = store.documents()
    assert {d.classification for d in docs} == {"public"}
    assert any(d.source == "https://www.opti-24.ru/tariffs/" for d in docs)
    texts = " ".join(c["text"] for _, c in store.active_chunks(("public",)))
    assert "Тариф Drive" in texts and "Меню" not in texts
    added, skipped = ingest(out, store)
    assert not added and len(skipped) == 3


def test_main_reports_empty_site(tmp_path: Path, capsys) -> None:
    code = main(["--out", str(tmp_path), "--delay", "0"], fetch=lambda url: (0, "", b""))
    assert code == 1
    assert "ничего не отдал" in capsys.readouterr().err


def test_main_ingest(tmp_path: Path, capsys) -> None:
    site, calls = make_site()
    args = ["--out", str(tmp_path / "site"), "--delay", "0", "--ingest", "--store", str(tmp_path / "store")]
    assert main(args, fetch=fake(site, calls)) == 0
    output = capsys.readouterr().out
    assert "В базе знаний: 3" in output
    assert "dogovor.pdf — сайт отдал страницу вместо файла" in output


def test_guess_kind() -> None:
    assert guess_kind("https://cdn.opti-24.ru/upload/api_prices.docx", "") == "tariff"
    assert guess_kind("https://www.opti-24.ru/dogovor/", "Договор") == "regulation"
    assert guess_kind("https://www.opti-24.ru/about/", "О компании") == "other"
