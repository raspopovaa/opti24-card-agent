"""Скачивание публичных страниц сайта в базу знаний: без сети, на поддельном сайте."""

from __future__ import annotations

import json
from pathlib import Path

from agent.ingest import Store
from scripts.fetch_site import clean_link_text, crawl, guess_kind, ingest, main

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
    added, skipped, archived = ingest(out, store)
    assert len(added) == 3 and not skipped and not archived
    docs = store.documents()
    assert {d.classification for d in docs} == {"public"}
    assert any(d.source == "https://www.opti-24.ru/tariffs/" for d in docs)
    texts = " ".join(c["text"] for _, c in store.active_chunks(("public",)))
    assert "Тариф Drive" in texts and "Меню" not in texts
    added, skipped, archived = ingest(out, store)
    assert not added and len(skipped) == 3 and not archived


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
    assert guess_kind("https://www.opti-24.ru/about/", "О нас — ценности и подход") == "other"
    assert guess_kind("https://www.opti-24.ru/contacts/", "Контакты – как связаться с нами") == "other"


def test_clean_link_text() -> None:
    assert clean_link_text("14.11.202327.3 KbДоверенность на получение сервисных карт.docx Скачать") == (
        "Доверенность на получение сервисных карт"
    )
    assert clean_link_text("Договор") == "Договор"


def test_one_address_per_page_and_no_news(tmp_path: Path) -> None:
    site, calls = make_site(**{
        HOME: page(
            "Топливные карты", "Топливные карты для юридических лиц и автопарков по всей стране.",
            ("/azs/", "АЗС"), ("/azs", "АЗС"), ("https://opti-24.ru/azs/?owntype=opti", "АЗС ОПТИ"),
            ("/news/1/", "Новость"),
        ),
        "https://www.opti-24.ru/azs/": page("АЗС", "Список АЗС, где принимают топливные карты компании."),
    })
    crawl(tmp_path, delay=0, fetch=fake(site, calls))
    assert calls.count("https://www.opti-24.ru/azs/") == 1
    assert not any("news" in u or "owntype" in u for u in calls)


def test_news_with_flag(tmp_path: Path) -> None:
    site, calls = make_site(**{HOME: page("Главная", "Главная страница сайта о топливных картах.", ("/news/1/", "Новость"))})
    crawl(tmp_path, delay=0, with_news=True, fetch=fake(site, calls))
    assert "https://www.opti-24.ru/news/1/" in calls


def test_svg_title_does_not_leak_into_page_title(tmp_path: Path) -> None:
    body = (
        "<html><head><title>Документы</title></head><body><svg><title>x-circle</title></svg>"
        "<p>Формы и соглашения</p></body></html>"
    ).encode()
    site, calls = make_site(**{HOME: (200, "text/html", body)})
    items = crawl(tmp_path, delay=0, fetch=fake(site, calls))
    assert items[0].title == "Документы"


def test_prune_archives_pages_gone_from_site(tmp_path: Path) -> None:
    site, calls = make_site()
    out, store = tmp_path / "site", Store(tmp_path / "store")
    crawl(out, delay=0, fetch=fake(site, calls))
    ingest(out, store)
    site["https://www.opti-24.ru/"] = page("Топливные карты", "Топливные карты для юридических лиц и автопарков по всей стране.")
    crawl(out, delay=0, fetch=fake(site, calls))
    _, _, archived = ingest(out, store, prune=False)
    assert not archived
    _, _, archived = ingest(out, store, prune=True)
    assert len(archived) == 2
    assert {d.source for d in store.documents() if d.status == "active"} == {HOME}


def test_relabelled_page_replaces_old_document(tmp_path: Path) -> None:
    site, calls = make_site()
    out, store = tmp_path / "site", Store(tmp_path / "store")
    crawl(out, delay=0, fetch=fake(site, calls))
    ingest(out, store)
    site["https://www.opti-24.ru/tariffs/"] = page("Тарифы 2026", "Новый тариф Drive для дизельного топлива на АЗС сети.")
    crawl(out, delay=0, fetch=fake(site, calls))
    _, _, archived = ingest(out, store)
    assert archived == ["opti-24.ru: Тарифы (https://www.opti-24.ru/tariffs/)"]
    active = [d for d in store.documents() if d.status == "active" and d.source.endswith("/tariffs/")]
    assert [d.title for d in active] == ["opti-24.ru: Тарифы 2026"]
