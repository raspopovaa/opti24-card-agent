"""Публичные страницы и документы сайта opti-24.ru в базу знаний агента.

Запускается там, где сайт открывается (облачные IP сайт может не пускать):

    python scripts/fetch_site.py --out ~/Downloads/opti24-site
    python scripts/fetch_site.py --out ~/Downloads/opti24-site --ingest

Скрипт обходит страницы www.opti-24.ru, начиная с главной. Ссылки на pdf, docx, xlsx и pptx
он скачивает и с cdn.opti-24.ru. Правила robots.txt соблюдаются. Личный кабинет, вход и поиск
не обходятся. Файлы и manifest.json кладутся в --out.

С --ingest файлы загружаются в knowledge_store/ с грифом public. Повторный запуск обновляет
изменившиеся документы, а неизменные пропускает. Если вместо документа сайт отдал HTML (такой
файл отдаётся только браузеру), скрипт перечислит эти ссылки: скачайте их вручную и загрузите
через scripts/ingest_knowledge.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
import urllib.robotparser
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.ingest import MAX_FILE_BYTES, IngestError, Store  # noqa: E402

START_URL = "https://www.opti-24.ru/"
PAGE_HOSTS = {"www.opti-24.ru", "opti-24.ru"}
FILE_HOSTS = PAGE_HOSTS | {"cdn.opti-24.ru"}
DOCUMENT_TYPES = (".pdf", ".docx", ".xlsx", ".pptx")
SKIPPED_TYPES = (".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".css", ".js", ".zip", ".mp4", ".doc", ".xls")
PRIVATE_PATHS = ("/lk", "/login", "/auth", "/cabinet", "/personal", "/bitrix", "/search", "/ajax", "/upload/iblock")
USER_AGENT = "opti24-card-agent knowledge sync/1.0"

Fetch = Callable[[str], tuple[int, str, bytes]]


@dataclass(slots=True)
class Item:
    url: str
    title: str
    file: str
    kind: str
    status: str  # saved | not_a_file | error
    detail: str = ""


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self.title = ""
        self._href: str | None = None
        self._text: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            self._href = dict(attrs).get("href")
            self._text = []
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href:
            self.links.append((self._href, " ".join("".join(self._text).split())))
            self._href = None
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)
        if self._in_title:
            self.title += data


def http_fetch(url: str) -> tuple[int, str, bytes]:
    """Статус, тип содержимого и тело ответа; 0 — сайт не ответил."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310  # nosec B310
            return response.status, response.headers.get_content_type(), response.read(MAX_FILE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        return exc.code, "", b""
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"Нет ответа: {url} ({exc})", file=sys.stderr)
        return 0, "", b""


def guess_kind(url: str, title: str) -> str:
    text = f"{url} {title}".casefold()
    for kind, words in (
        ("tariff", ("тариф", "price", "стоимост", "цен")),
        ("faq", ("faq", "вопрос", "ответ")),
        ("instruction", ("инструкц", "instruction", "памятк", "как ")),
        ("regulation", ("договор", "dogovor", "оферт", "регламент", "правил")),
    ):
        if any(word in text for word in words):
            return kind
    return "other"


def _normalize(url: str) -> str:
    url = urldefrag(url)[0]
    parts = urlparse(url)
    if parts.scheme == "http":
        url = "https" + url[4:]
    return url


def _is_document(url: str) -> bool:
    return urlparse(url).path.casefold().endswith(DOCUMENT_TYPES)


def _wanted(url: str) -> bool:
    parts = urlparse(url)
    path = parts.path.casefold()
    if parts.scheme not in {"http", "https"} or path.endswith(SKIPPED_TYPES):
        return False
    if _is_document(url):
        return parts.hostname in FILE_HOSTS
    return parts.hostname in PAGE_HOSTS and not path.startswith(PRIVATE_PATHS)


def _robots(start: str, fetch: Fetch) -> urllib.robotparser.RobotFileParser:
    robots = urllib.robotparser.RobotFileParser()
    status, _, body = fetch(urljoin(start, "/robots.txt"))
    robots.parse(body.decode("utf-8", "replace").splitlines() if status == 200 else [])
    return robots


def crawl(out: Path, *, start: str = START_URL, max_pages: int = 60, delay: float = 1.0,
          fetch: Fetch = http_fetch) -> list[Item]:
    """Скачивает страницы и документы в ``out`` и пишет manifest.json."""
    out.mkdir(parents=True, exist_ok=True)
    robots = _robots(start, fetch)
    queue: list[tuple[str, str]] = [(_normalize(start), "")]
    seen = {queue[0][0]}
    items: list[Item] = []
    pages = 0
    while queue:
        url, link_text = queue.pop(0)
        document = _is_document(url)
        if not document and pages >= max_pages:
            continue
        if not robots.can_fetch(USER_AGENT, url):
            continue
        if items or pages:
            time.sleep(delay)
        status, content_type, body = fetch(url)
        if status != 200 or len(body) > MAX_FILE_BYTES:
            detail = "больше 30 МБ" if status == 200 else f"HTTP {status or 'нет ответа'}"
            items.append(Item(url, link_text, "", "", "error", detail))
            continue
        name = hashlib.sha1(url.encode(), usedforsecurity=False).hexdigest()[:12]
        if document:
            if content_type == "text/html":
                items.append(Item(url, link_text, "", "", "not_a_file", "сайт отдал страницу вместо файла"))
                continue
            title = link_text or Path(urlparse(url).path).stem
            file = name + Path(urlparse(url).path).suffix.casefold()
        else:
            pages += 1
            parser = _Links()
            parser.feed(body.decode("utf-8", "replace"))
            title = " ".join(parser.title.split()) or urlparse(url).path
            file = name + ".html"
            for href, text in parser.links:
                link = _normalize(urljoin(url, href))
                if link not in seen and _wanted(link):
                    seen.add(link)
                    queue.append((link, text))
        (out / file).write_bytes(body)
        items.append(Item(url, title, file, guess_kind(url, title), "saved"))
    _unique_titles(items)
    (out / "manifest.json").write_text(
        json.dumps([asdict(i) for i in items], ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return items


def _unique_titles(items: list[Item]) -> None:
    """Одинаковые заголовки страниц различаются путём: иначе одна страница заменит другую в базе."""
    saved = [i for i in items if i.status == "saved"]
    titles = [i.title for i in saved]
    for item in saved:
        if titles.count(item.title) > 1:
            item.title = f"{item.title} ({urlparse(item.url).path})"


def ingest(out: Path, store: Store) -> tuple[list[str], list[str]]:
    """Загружает сохранённые файлы с грифом public; возвращает (загружено, пропущено)."""
    items = [Item(**i) for i in json.loads((out / "manifest.json").read_text(encoding="utf-8"))]
    added, skipped = [], []
    for item in items:
        if item.status != "saved":
            continue
        try:
            doc = store.add(
                out / item.file, title=f"opti-24.ru: {item.title}", kind=item.kind,
                version=date.today().isoformat(), classification="public", source=item.url,
            )
        except IngestError as exc:
            skipped.append(f"{item.url}: {exc}")
        else:
            added.append(f"{doc.title}, фрагментов: {doc.chunks}")
    return added, skipped


def main(argv: list[str] | None = None, fetch: Fetch = http_fetch) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="папка для скачанных файлов")
    parser.add_argument("--start", default=START_URL)
    parser.add_argument("--max-pages", type=int, default=60)
    parser.add_argument("--delay", type=float, default=1.0, help="пауза между запросами, секунды")
    parser.add_argument("--ingest", action="store_true", help="загрузить в базу знаний")
    parser.add_argument("--store", type=Path, help="папка хранилища (по умолчанию knowledge_store/)")
    args = parser.parse_args(argv)

    out = args.out.expanduser()
    items = crawl(out, start=args.start, max_pages=args.max_pages, delay=args.delay, fetch=fetch)
    saved = [i for i in items if i.status == "saved"]
    print(f"Скачано: {len(saved)} (страниц и документов), папка {out}")
    for item in items:
        if item.status != "saved":
            print(f"Не скачано: {item.url} — {item.detail}")
    if not saved:
        print("Сайт ничего не отдал: проверьте, открывается ли он в браузере с этого компьютера.", file=sys.stderr)
        return 1
    if args.ingest:
        added, skipped = ingest(out, Store(args.store))
        print(f"В базе знаний: {len(added)} новых или обновлённых, пропущено: {len(skipped)}")
        for line in added:
            print(f"  + {line}")
        for line in skipped:
            print(f"  = {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
