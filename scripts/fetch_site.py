"""Публичные страницы и документы сайта opti-24.ru в базу знаний агента.

Запускается там, где сайт открывается (облачные IP сайт может не пускать):

    python scripts/fetch_site.py --out ~/Downloads/opti24-site
    python scripts/fetch_site.py --out ~/Downloads/opti24-site --ingest

Скрипт обходит страницы www.opti-24.ru, начиная с главной. Ссылки на pdf, docx, xlsx и pptx
он скачивает и с cdn.opti-24.ru. Правила robots.txt соблюдаются. Личный кабинет, вход и поиск
не обходятся, новости — только с --with-news. Файлы и manifest.json кладутся в --out.

С --ingest файлы загружаются в knowledge_store/ с грифом public. Повторный запуск обновляет
изменившиеся документы, а неизменные пропускает. С --prune документы сайта, которых больше нет
среди скачанных, уходят в архив. Если вместо документа сайт отдал HTML (такой
файл отдаётся только браузеру), скрипт перечислит эти ссылки: скачайте их вручную и загрузите
через scripts/ingest_knowledge.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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

from agent.ingest import MAX_FILE_BYTES, Document, IngestError, Store  # noqa: E402

START_URL = "https://www.opti-24.ru/"
PAGE_HOSTS = {"www.opti-24.ru", "opti-24.ru"}
FILE_HOSTS = PAGE_HOSTS | {"cdn.opti-24.ru"}
DOCUMENT_TYPES = (".pdf", ".docx", ".xlsx", ".pptx")
SKIPPED_TYPES = (".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".css", ".js", ".zip", ".mp4", ".doc", ".xls")
PRIVATE_PATHS = ("/lk", "/login", "/auth", "/cabinet", "/personal", "/bitrix", "/search", "/ajax", "/upload/iblock")
NEWS_PATHS = ("/news",)
SITE_PREFIXES = tuple(f"https://{host}/" for host in sorted(FILE_HOSTS))
_LINK_NOISE = (
    re.compile(r"^\s*\d{2}\.\d{2}\.\d{4}\s*[\d.,]+\s*(?:kb|mb|кб|мб)\s*", re.IGNORECASE),
    re.compile(r"\s*скачать\s*$", re.IGNORECASE),
    re.compile(r"\.(?:pdf|docx|xlsx|pptx)\s*$", re.IGNORECASE),
)
USER_AGENT = "opti24-card-agent knowledge sync/1.0"
DOWNLOAD_SECONDS = 60

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
        elif tag == "title" and not self.title:  # только заголовок страницы, не <title> иконок SVG
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
    """Статус, тип содержимого и тело ответа; 0 — сайт не ответил или файл качался дольше минуты."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    print(f"→ {url}", file=sys.stderr, flush=True)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310  # nosec B310
            # timeout ограничивает одно чтение; медленную отдачу по кускам ограничивает общий срок.
            deadline, body = time.monotonic() + DOWNLOAD_SECONDS, bytearray()
            while chunk := response.read(64 * 1024):
                body += chunk
                if len(body) > MAX_FILE_BYTES or time.monotonic() > deadline:
                    raise TimeoutError(f"файл не скачался за {DOWNLOAD_SECONDS} с или больше 30 МБ")
            return response.status, response.headers.get_content_type(), bytes(body)
    except urllib.error.HTTPError as exc:
        return exc.code, "", b""
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"Нет ответа: {url} ({exc})", file=sys.stderr)
        return 0, "", b""


def guess_kind(url: str, title: str) -> str:
    words = re.findall(r"[а-яёa-z]+", f"{url} {title}".casefold())
    for kind, stems in (
        ("tariff", ("тариф", "price", "prices", "стоимост")),
        ("faq", ("faq", "вопрос")),
        ("instruction", ("инструкц", "instruction", "памятк")),
        ("regulation", ("договор", "dogovor", "оферт", "oferta", "регламент", "правил", "rules")),
    ):
        if any(word.startswith(stems) for word in words):
            return kind
    return "other"


def clean_link_text(text: str) -> str:
    """«15.11.2023 17.4 Kb Доверенность.docx Скачать» → «Доверенность»."""
    for pattern in _LINK_NOISE:
        text = pattern.sub("", text)
    return text.strip()


def _normalize(url: str, page_host: str = "") -> str:
    """Один адрес на страницу: https, без якоря; у страниц без запроса, с «/» в конце и одним хостом."""
    parts = urlparse(urldefrag(url)[0])._replace(scheme="https")
    if parts.hostname in PAGE_HOSTS and not _is_document(parts.geturl()):
        last = parts.path.rsplit("/", 1)[-1]
        path = parts.path if parts.path.endswith("/") or "." in last else parts.path + "/"
        parts = parts._replace(netloc=page_host or parts.netloc, path=path, query="")
    return parts.geturl()


def _is_document(url: str) -> bool:
    return urlparse(url).path.casefold().endswith(DOCUMENT_TYPES)


def _wanted(url: str, with_news: bool = False) -> bool:
    parts = urlparse(url)
    path = parts.path.casefold()
    if parts.scheme not in {"http", "https"} or path.endswith(SKIPPED_TYPES):
        return False
    if _is_document(url):
        return parts.hostname in FILE_HOSTS
    skipped = PRIVATE_PATHS if with_news else PRIVATE_PATHS + NEWS_PATHS
    return parts.hostname in PAGE_HOSTS and not path.startswith(skipped)


def _robots(start: str, fetch: Fetch) -> urllib.robotparser.RobotFileParser:
    robots = urllib.robotparser.RobotFileParser()
    status, _, body = fetch(urljoin(start, "/robots.txt"))
    robots.parse(body.decode("utf-8", "replace").splitlines() if status == 200 else [])
    return robots


def crawl(out: Path, *, start: str = START_URL, max_pages: int = 60, delay: float = 1.0,
          with_news: bool = False, fetch: Fetch = http_fetch) -> list[Item]:
    """Скачивает страницы и документы в ``out`` и пишет manifest.json."""
    out.mkdir(parents=True, exist_ok=True)
    robots = _robots(start, fetch)
    host = urlparse(start).netloc
    queue: list[tuple[str, str]] = [(_normalize(start, host), "")]
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
            title = clean_link_text(link_text) or Path(urlparse(url).path).stem
            file = name + Path(urlparse(url).path).suffix.casefold()
        else:
            pages += 1
            parser = _Links()
            parser.feed(body.decode("utf-8", "replace"))
            title = " ".join(parser.title.split()) or urlparse(url).path
            file = name + ".html"
            for href, text in parser.links:
                link = _normalize(urljoin(url, href), host)
                if link not in seen and _wanted(link, with_news):
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


def ingest(out: Path, store: Store, *, prune: bool = False) -> tuple[list[str], list[str], list[str]]:
    """Загружает сохранённые файлы с грифом public; возвращает (загружено, пропущено, в архиве).

    Документ того же URL с другим названием или разделом уходит в архив, чтобы не было двух версий.
    С ``prune`` в архив уходят и документы сайта, которых нет среди скачанных сейчас.
    """
    items = [Item(**i) for i in json.loads((out / "manifest.json").read_text(encoding="utf-8"))]
    saved = [i for i in items if i.status == "saved"]
    added, skipped, archived = [], [], []
    for item in saved:
        path = out / item.file
        version = f"{date.today().isoformat()}-{hashlib.sha256(path.read_bytes()).hexdigest()[:6]}"
        try:
            doc = store.add(
                path, title=f"opti-24.ru: {item.title}", kind=item.kind,
                version=version, classification="public", source=item.url,
            )
        except IngestError as exc:
            skipped.append(f"{item.url}: {exc}")
            continue
        added.append(f"{doc.title}, фрагментов: {doc.chunks}")
        archived += _archive(store, lambda d, url=item.url, new=doc.id: d.source == url and d.id != new)
    if prune:
        current = {i.url for i in saved}
        archived += _archive(store, lambda d: d.source.startswith(SITE_PREFIXES) and d.source not in current)
    return added, skipped, archived


def _archive(store: Store, stale: Callable[[Document], bool]) -> list[str]:
    stale_docs = {d.id: d for d in store.documents() if d.status == "active" and stale(d)}
    for doc_id in stale_docs:
        store.archive(doc_id)
    return [f"{d.title} ({d.source})" for d in stale_docs.values()]


def main(argv: list[str] | None = None, fetch: Fetch = http_fetch) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--out", type=Path, required=True, help="папка для скачанных файлов")
    parser.add_argument("--start", default=START_URL)
    parser.add_argument("--max-pages", type=int, default=60)
    parser.add_argument("--delay", type=float, default=1.0, help="пауза между запросами, секунды")
    parser.add_argument("--with-news", action="store_true", help="обходить и раздел новостей")
    parser.add_argument("--ingest", action="store_true", help="загрузить в базу знаний")
    parser.add_argument("--prune", action="store_true", help="в архив документы сайта, которых нет среди скачанных")
    parser.add_argument("--store", type=Path, help="папка хранилища (по умолчанию knowledge_store/)")
    args = parser.parse_args(argv)

    out = args.out.expanduser()
    items = crawl(out, start=args.start, max_pages=args.max_pages, delay=args.delay,
                  with_news=args.with_news, fetch=fetch)
    saved = [i for i in items if i.status == "saved"]
    print(f"Скачано: {len(saved)} (страниц и документов), папка {out}")
    for item in items:
        if item.status != "saved":
            print(f"Не скачано: {item.url} — {item.detail}")
    if not saved:
        print("Сайт ничего не отдал: проверьте, открывается ли он в браузере с этого компьютера.", file=sys.stderr)
        return 1
    if args.ingest:
        added, skipped, archived = ingest(out, Store(args.store), prune=args.prune)
        print(f"В базе знаний: {len(added)} новых или обновлённых, пропущено: {len(skipped)}, в архив: {len(archived)}")
        for prefix, lines in (("+", added), ("=", skipped), ("-", archived)):
            for line in lines:
                print(f"  {prefix} {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
