"""Загрузка документов в базу знаний: извлечение текста, нарезка, хранилище.

Поддержаны форматы, для которых не нужны внешние библиотеки или ключи: текст,
Markdown, HTML, CSV/TSV, JSON, Word (docx), Excel (xlsx), PowerPoint (pptx),
письма (eml) и PDF с текстовым слоем (через ``pdftotext``). Сканы, фото, аудио и
видео требуют распознавания (OCR, ASR) и проверки на персональные данные;
пока такой файл отклоняется понятной ошибкой, молча пропускать его нельзя.

Хранилище — папка из ``OPTI24_KNOWLEDGE_STORE`` (по умолчанию ``knowledge_store/``,
в git не попадает): ``documents.json`` с описанием документов и ``chunks.jsonl``
с фрагментами. Новая версия документа переводит старую в архив, поиск идёт только
по действующим версиям.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import zipfile
from dataclasses import asdict, dataclass
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

DEFAULT_STORE = Path(__file__).resolve().parent.parent / "knowledge_store"
KINDS = ("regulation", "tariff", "faq", "instruction", "norm", "other")
CLASSIFICATIONS = ("public", "internal")
CHUNK_CHARS = 900
MAX_FILE_BYTES = 30 * 1024 * 1024
MAX_UNPACKED_BYTES = 100 * 1024 * 1024  # защита от «бомб» внутри docx/xlsx/pptx

HINTS = {
    **dict.fromkeys(
        (".png", ".jpg", ".jpeg", ".webp", ".heic", ".tif", ".tiff", ".gif"),
        "изображение: нужно распознавание текста (OCR) и проверка на персональные данные",
    ),
    **dict.fromkeys(
        (".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4", ".mov", ".mkv"),
        "аудио и видео: нужна расшифровка речи (ASR) и проверка на персональные данные",
    ),
    **dict.fromkeys((".zip", ".7z", ".tar", ".gz"), "архив: распакуйте и загрузите файлы по одному"),
    **dict.fromkeys((".doc", ".xls", ".ppt", ".rtf"), "старый формат: пересохраните как docx, xlsx или pptx"),
}


class IngestError(Exception):
    """Файл нельзя загрузить; текст ошибки понятен человеку."""


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    title: str
    kind: str
    version: str
    valid_from: str
    classification: str
    status: str  # active | archived
    source: str
    sha256: str
    added: str
    chunks: int


# ---------- извлечение текста ----------


class _Html(HTMLParser):
    SKIP = {"script", "style", "nav", "header", "footer", "noscript"}
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP:
            self._skip += 1
        elif tag in {"h1", "h2", "h3", "h4"}:
            self.parts.append("\n\n## ")
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def _zip_xml(path: Path, names: list[str]) -> list[ElementTree.Element]:
    with zipfile.ZipFile(path) as archive:
        if sum(i.file_size for i in archive.infolist()) > MAX_UNPACKED_BYTES:
            raise IngestError("Файл слишком велик после распаковки.")
        return [ElementTree.fromstring(archive.read(n)) for n in names if n in archive.namelist()]


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _docx(path: Path) -> str:
    (root,) = _zip_xml(path, ["word/document.xml"]) or (None,)
    if root is None:
        raise IngestError("В файле нет word/document.xml: это не docx.")
    lines: list[str] = []
    for paragraph in root.iter():
        if _local(paragraph.tag) != "p":
            continue
        text = "".join(t.text or "" for t in paragraph.iter() if _local(t.tag) == "t")
        if text.strip():
            style = next((v for s in paragraph.iter() if _local(s.tag) == "pStyle" for v in s.attrib.values()), "")
            lines.append(f"## {text.strip()}" if str(style).lower().startswith(("heading", "заголовок")) else text)
    return "\n".join(lines)


def _xlsx(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    shared: list[str] = []
    for root in _zip_xml(path, ["xl/sharedStrings.xml"]):
        for item in root:
            shared.append("".join(t.text or "" for t in item.iter() if _local(t.tag) == "t"))
    sheets = sorted(n for n in names if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n))
    out: list[str] = []
    for number, sheet in enumerate(sheets, 1):
        (root,) = _zip_xml(path, [sheet])
        rows: list[list[str]] = []
        for row in root.iter():
            if _local(row.tag) != "row":
                continue
            cells: list[str] = []
            for cell in row:
                column = _column(cell.get("r", ""))
                if column is not None:
                    cells.extend([""] * (column - len(cells)))  # пропущенные пустые ячейки держат место
                value = next((v.text or "" for v in cell if _local(v.tag) == "v"), "")
                if cell.get("t") == "s" and value.isdigit() and int(value) < len(shared):
                    value = shared[int(value)]
                elif cell.get("t") == "inlineStr":
                    value = "".join(t.text or "" for t in cell.iter() if _local(t.tag) == "t")
                cells.append(value.strip())
            if any(cells):
                rows.append(cells)
        out.append(f"## Лист {number}\n" + _table(rows))
    return "\n\n".join(out)


def _column(ref: str) -> int | None:
    """Номер столбца (с нуля) по ссылке на ячейку вида «C2»."""
    letters = re.match(r"[A-Z]+", ref)
    if not letters:
        return None
    number = 0
    for char in letters.group():
        number = number * 26 + ord(char) - 64
    return number - 1


def _table(rows: list[list[str]]) -> str:
    """Строка таблицы вместе с шапкой столбцов: фрагмент понятен без остальных строк."""
    if not rows:
        return ""
    head, body = rows[0], rows[1:] or []
    if not body:
        return " | ".join(head)
    return "\n".join("; ".join(f"{h}: {v}" for h, v in zip(head, row, strict=False) if v) for row in body)


def _pptx(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        slides = sorted(
            (n for n in archive.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
            key=lambda n: int(re.findall(r"\d+", n)[-1]),
        )
    out = []
    for number, slide in enumerate(slides, 1):
        (root,) = _zip_xml(path, [slide])
        text = "\n".join(
            "".join(t.text or "" for t in p.iter() if _local(t.tag) == "t")
            for p in root.iter()
            if _local(p.tag) == "p"
        )
        out.append(f"## Слайд {number}\n{text.strip()}")
    return "\n\n".join(out)


def _pdf(path: Path) -> str:
    if shutil.which("pdftotext") is None:
        raise IngestError("Для PDF нужна утилита pdftotext (пакет poppler-utils).")
    result = subprocess.run(  # noqa: S603
        ["pdftotext", str(path), "-"], capture_output=True, text=True, timeout=120, check=False
    )
    if result.returncode != 0:
        raise IngestError("PDF не читается или защищён паролем.")
    if len(re.sub(r"\s+", "", result.stdout)) < 50:
        raise IngestError("В PDF нет текстового слоя (скан): нужно распознавание текста (OCR).")
    return result.stdout


def _csv(raw: str, delimiter: str | None) -> str:
    if delimiter is None:
        reader = csv.reader(io.StringIO(raw), csv.Sniffer().sniff(raw[:4096], delimiters=";,\t|"))
    else:
        reader = csv.reader(io.StringIO(raw), delimiter=delimiter)
    return _table([[c.strip() for c in row] for row in reader if any(row)])


def _eml(raw: bytes) -> str:
    from email import policy
    from email.parser import BytesParser

    message = BytesParser(policy=policy.default).parsebytes(raw)
    body = message.get_body(preferencelist=("plain", "html"))
    text = body.get_content() if body else ""
    if body and body.get_content_type() == "text/html":
        parser = _Html()
        parser.feed(text)
        text = "".join(parser.parts)
    return f"## {message['subject'] or 'Письмо'}\n{text}"


def _decode(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise IngestError("Не удалось определить кодировку текста (ожидается UTF-8 или Windows-1251).")


def extract_text(path: Path) -> str:
    """Текст документа с заголовками разделов в виде «## ...»."""
    suffix = path.suffix.lower()
    if suffix in HINTS:
        raise IngestError(f"Формат {suffix} пока не поддержан: {HINTS[suffix]}.")
    if not path.is_file():
        raise IngestError(f"Файл не найден: {path}")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise IngestError("Файл больше 30 МБ: разбейте его на части.")
    try:
        if suffix == ".docx":
            return _docx(path)
        if suffix == ".xlsx":
            return _xlsx(path)
        if suffix == ".pptx":
            return _pptx(path)
        if suffix == ".pdf":
            return _pdf(path)
        raw = path.read_bytes()
        if suffix == ".eml":
            return _eml(raw)
        if suffix in {".html", ".htm"}:
            parser = _Html()
            parser.feed(_decode(raw))
            return "".join(parser.parts)
        if suffix in {".csv", ".tsv"}:
            return _csv(_decode(raw), "\t" if suffix == ".tsv" else None)
        if suffix == ".json":
            return json.dumps(json.loads(_decode(raw)), ensure_ascii=False, indent=1)
        if suffix in {".txt", ".md", ".markdown", ".srt", ".vtt", ".log", ""}:
            return _decode(raw)
    except zipfile.BadZipFile as exc:
        raise IngestError(f"Файл повреждён или это не {suffix}.") from exc
    except (ElementTree.ParseError, json.JSONDecodeError, csv.Error) as exc:
        raise IngestError(f"Файл {suffix} не разобрать: {exc}") from exc
    raise IngestError(f"Формат {suffix} не поддержан.")


# ---------- нарезка ----------

_CLAUSE = re.compile(r"^\s*(\d+(?:\.\d+){0,3})\.?\s+(\S.*)$")
_HEADING = re.compile(r"^#{1,4}\s+(.+)$")


def split_chunks(text: str, doc_title: str) -> list[tuple[str, str]]:
    """Пары (заголовок, текст): по «## », по пунктам «3.2.», длинное делится по абзацам."""
    sections: list[tuple[str, list[str]]] = [(doc_title, [])]
    for line in text.replace("\r", "").splitlines():
        title_line = _HEADING.match(line)
        clause = _CLAUSE.match(line)
        if title_line:
            sections.append((title_line.group(1).strip()[:120], []))
        elif clause and ("." in clause.group(1) or len(line) > 12):
            sections.append((f"п. {clause.group(1)} {clause.group(2)[:60]}".strip(), [line.strip()]))
        else:
            sections[-1][1].append(line.strip())
    chunks: list[tuple[str, str]] = []
    for heading, lines in sections:
        paragraphs = [p for p in re.split(r"\n\s*\n|\n", "\n".join(lines)) if p.strip()]
        current = ""
        for paragraph in paragraphs:
            while len(paragraph) > CHUNK_CHARS:
                cut = paragraph.rfind(" ", 0, CHUNK_CHARS) or CHUNK_CHARS
                if current:
                    chunks.append((heading, current))
                    current = ""
                chunks.append((heading, paragraph[:cut].strip()))
                paragraph = paragraph[cut:].strip()
            if current and len(current) + len(paragraph) > CHUNK_CHARS:
                chunks.append((heading, current))
                current = ""
            current = f"{current}\n{paragraph}".strip()
        if current:
            chunks.append((heading, current))
    return [(h, t) for h, t in chunks if len(t) >= 20]


# ---------- хранилище ----------


class Store:
    def __init__(self, root: Path | None = None) -> None:
        env = os.environ.get("OPTI24_KNOWLEDGE_STORE")
        self.root = root or (Path(env) if env else DEFAULT_STORE)
        self.documents_path = self.root / "documents.json"
        self.chunks_path = self.root / "chunks.jsonl"

    def documents(self) -> list[Document]:
        if not self.documents_path.exists():
            return []
        return [Document(**d) for d in json.loads(self.documents_path.read_text(encoding="utf-8"))]

    def _save(self, documents: list[Document], chunks: list[dict[str, str]]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.documents_path.write_text(
            json.dumps([asdict(d) for d in documents], ensure_ascii=False, indent=1), encoding="utf-8"
        )
        self.chunks_path.write_text(
            "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks), encoding="utf-8"
        )

    def _chunks(self) -> list[dict[str, str]]:
        if not self.chunks_path.exists():
            return []
        return [json.loads(line) for line in self.chunks_path.read_text(encoding="utf-8").splitlines() if line]

    def add(
        self,
        path: Path,
        *,
        title: str,
        kind: str = "other",
        version: str = "",
        valid_from: str = "",
        classification: str = "internal",
        source: str = "",
    ) -> Document:
        if kind not in KINDS:
            raise IngestError(f"Раздел должен быть одним из: {', '.join(KINDS)}.")
        if classification not in CLASSIFICATIONS:
            raise IngestError(f"Гриф должен быть одним из: {', '.join(CLASSIFICATIONS)}.")
        if valid_from:
            try:
                date.fromisoformat(valid_from)
            except ValueError as exc:
                raise IngestError("Дата начала действия нужна в формате ГГГГ-ММ-ДД.") from exc
        text = extract_text(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        documents, chunks = self.documents(), self._chunks()
        if any(d.sha256 == digest and d.status == "active" for d in documents):
            raise IngestError("Этот файл уже загружен и действует.")
        pieces = split_chunks(text, title)
        if not pieces:
            raise IngestError("В файле не нашлось текста для базы знаний.")
        doc_id = hashlib.sha1(f"{title}|{kind}".encode()).hexdigest()[:10]
        previous = [d for d in documents if d.id == doc_id and d.status == "active"]
        if previous and previous[0].version == version and version:
            raise IngestError(f"Версия {version} документа «{title}» уже загружена.")
        # Версия с будущей датой начала не заменяет действующую: старая работает, пока новая не вступила в силу.
        if not valid_from or valid_from <= date.today().isoformat():
            documents = [
                Document(**{**asdict(d), "status": "archived"}) if d.id == doc_id and d.status == "active" else d
                for d in documents
            ]
            chunks = [{**c, "status": "archived"} if c["doc_id"] == doc_id else c for c in chunks]
        document = Document(
            id=doc_id, title=title, kind=kind, version=version, valid_from=valid_from,
            classification=classification, status="active", source=source or path.name,
            sha256=digest, added=date.today().isoformat(), chunks=len(pieces),
        )
        documents.append(document)
        chunks.extend(
            {"doc_id": doc_id, "version": version, "sha256": digest, "status": "active", "heading": h, "text": t} for h, t in pieces
        )
        self._save(documents, chunks)
        return document

    def archive(self, doc_id: str) -> int:
        documents, chunks = self.documents(), self._chunks()
        hit = [d for d in documents if d.id == doc_id and d.status == "active"]
        if not hit:
            raise IngestError(f"Действующего документа {doc_id} нет.")
        documents = [Document(**{**asdict(d), "status": "archived"}) if d in hit else d for d in documents]
        chunks = [{**c, "status": "archived"} if c["doc_id"] == doc_id else c for c in chunks]
        self._save(documents, chunks)
        return len(hit)

    def active_chunks(self, allow: tuple[str, ...] = CLASSIFICATIONS) -> list[tuple[Document, dict[str, str]]]:
        """Фрагменты версий, действующих сегодня; ``allow`` ограничивает грифы доступа."""
        today = date.today().isoformat()
        current: dict[str, Document] = {}
        for doc in self.documents():
            if doc.status != "active" or doc.classification not in allow or doc.valid_from > today:
                continue  # будущая версия не отвечает раньше срока
            best = current.get(doc.id)
            if best is None or doc.valid_from >= best.valid_from:
                current[doc.id] = doc
        return [
            (current[c["doc_id"]], c)
            for c in self._chunks()
            if c["status"] == "active" and c["doc_id"] in current and c.get("sha256") == current[c["doc_id"]].sha256
        ]
