"""База знаний по договору топливных карт: поиск по выжимкам в docs/knowledge.

Выжимки — markdown-файлы с заголовком (frontmatter) и разделами «## ...».
Поиск простой и предсказуемый: слова вопроса сокращаются до основы (первые
пять букв, чтобы «блокировка» нашла «заблокировать» и «блокировки»), раздел
получает очки за каждую найденную основу, заголовок раздела весит втрое.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "docs" / "knowledge"
STEM = 5
MAX_SECTION_CHARS = 1500

MIN_SCORE = 2  # одно случайное слово («Москва») ещё не ответ
# Слова клиента, которых нет в тексте договора: «потерялась» ищет раздел про блокировку.
SYNONYMS = {  # ключи — основы по первым пяти буквам
    "потер": ("заблокировать", "утрата"),
    "украл": ("заблокировать", "утрата"),
    "краж": ("заблокировать", "утрата"),
    "украд": ("заблокировать", "утрата"),
    "верну": ("возврат",),
    "вернё": ("возврат",),
}
_WORD = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)
_STOP = {
    "как", "какой", "какая", "какие", "что", "где", "когда", "почему", "зачем", "для", "это",
    "нас", "наш", "наша", "наши", "мне", "меня", "или", "если", "при", "про", "есть", "будет",
    "можно", "нужно", "надо", "сколько", "который", "после", "чтобы", "все", "всё",
}


@dataclass(frozen=True, slots=True)
class Section:
    file: str
    title: str
    heading: str
    edition: str
    text: str
    source: str = ""  # путь или название для ответа; пусто — выжимка из docs/knowledge


def _stems(text: str) -> set[str]:
    return {
        word[:STEM] for word in _WORD.findall(text.casefold()) if len(word) >= 3 and word not in _STOP
    }


def _frontmatter(raw: str) -> tuple[dict[str, str], str]:
    if not raw.startswith("---\n"):
        return {}, raw
    head, _, body = raw[4:].partition("\n---\n")
    meta = {}
    for line in head.splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta, body


def load_store_sections(store_dir: Path | None = None) -> list[Section]:
    """Фрагменты загруженных документов (agent/ingest.py), только действующие версии."""
    from .ingest import Store

    store = Store(store_dir)
    # Агент работает от имени клиента: внутренние документы он отдаёт, только если оператор включил это явно
    # (OPTI24_KNOWLEDGE_INTERNAL=1). По умолчанию — только документы с грифом public.
    allow = ("public", "internal") if os.environ.get("OPTI24_KNOWLEDGE_INTERNAL", "0") == "1" else ("public",)
    return [
        Section(
            file=doc.id,
            title=doc.title,
            heading=chunk["heading"],
            edition=" ".join(x for x in (doc.version, f"с {doc.valid_from}" if doc.valid_from else "") if x),
            text=chunk["text"],
            source=f"{doc.title} ({doc.kind}, {doc.source})",
        )
        for doc, chunk in store.active_chunks(allow)
    ]


def load_sections(directory: Path | None = None) -> list[Section]:
    """Разделы всех выжимок и загруженных документов; README не индексируется."""
    root = directory or Path(os.environ.get("OPTI24_KNOWLEDGE_DIR") or DEFAULT_DIR)
    sections: list[Section] = []
    for path in sorted(root.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        meta, body = _frontmatter(path.read_text(encoding="utf-8"))
        for chunk in re.split(r"^## ", body, flags=re.MULTILINE)[1:]:
            heading, _, text = chunk.partition("\n")
            sections.append(
                Section(
                    file=path.name,
                    title=meta.get("title", path.stem),
                    heading=heading.strip(),
                    edition=meta.get("edition", ""),
                    text=text.strip(),
                )
            )
    if directory is None:  # явная папка в тестах и скриптах читается отдельно, без личного хранилища
        sections.extend(load_store_sections())
    return sections


def rank(sections: list[Section], query: str) -> list[tuple[int, Section]]:
    """Разделы с очками по основам слов, лучшие первыми; ниже MIN_SCORE не попадают."""
    wanted = _stems(query)
    for stem in list(wanted):
        for synonym in SYNONYMS.get(stem, ()):
            wanted |= _stems(synonym)
    if not wanted:
        return []
    scored = []
    for section in sections:
        matched = wanted & (_stems(section.heading) | _stems(section.text))
        score = 3 * len(wanted & _stems(section.heading)) + len(wanted & _stems(section.text))
        if score >= MIN_SCORE and (len(matched) >= 2 or len(wanted) == 1):
            scored.append((score, section))
    scored.sort(key=lambda pair: -pair[0])
    return scored


def as_result(section: Section) -> dict[str, str]:
    return {
        "source": section.source or f"docs/knowledge/{section.file}",
        "section": section.heading,
        "edition": section.edition,
        "text": section.text[:MAX_SECTION_CHARS],
    }


def search(sections: list[Section], query: str, top: int = 3) -> list[dict[str, str]]:
    """Лучшие разделы по вопросу; пустой список — в базе знаний ответа нет."""
    return [as_result(section) for _, section in rank(sections, query)[:top]]
