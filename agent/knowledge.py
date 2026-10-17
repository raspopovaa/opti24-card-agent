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


def load_sections(directory: Path | None = None) -> list[Section]:
    """Разделы всех выжимок; README и служебные файлы не индексируются."""
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
    return sections


def search(sections: list[Section], query: str, top: int = 3) -> list[dict[str, str]]:
    """Лучшие разделы по вопросу; пустой список — в базе знаний ответа нет."""
    wanted = _stems(query)
    if not wanted:
        return []
    scored = []
    for section in sections:
        score = 3 * len(wanted & _stems(section.heading)) + len(wanted & _stems(section.text))
        if score >= MIN_SCORE:
            scored.append((score, section))
    scored.sort(key=lambda pair: -pair[0])
    return [
        {
            "source": f"docs/knowledge/{section.file}",
            "section": section.heading,
            "edition": section.edition,
            "text": section.text[:MAX_SECTION_CHARS],
        }
        for _, section in scored[:top]
    ]
