"""Классификатор запросов: запрос клиента → тема из матрицы (agent/topics.toml) → инструменты.

Два пути к теме. Быстрый: код сравнивает вопрос с ключевыми словами тем и, если совпадение
уверенное, сразу даёт модели подсказку (тема, инструменты, чего не хватает). Смысловой: если
слова не дали уверенной темы (перефразировки, разговорные слова), тему выбирает модель по
каталогу тем в инструкции и открывает её через open_toolbox. Неуверенной подсказки нет совсем:
неверная подсказка хуже никакой, она уводит модель не к тем инструментам.
"""

from __future__ import annotations

import json
import os
import re
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from functools import cache
from pathlib import Path
from typing import Any

MATRIX_PATH = Path(__file__).with_name("topics.toml")
MAX_TOPICS = 2  # инструменты берутся из двух лучших тем: вопрос часто затрагивает две
MIN_CONFIDENT_SCORE = 2  # одно многозначное слово («счёт», «заправка») — ещё не тема

_WORD = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)

# Признаки параметров в тексте вопроса. Ошибка в сторону «нашли» безопасна: модель всё равно
# увидит вопрос целиком; ошибка в сторону «не нашли» лишь добавит уточняющий вопрос.
_MONTHS = r"январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр"
_CAPITALIZED = re.compile(r"[А-ЯЁ][а-яё]{2,}")


def _names_inside_sentence(original: str) -> bool:
    """Есть ли слово с заглавной буквы не в начале предложения: имя водителя, марка машины."""
    for sentence in re.split(r"[.!?]+", original):
        words = sentence.split()
        if any(_CAPITALIZED.fullmatch(word.strip(",;:«»\"()")) for word in words[1:]):
            return True
    return False


def _period(text: str, _: str) -> bool:
    return bool(
        re.search(r"\d{4}-\d{2}-\d{2}|\b\d{1,2}\.\d{1,2}(\.\d{2,4})?\b|\b20\d{2}\b", text)
        or re.search(rf"\b({_MONTHS})|прошл|текущ|этот месяц|этом месяце|недел|вчера|сегодня|квартал", text)
    )


def _card(text: str, original: str) -> bool:
    return bool(re.search(r"\b\d{6,19}\b|\bc\d+\b|все карты", text)) or _names_inside_sentence(original)


DETECTORS: dict[str, Callable[[str, str], bool]] = {
    "period": _period,
    "card": _card,
    "product": lambda text, _: bool(re.search(r"\bдт\b|дизел|\bаи[- ]?\d{2}|бензин|\bгаз\b|\bсуг\b|пропан", text)),
    "station": lambda text, _: bool(re.search(r"азс\s*№?\s*\d+", text)),
    "liters": lambda text, _: bool(re.search(r"\d+[.,]?\d*\s*(л\b|литр)", text)),
    "decline_code": lambda text, _: bool(re.search(r"код\w*\s+(ответа\s+)?[a-zа-я0-9]{1,4}\b", text)),
    "transaction_id": lambda text, _: bool(re.search(r"(операци|транзакци)\w*\s+№?\s*\d+", text)),
    "reason": lambda text, _: bool(re.search(r"уволи|потер|утер|украл|краж|причин|сломал", text)),
    "amount": lambda text, _: bool(re.search(r"\d+\s*(л\b|литр|руб|₽|р\b)", text)),
    "region": lambda text, _: bool(re.search(r"област|край|республик|москв|петербург|регион", text)),
}

PARAM_TITLES = {
    "period": "период",
    "card": "карта",
    "product": "вид топлива",
    "station": "АЗС",
    "liters": "литры",
    "decline_code": "код ответа из чека",
    "transaction_id": "номер операции",
    "reason": "причина",
    "amount": "размер",
    "region": "регион",
}


@dataclass(frozen=True, slots=True)
class Param:
    kind: str
    required: bool = False
    default: str | None = None
    ask: str | None = None


@dataclass(frozen=True, slots=True)
class Section:
    id: str
    title: str
    summary: str


@dataclass(frozen=True, slots=True)
class Topic:
    id: str
    title: str
    goal: str
    section: str
    examples: tuple[str, ...]
    keywords: tuple[str, ...]
    tools: tuple[str, ...]
    answer_fields: tuple[str, ...]
    params: tuple[Param, ...] = ()
    dictionaries: tuple[str, ...] = ()
    notes: str = ""
    supported: bool = True
    reply: str = ""


@dataclass(frozen=True, slots=True)
class Classification:
    topics: tuple[Topic, ...]  # главная тема первой, не больше MAX_TOPICS
    missing: tuple[Param, ...] = field(default=())  # обязательные параметры главной темы, которых нет
    # Уверенно ли определена тема: только тогда даётся подсказка и заранее открываются инструменты.
    confident: bool = False

    @property
    def primary(self) -> Topic | None:
        return self.topics[0] if self.topics else None

    @property
    def tools(self) -> list[str]:
        names: list[str] = []
        for topic in self.topics:
            names += [name for name in topic.tools if name not in names]
        return names


@cache
def _matrix_data(path: Path = MATRIX_PATH) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


@cache
def load_sections(path: Path = MATRIX_PATH) -> tuple[Section, ...]:
    return tuple(Section(**row) for row in _matrix_data(path)["section"])


@cache
def load_matrix(path: Path = MATRIX_PATH) -> tuple[Topic, ...]:
    data = _matrix_data(path)
    topics = []
    for row in data["topic"]:
        topics.append(
            Topic(
                id=row["id"],
                title=row["title"],
                goal=row["goal"],
                section=row["section"],
                examples=tuple(row.get("examples", ())),
                keywords=tuple(row.get("keywords", ())),
                tools=tuple(row.get("tools", ())),
                answer_fields=tuple(row.get("answer_fields", ())),
                dictionaries=tuple(row.get("dictionaries", ())),
                params=tuple(Param(**param) for param in row.get("params", ())),
                notes=row.get("notes", ""),
                supported=row.get("supported", True),
                reply=row.get("reply", ""),
            )
        )
    return tuple(topics)


def _normalize(text: str) -> str:
    return text.casefold().replace("ё", "е")


def _score(topic: Topic, text: str, words: list[str]) -> int:
    total = 0
    for keyword in topic.keywords:
        strong = keyword.endswith("!")
        key = _normalize(keyword.rstrip("!"))
        if " " in key:
            hit, weight = key in text, 2
        else:
            hit, weight = any(word.startswith(key) for word in words), 1
        if hit:
            total += weight + int(strong)
    return total


def has_param(kind: str, question: str) -> bool:
    return DETECTORS[kind](_normalize(question), question)


def classify(question: str, matrix: tuple[Topic, ...] | None = None) -> Classification:
    """Тема вопроса (и вторая, если есть) и недостающие обязательные параметры главной темы."""
    topics = matrix or load_matrix()
    text = _normalize(question)
    words = _WORD.findall(text)
    scored = [(score, -index, topic) for index, topic in enumerate(topics) if (score := _score(topic, text, words))]
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    chosen = tuple(topic for _, _, topic in scored[:MAX_TOPICS])
    if not chosen:
        return Classification(())
    top = scored[0][0]
    runner_up = scored[1][0] if len(scored) > 1 else 0
    confident = top >= MIN_CONFIDENT_SCORE and top > runner_up
    missing = tuple(
        param for param in chosen[0].params if param.required and not has_param(param.kind, question)
    )
    return Classification(chosen, missing, confident)


def hint(found: Classification) -> str:
    """Строка-подсказка для модели; пустая, если тема не определилась уверенно."""
    primary = found.primary
    if primary is None or not found.confident:
        return ""
    if not primary.supported:
        return f"[Подсказка: тема «{primary.title}» — запрос вне возможностей агента. {primary.reply} Инструменты не вызывай.]"
    parts = [f"тема «{primary.title}» — {primary.goal[:1].lower()}{primary.goal[1:]}"]
    if len(found.topics) > 1:
        parts.append(f"также тема «{found.topics[1].title}»")
    if found.tools:
        parts.append(f"сначала инструменты: {', '.join(found.tools)}")
    for param in found.missing:
        title = PARAM_TITLES.get(param.kind, param.kind)
        if param.default == "current_month":
            parts.append(f"{title} не указан — возьми текущий месяц с 1-го числа по сегодня и назови его в ответе")
        elif param.ask:
            parts.append(f"не указан параметр «{title}» — сначала спроси: «{param.ask}», платные инструменты до ответа не вызывай")
    return "[Подсказка: " + "; ".join(parts) + ".]"


def log_question(question: str, found: Classification, path: str | None = None) -> None:
    """Пишет вопрос и его темы в журнал, если он включён (AGENT_TOPIC_LOG).

    Журнал нужен, чтобы пополнять матрицу тем реальными вопросами. Вопросы могут содержать
    имена и номера карт, поэтому журнал включается явно и хранится только локально.
    """
    target = path or os.environ.get("AGENT_TOPIC_LOG")
    if not target:
        return
    record = {
        "date": date.today().isoformat(),
        "question": question,
        "topics": [topic.id for topic in found.topics],
        "confident": found.confident,
        "missing": [param.kind for param in found.missing],
    }
    with Path(target).expanduser().open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def catalog_text() -> str:
    """Каталог разделов и тем для инструкции модели: по нему модель сама выбирает тему."""
    topics = load_matrix()
    lines = []
    for section in load_sections():
        members = [f"{t.id} — {t.title}" for t in topics if t.section == section.id]
        lines.append(f"- {section.title} ({section.id}): " + "; ".join(members) + ".")
    return "\n".join(lines)
