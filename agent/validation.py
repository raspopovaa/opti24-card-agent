"""Проверка ответа агента кодом, без второй модели.

Модель пишет текст по данным инструментов, и при пересказе может исказить цифру или пропустить
то, что инструкция требует назвать. Две проверки ловят это до того, как ответ увидит человек:

* числа: каждая сумма, объём, цена, процент или крупное число из ответа должно найтись в данных,
  которые инструменты вернули в этой сессии (с учётом округления, «тыс.», «млн» и процентов);
* обязательные элементы по теме (``answer_must`` в agent/topics.toml): период для расходов,
  оговорка «причина вероятная» для отказа на АЗС, пункт договора для условий и т. п.

Проверка консервативна: число, которого нет в данных, помечается, даже если модель его верно
вычислила сама. Инструкция требует считать кодом, поэтому такое число и стоит перепроверить.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from .classifier import ANSWER_TITLES, DETECTORS, Topic, load_matrix

# Число с группами разрядов через пробел («11 600,00») или без них; дробная часть через , или .
# Разряды разделяют обычный, неразрывный, узкий, тонкий и цифровой пробелы.
_GROUP_SEP = r"[ \u00a0\u202f\u2009\u2007]"
_NUMBER = re.compile(r"(?<![\w*№])(\d{1,3}(?:" + _GROUP_SEP + r"\d{3})+|\d+)(?:[.,](\d+))?(?![\w.,]\d)")
_UNIT = re.compile(r"\s*(тыс|млн|₽|руб|р\.|л\b|литр|%)", re.IGNORECASE)
_SCALE = {"тыс": Decimal(1000), "млн": Decimal(1_000_000)}
_APPROX = re.compile(r"(около|примерно|порядка|почти|≈|~)\s*$", re.IGNORECASE)
# Даты и время — не величины: 2026-09-30, 01.09.2026, 01.09, 14:35.
_DATE_OR_TIME = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}\.\d{1,2}\.\d{2,4}\b|\b(0?[1-9]|[12]\d|3[01])\.(0[1-9]|1[0-2])\b(?!\s*(₽|руб|р\.|л\b|%))"
    r"|\b\d{1,2}:\d{2}\b"
)
_MAX_SMALL_COUNT = 31  # «3 карты», «2 заправки», дни месяца: без единиц такие числа не проверяем
_MIN_IDENTIFIER_DIGITS = 6  # номера карт, договоров, ID — не величины


@dataclass(frozen=True, slots=True)
class _Claim:
    text: str
    value: Decimal
    tolerance: Decimal
    percent: bool


def _decimal(whole: str, fraction: str | None) -> Decimal | None:
    digits = re.sub(r"\D", "", whole) + (f".{fraction}" if fraction else "")
    try:
        return Decimal(digits)
    except InvalidOperation:
        return None


def numbers_in(text: str) -> list[Decimal]:
    """Все числа из текста или JSON-строки: для данных инструментов, без фильтров значимости."""
    values: list[Decimal] = []
    for match in _NUMBER.finditer(text):
        value = _decimal(match.group(1), match.group(2))
        if value is not None:
            values.append(value)
        # «8 240» может быть и одним числом, и двумя: в данных берём оба прочтения.
        if re.search(_GROUP_SEP, match.group(1)):
            values += [Decimal(part) for part in re.split(_GROUP_SEP, match.group(1))]
    return values


def _claims(answer: str) -> list[_Claim]:
    """Величины из ответа, которые стоит сверить с данными."""
    # Маскированные номера карт («****2323») — не величины; остальную разметку (**жирный**) убираем,
    # иначе знак перед числом мешает его распознать.
    text = re.sub(r"\*{3,}\d+", " ", answer)  # маска карты «****2323»; «**123» — жирный шрифт
    text = _DATE_OR_TIME.sub(" ", text.replace("*", " ").replace("`", " "))
    claims = []
    for match in _NUMBER.finditer(text):
        whole, fraction = match.group(1), match.group(2)
        value = _decimal(whole, fraction)
        if value is None:
            continue
        unit_match = _UNIT.match(text, match.end())
        unit = unit_match.group(1).lower() if unit_match else ""
        plain_digits = re.sub(r"\D", "", whole)
        if not unit and not fraction and not re.search(_GROUP_SEP, whole) and len(plain_digits) >= _MIN_IDENTIFIER_DIGITS:
            continue  # номер карты, договора или операции
        if not unit and not fraction and (value <= _MAX_SMALL_COUNT or 2000 <= value <= 2100):
            continue  # счётчики, дни, годы
        scale = _SCALE.get(unit, Decimal(1))
        tolerance = Decimal("0.5") * Decimal(10) ** -len(fraction or "") * scale
        if _APPROX.search(text[max(0, match.start() - 12) : match.start()]):
            tolerance = max(tolerance, value * scale / 100)
        shown = text[match.start() : unit_match.end() if unit_match else match.end()].strip()
        claims.append(_Claim(shown, value * scale, tolerance + Decimal("1e-9"), unit == "%"))
    return claims


class Evidence:
    """Числа из данных, которые агент получил в сессии: ответы инструментов, вопросы, инструкция."""

    def __init__(self) -> None:
        self._values: set[Decimal] = set()

    def add(self, payload: object) -> None:
        if not isinstance(payload, str):
            self._walk(payload)
            return
        try:
            parsed = json.loads(payload)
        except ValueError:  # не JSON: текст инструкции, вопроса или ошибки
            self._values.update(abs(v) for v in numbers_in(payload))
            return
        self._walk(parsed)

    def _walk(self, node: Any) -> None:
        if isinstance(node, bool) or node is None:
            return
        if isinstance(node, int | float):
            self._values.add(abs(Decimal(str(node))))
        elif isinstance(node, str):
            self._values.update(abs(v) for v in numbers_in(node))
        elif isinstance(node, dict):
            for key, value in node.items():
                self._walk(key)
                self._walk(value)
        elif isinstance(node, list | tuple):
            for value in node:
                self._walk(value)

    def supports(self, claim: _Claim) -> bool:
        if any(abs(value - claim.value) <= claim.tolerance for value in self._values):
            return True
        if claim.percent:  # 3,75 % в данных бывает долей: 0.0375
            share, tolerance = claim.value / 100, claim.tolerance / 100
            return any(abs(value - share) <= tolerance for value in self._values)
        return False


def _matches(pattern: str) -> Callable[[str, str], bool]:
    return lambda text, _: bool(re.search(pattern, text))


# Обязательные элементы ответа: признак в нормализованном тексте ответа. Названия — в classifier,
# чтобы подсказка модели заранее говорила, что назвать.
ANSWER_CHECKS: dict[str, Callable[[str, str], bool]] = {
    "period": DETECTORS["period"],
    "probable": _matches(r"вероятн|возможн|скорее всего|может быть|могл"),
    "heuristic": _matches(r"требу\w* проверк|провер(ить|ьте)|эвристик|не доказ"),
    "clause": _matches(r"пункт|п\.\s*\d|раздел|приложени"),
    "typical": _matches(r"типов|могут отличаться|сверь|сверить"),
}

# Ответ-вопрос или сообщение о неудаче: требования к содержанию к нему не применяются.
# Ответ «данных нет» или «ничего не найдено»: для него требования к содержанию не применяются.
_NO_RESULT = re.compile(
    r"не наш|не найден|нет данных|не удалось|ошибк|недоступ|нет доступа|не было|не обнаружен|"
    r"не проводил|нет операций|\b0 (заправок|операций|карт)"
)


@dataclass(frozen=True, slots=True)
class AnswerCheck:
    unverified: tuple[str, ...] = ()  # числа так, как они написаны в ответе
    missing: tuple[str, ...] = ()  # названия обязательных элементов, которых нет в ответе

    @property
    def ok(self) -> bool:
        return not self.unverified and not self.missing

    def problems(self) -> list[str]:
        """Для evals: каждая находка — отдельная строка."""
        found = [f"число не из данных: {text}" for text in self.unverified]
        return found + [f"в ответе нет: {title}" for title in self.missing]

    def note(self) -> str:
        """Предупреждение для человека; пустое, если всё в порядке."""
        parts = []
        if self.unverified:
            parts.append(f"чисел {', '.join(self.unverified)} нет в данных, полученных от API, — перепроверьте их")
        if self.missing:
            parts.append(f"в ответе не названо: {', '.join(self.missing)}")
        return "Проверка ответа: " + "; ".join(parts) + "." if parts else ""


def required_elements(called_tools: Iterable[str], topics: tuple[Topic, ...] | None = None) -> list[str]:
    """Обязательные элементы тем, главный инструмент которых был вызван в этом вопросе."""
    called = set(called_tools)
    kinds: list[str] = []
    for topic in topics or load_matrix():
        if topic.tools and topic.tools[0] in called:
            kinds += [kind for kind in topic.answer_must if kind not in kinds]
    return kinds


def _normalize(text: str) -> str:
    return text.casefold().replace("ё", "е")


def _asks_or_fails(answer: str) -> bool:
    last = answer.strip().rsplit("\n", 1)[-1]
    return last.rstrip().endswith("?") or bool(_NO_RESULT.search(_normalize(answer)))


def check_answer(
    answer: str, evidence: Evidence, called_tools: Iterable[str], topics: tuple[Topic, ...] | None = None
) -> AnswerCheck:
    """Сверяет ответ с данными сессии и с обязательными элементами тем вызванных инструментов."""
    unverified = tuple(dict.fromkeys(c.text for c in _claims(answer) if not evidence.supports(c)))
    missing: tuple[str, ...] = ()
    if not _asks_or_fails(answer):
        text = _normalize(answer)
        missing = tuple(
            ANSWER_TITLES[kind]
            for kind in required_elements(called_tools, topics)
            if not ANSWER_CHECKS[kind](text, answer)
        )
    return AnswerCheck(unverified, missing)
