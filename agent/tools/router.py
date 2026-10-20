"""Маршрутизатор инструментов: какие схемы модель видит на текущем шаге.

Запрос классифицируется по матрице тем (agent/topics.toml). Если тема определена уверенно,
модель сразу видит ядро и инструменты темы. Если нет (перефразировки, разговорные слова), модель
выбирает тему по каталогу в инструкции и открывает её или весь раздел через ``open_toolbox``.
Открытое остаётся открытым до конца сессии: уточняющий вопрос («а за август?») не содержит
слов темы, но относится к ней.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from anthropic import beta_async_tool

from ..classifier import PARAM_TITLES, Topic, classify, load_matrix, load_sections
from ..config import Mode
from .catalog import CORE, WRITE_TOOLS
from .common import _json

OPEN_TOOLBOX_DESCRIPTION = (
    "Открыть тему или раздел из каталога тем в инструкции, если нужного инструмента нет в списке. "
    "Бесплатно, без запросов к API. Возвращает инструменты и что спросить у пользователя."
)


class ToolRouter:
    """Выбирает видимые инструменты из полного набора; сам инструменты не выполняет."""

    def __init__(self, tools: Sequence[Any], mode: Mode) -> None:
        self._all = {tool.name: tool for tool in tools}
        unknown = {name for topic in load_matrix() for name in topic.tools} - self._all.keys()
        if unknown or set(CORE) - self._all.keys():
            raise ValueError(f"Матрица тем ссылается на несуществующие инструменты: {sorted(unknown)}")
        self._mode = mode
        self._topics = {topic.id: topic for topic in load_matrix()}
        self._sections = {section.id: section for section in load_sections()}
        self.open_tools: set[str] = set()
        self.open_toolbox = beta_async_tool(name="open_toolbox", description=OPEN_TOOLBOX_DESCRIPTION)(
            self._open_toolbox
        )

    def _open(self, topics: list[Topic]) -> list[str]:
        opened = [name for topic in topics for name in topic.tools if self._allowed(name)]
        opened = list(dict.fromkeys(opened))
        self.open_tools |= set(opened)
        return opened

    async def _open_toolbox(self, topic: str) -> str:
        """Открыть тему или раздел и получить их инструменты.

        Args:
            topic: Ключ темы (например, fuel_price) или раздела (например, spending) из каталога.
        """
        key = topic.strip().lower()
        if key in self._sections:
            members = [t for t in self._topics.values() if t.section == key and t.supported]
            return _json(
                {
                    "opened": key,
                    "title": self._sections[key].title,
                    "topics": [{"id": t.id, "title": t.title} for t in members],
                    "tools": self._open(members),
                }
            )
        found = self._topics.get(key)
        if found is None:
            return _json(
                {"error": "unknown_topic", "message": f"Нет темы или раздела {topic!r}.", "sections": sorted(self._sections)}
            )
        if not found.supported:
            return _json({"opened": key, "title": found.title, "tools": [], "reply": found.reply})
        return _json(
            {
                "opened": key,
                "title": found.title,
                "tools": self._open([found]),
                # Чего без вопроса пользователю не узнать и что подставлять по умолчанию.
                "ask_if_missing": [
                    {"param": PARAM_TITLES.get(p.kind, p.kind), "ask": p.ask} for p in found.params if p.required and p.ask
                ],
                "defaults": [
                    f"{PARAM_TITLES.get(p.kind, p.kind)}: прошлый месяц (даты в квадратных скобках)"
                    for p in found.params
                    if p.default == "previous_month"
                ],
            }
        )

    def _allowed(self, name: str) -> bool:
        return not (self._mode is Mode.READ_ONLY and name in WRITE_TOOLS)

    def preload(self, question: str) -> set[str]:
        """Открывает инструменты уверенно определённой темы вопроса; возвращает только вновь открытые."""
        found = classify(question)
        if not found.confident:
            return set()
        new = {name for name in found.tools if self._allowed(name)} - self.open_tools
        self.open_tools |= new
        return new

    def visible(self) -> list[Any]:
        """Инструменты, схемы которых отправляются модели на этом шаге."""
        wanted = set(CORE) | self.open_tools
        # Порядок как в полном наборе: схемы одинаково упорядочены между шагами.
        return [*(tool for name, tool in self._all.items() if name in wanted), self.open_toolbox]
