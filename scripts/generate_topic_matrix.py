"""Документ «Матрица тем» (docs/topic-matrix.md) из матрицы, кода инструментов и SDK.

Что берётся откуда, чтобы документ не расходился с кодом:
* темы, примеры, параметры и поля ответа — из agent/topics.toml;
* какие методы SDK вызывает инструмент — из кода agent/tools/*.py;
* тарифицируемый ли метод и есть ли он на DEMO — из метаданных SDK (OPERATION_METADATA_BY_NAME);
* что значит поле ответа — из описаний полей моделей ответа SDK.

    python scripts/generate_topic_matrix.py           # перезаписать документ
    python scripts/generate_topic_matrix.py --check   # код 1, если документ устарел (для CI)
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import pkgutil
import re
import sys
import typing
from functools import cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import apisdkopti24.services as sdk_services  # noqa: E402
from apisdkopti24.endpoints import OPERATION_METADATA_BY_NAME  # noqa: E402

from agent.classifier import ANSWER_TITLES, MIN_CONFIDENT_SCORE, PARAM_TITLES, Topic, load_matrix, load_sections  # noqa: E402
from agent.tools.reference import DICTIONARIES  # noqa: E402

DOC = ROOT / "docs" / "topic-matrix.md"

# Помощники контекста инструментов → методы SDK, которые они вызывают.
HELPER_OPERATIONS = {
    "transactions(": ("get_transactions_v2", "get_card_transactions_v2"),
    "station_info(": ("get_azs_list_v1",),
    "purchase_check(": ("check_purchase", "get_dictionary"),
    "regions_for(": ("get_azs_list_v1", "get_dictionary"),
    "product_label(": ("get_dictionary",),
    "goods_label(": ("get_dictionary",),
    "dictionary_label(": ("get_dictionary",),
    "dictionary(": ("get_dictionary",),
    "limits_payload(": ("get_dictionary",),
}


@cache
def tool_operations() -> dict[str, tuple[str, ...]]:
    """Инструмент → методы SDK по коду agent/tools/*.py (вызовы client.<сервис>.<метод> и помощники)."""
    result: dict[str, tuple[str, ...]] = {}
    for path in sorted((ROOT / "agent" / "tools").glob("*.py")):
        for block in path.read_text(encoding="utf-8").split("@beta_async_tool")[1:]:
            name = re.search(r"async def (\w+)\(", block)
            if name is None:
                continue
            ops = [m.group(1) for m in re.finditer(r"client\.\w+\.(\w+)\(", block)]
            ops = [op if op in OPERATION_METADATA_BY_NAME else op.removesuffix("_to") for op in ops]
            for helper, helper_ops in HELPER_OPERATIONS.items():
                if re.search(rf"(?<![\w.]){re.escape(helper)}", block):
                    ops += helper_ops
            result[name.group(1)] = tuple(dict.fromkeys(ops))
    return result


@cache
def _service_methods() -> dict[str, Any]:
    methods: dict[str, Any] = {}
    for module_info in pkgutil.iter_modules(sdk_services.__path__):
        module = importlib.import_module(f"apisdkopti24.services.{module_info.name}")
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__:
                continue
            for name, fn in inspect.getmembers(cls, inspect.iscoroutinefunction):
                if name in OPERATION_METADATA_BY_NAME:
                    methods.setdefault(name, fn)
    return methods


def _walk(model: Any, prefix: str = "", depth: int = 0) -> dict[str, str]:
    fields: dict[str, str] = {}
    for field_name, field in getattr(model, "model_fields", {}).items():
        path = prefix + field_name
        fields[path] = (field.description or "").strip()
        stack, seen = [field.annotation], set()
        while stack:
            annotation = stack.pop()
            if id(annotation) in seen:
                continue
            seen.add(id(annotation))
            if hasattr(annotation, "model_fields") and depth < 5:
                fields.update(_walk(annotation, path + ".", depth + 1))
            stack += list(typing.get_args(annotation))
    return fields


@cache
def response_fields(operation: str) -> dict[str, str]:
    """Пути полей модели ответа операции SDK и их описания."""
    return _walk(typing.get_type_hints(_service_methods()[operation])["return"])


def _operation_label(operation: str) -> str:
    meta = OPERATION_METADATA_BY_NAME[operation]
    marks = ["тарифицируемый" if meta["billable"] else "бесплатный"]
    if not meta["demo_available"]:
        marks.append("только рабочий API")
    return f"`{operation}` ({', '.join(marks)})"


def _params(topic: Topic) -> str:
    parts = []
    for param in topic.params:
        title = PARAM_TITLES.get(param.kind, param.kind)
        if param.default == "previous_month":
            parts.append(f"{title} — по умолчанию прошлый месяц")
        elif param.required and param.ask:
            parts.append(f"{title} — спросить: «{param.ask}»")
        elif param.required:
            parts.append(f"{title} — обязательно")
        else:
            parts.append(f"{title} — если есть")
    return "; ".join(parts) or "—"


def _topic_operations(topic: Topic) -> list[str]:
    ops: list[str] = []
    for tool in topic.tools:
        ops += [op for op in tool_operations().get(tool, ()) if op not in ops]
    return ops


def render() -> str:
    topics = load_matrix()
    supported = [t for t in topics if t.supported]
    lines = [
        "# Матрица тем",
        "",
        "Агент относит запрос клиента к теме и показывает модели только инструменты этой темы.",
        "Документ собирается из `agent/topics.toml`, кода инструментов и SDK `apisdkopti24`",
        "скриптом `python scripts/generate_topic_matrix.py`: не правьте его вручную.",
        "",
        "Как это работает:",
        "",
        "1. Быстрый путь. Классификатор (`agent/classifier.py`) сравнивает вопрос с ключевыми словами тем.",
        f"   Если главная тема набрала не меньше {MIN_CONFIDENT_SCORE} и строго больше следующей, она считается",
        "   уверенной: модель получает подсказку (тема, цель, инструменты, чего не хватает), а у",
        "   OpenAI-совместимых провайдеров сразу видит инструменты темы.",
        "2. Смысловой путь. Если уверенности нет (перефразировки, разговорные слова, ничья двух тем),",
        "   подсказки нет: модель сама выбирает тему по каталогу разделов и тем в инструкции и открывает",
        "   её или весь раздел вызовом `open_toolbox`, который скажет, что спросить у пользователя.",
        "3. Без обязательного параметра (например, карты для разбора отказа) модель сначала спрашивает",
        "   пользователя и не тратит тарифицируемые запросы. У Claude видны все инструменты, каталог тем — в инструкции.",
        "",
        "Как пополнять: добавьте реальный вопрос клиента в `examples` подходящей темы (или заведите",
        "тему), допишите `keywords`, запустите `pytest` и этот скрипт. Реальные вопросы можно собирать",
        "журналом `AGENT_TOPIC_LOG=путь.jsonl`: он включается явно и хранится только у вас.",
        "",
        f"Тем: {len(topics)}, из них {len(supported)} агент выполняет, {len(topics) - len(supported)} — вне его задач.",
        "",
        "## Сводная таблица",
        "",
        "| Раздел | Тема | Цель | Инструменты | Методы SDK | Что спросить |",
        "|---|---|---|---|---|---|",
    ]
    section_titles = {section.id: section.title for section in load_sections()}
    for topic in supported:
        tools = ", ".join(f"`{name}`" for name in topic.tools) or "—"
        ops = ", ".join(_operation_label(op) for op in _topic_operations(topic)) or "без API"
        lines.append(
            f"| {section_titles[topic.section]} | **{topic.title}** (`{topic.id}`) | {topic.goal} | {tools} | {ops} | {_params(topic)} |"
        )
    lines += ["", "Вне задач агента:", ""]
    lines += [f"- **{t.title}** (`{t.id}`): {t.reply}" for t in topics if not t.supported]
    lines += ["", "## Темы подробно", ""]
    for topic in supported:
        lines += [f"### {topic.title} (`{topic.id}`)", "", topic.goal, ""]
        lines.append("Примеры запросов: " + "; ".join(f"«{example}»" for example in topic.examples) + ".")
        lines.append("")
        if topic.tools:
            lines.append("Инструменты и методы SDK:")
            lines.append("")
            for tool in topic.tools:
                ops = ", ".join(_operation_label(op) for op in tool_operations().get(tool, ())) or "без запросов к API"
                lines.append(f"- `{tool}` → {ops}")
            lines.append("")
        if topic.answer_fields:
            lines.append("Данные из ответа SDK:")
            lines.append("")
            for ref in topic.answer_fields:
                operation, path = ref.split(":", 1)
                description = response_fields(operation).get(path, "")
                lines.append(f"- `{operation}` → `{path}`" + (f" — {description}" if description else ""))
            lines.append("")
        if topic.dictionaries:
            lines.append(
                "Справочники для расшифровки: "
                + ", ".join(f"`{name}` ({DICTIONARIES[name]})" for name in topic.dictionaries)
                + "."
            )
            lines.append("")
        lines.append(f"Параметры: {_params(topic)}.")
        lines.append("")
        if topic.answer_must:
            titles = ", ".join(ANSWER_TITLES[kind] for kind in topic.answer_must)
            lines += [f"В ответе обязательно: {titles} (проверяет `agent/validation.py`).", ""]
        if topic.notes:
            lines += [f"Важно: {topic.notes}", ""]
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Сборка docs/topic-matrix.md")
    parser.add_argument("--check", action="store_true", help="проверить, что документ актуален")
    args = parser.parse_args(argv)
    content = render()
    if args.check:
        current = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
        if current != content:
            print("docs/topic-matrix.md устарел: запустите python scripts/generate_topic_matrix.py")
            return 1
        return 0
    DOC.write_text(content, encoding="utf-8")
    print(f"Записан {DOC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
