"""Ядро агента: цикл «модель → инструмент → модель».

Для Claude цикл ведёт Tool Runner из SDK Anthropic, для остальных провайдеров
(OpenAI, DeepSeek, Qwen, Kimi, GLM, YandexGPT, GigaChat, Ollama) — цикл из ``llm_openai.py``. Инструменты,
инструкция и предохранители у всех одни и те же.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from apisdkopti24 import APIClient

from .classifier import classify, hint, log_question
from .config import Settings
from .core_types import Usage
from .prompt import SYSTEM_PROMPT
from .providers import get_provider
from .safety import ApprovalGate, AskHuman, BillingBudget
from .tools import build_tools
from .tools.router import ToolRouter


def make_llm(settings: Settings) -> Any:
    """Клиент выбранного провайдера. Создаётся лениво: тесты и дашборд обходятся без ключа."""
    if get_provider(settings.provider).kind == "anthropic":
        from anthropic import AsyncAnthropic

        return AsyncAnthropic(api_key=settings.api_key or None)
    from openai import AsyncOpenAI

    if not settings.api_key:
        raise RuntimeError(
            f"Не задан ключ для провайдера {settings.provider}: "
            f"переменная {get_provider(settings.provider).key_env}"
        )
    if not settings.api_key.isascii():
        raise RuntimeError(
            f"В переменной {get_provider(settings.provider).key_env} есть не-ASCII символы "
            "(русские буквы, «умные» кавычки или пробелы). Ключ состоит только из латиницы и цифр: "
            "скопируйте его заново и задайте так: export ИМЯ='sk-...' обычными кавычками."
        )
    if not settings.base_url or not settings.model:
        url_env = get_provider(settings.provider).base_url_env or "AGENT_BASE_URL"
        raise RuntimeError(
            f"Для провайдера {settings.provider} задайте адрес API и модель: "
            f"переменные {url_env} и AGENT_MODEL"
        )
    folder_env = get_provider(settings.provider).folder_env
    if folder_env and not settings.project:
        raise RuntimeError(f"Для провайдера {settings.provider} задайте ID каталога: переменная {folder_env}")
    return AsyncOpenAI(
        api_key=settings.api_key.strip(), base_url=settings.base_url, project=settings.project or None
    )


@dataclass(slots=True)
class FuelCardAgent:
    settings: Settings
    opti24: APIClient
    ask_human: AskHuman
    llm: Any = None
    usage: Usage = field(default_factory=Usage)
    history: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)  # для оценки выбора инструментов
    tool_inputs: list[tuple[str, dict[str, Any]]] = field(default_factory=list)  # и их аргументов
    budget: BillingBudget = field(init=False)
    gate: ApprovalGate = field(init=False)
    tools: list[Any] = field(init=False)
    router: ToolRouter | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.budget = BillingBudget(limit=self.settings.max_billable_calls)
        self.gate = ApprovalGate(
            mode=self.settings.mode, ask=self.ask_human, audit_log=self.settings.audit_log
        )
        self.tools = build_tools(self.opti24, self.budget, self.gate)
        # Двухуровневый выбор нужен там, где нет кеша промпта и модели слабее: у Claude полный набор
        # из 27 схем дёшев (кеш) и проверен на evals, поэтому для него маршрутизатор не включается.
        if get_provider(self.settings.provider).kind == "openai" and self.settings.tool_routing != "off":
            self.router = ToolRouter(self.tools, self.settings.mode)

    async def ask(self, question: str) -> str:
        if self.llm is None:
            self.llm = make_llm(self.settings)
        # Дата идёт в сообщении пользователя, а не в системной инструкции:
        # так инструкция остаётся неизменной и целиком берётся из кеша.
        user_message = {"role": "user", "content": f"[Сегодня {date.today():%Y-%m-%d}] {question}"}
        # Тема вопроса и подсказка по ней идут модели только с текущим вопросом: в истории они бы
        # устарела и сбивала следующие вопросы.
        found = classify(question)
        log_question(question, found)
        note = hint(found)
        current = {**user_message, "content": f"{user_message['content']}\n{note}"} if note else user_message
        messages = [*self.history, current]

        if get_provider(self.settings.provider).kind == "anthropic":
            final_text = await self._ask_claude(messages)
        else:
            from .llm_openai import run_openai_loop

            if self.router is not None:
                self.router.preload(question)
            final_text, called = await run_openai_loop(
                self.llm,
                self.settings.model,
                SYSTEM_PROMPT,
                messages,
                self.router.visible if self.router is not None else self.tools,
                self.settings.max_iterations,
                self.usage,
            )
            self.tool_calls += [name for name, _ in called]
            self.tool_inputs += called

        # В историю кладём только вопрос и итоговый ответ. Тяжёлые результаты
        # инструментов остаются за кадром, и следующий вопрос стоит дешевле.
        self.history += [user_message, {"role": "assistant", "content": final_text}]
        return final_text or "Не удалось получить ответ: попробуйте переформулировать вопрос."

    async def _ask_claude(self, messages: list[dict[str, Any]]) -> str:
        runner = self.llm.beta.messages.tool_runner(
            model=self.settings.model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=self.tools,
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": self.settings.effort},
            cache_control={"type": "ephemeral"},
            max_iterations=self.settings.max_iterations,
        )
        final_text = ""
        async for message in runner:
            self.usage.add_anthropic(message.usage)
            uses = [b for b in message.content if b.type == "tool_use"]
            self.tool_calls += [b.name for b in uses]
            self.tool_inputs += [(b.name, dict(b.input)) for b in uses]
            texts = [block.text for block in message.content if block.type == "text"]
            if texts:
                final_text = "\n".join(texts)
        return final_text
