"""Настройки агента: модель, режим работы и бюджеты.

Всё, что меняется между запусками, читается из переменных окружения, а не
зашивается в код. Секреты (ключи API) здесь не хранятся: ключ модели читается
из переменной провайдера (по умолчанию AGENT_API_KEY для Cloud.ru), а учётные
данные Opti24 — apisdkopti24 из своего .env-файла.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .providers import DEFAULT_PROVIDER, get_provider


def _int_env(name: str, default: int) -> int:
    """Целое из переменной окружения; нечисловое значение — понятная ошибка, а не трассировка."""
    raw = os.environ.get(name, str(default))
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"Переменная {name} должна быть целым числом, а не {raw[:30]!r}") from None


def _choice_env(name: str, default: str, allowed: set[str]) -> str:
    value = os.environ.get(name, default).strip().lower()
    if value not in allowed:
        raise ValueError(f"Переменная {name} должна быть одним из {sorted(allowed)}, а не {value[:30]!r}")
    return value


def default_env_file(kind: str) -> str:
    """``.env.<kind>`` в папке проекта, если есть, иначе ``~/.opti24-<kind>.env``."""
    local = Path(__file__).resolve().parent.parent / f".env.{kind}"
    return str(local if local.is_file() else f"~/.opti24-{kind}.env")


class Mode(StrEnum):
    """Что агенту разрешено делать с договором."""

    READ_ONLY = "read_only"  # только чтение: изменяющие инструменты отключены
    CONFIRM = "confirm"  # изменения выполняются только после подтверждения человеком


@dataclass(frozen=True, slots=True)
class Settings:
    provider: str
    model: str
    base_url: str
    api_key: str
    effort: str
    mode: Mode
    max_iterations: int
    max_billable_calls: int
    opti24_env_file: Path
    audit_log: Path
    project: str = ""  # ID каталога Yandex Cloud; для остальных провайдеров пусто
    tool_routing: str = "auto"  # двухуровневый выбор инструментов у OpenAI-совместимых провайдеров: auto/on включают, off отключает
    verify_answer: bool = False  # повторная сверка чисел инструментом тратит пакет API; AGENT_VERIFY_ANSWER=on включает

    @classmethod
    def from_env(cls) -> Settings:
        provider_name = os.environ.get("AGENT_PROVIDER", DEFAULT_PROVIDER)
        provider = get_provider(provider_name)
        # Самая дешёвая модель провайдера по умолчанию; поднимать, только если
        # оценка (evals/) покажет ошибки в выборе инструментов.
        model = os.environ.get("AGENT_MODEL", provider.default_model)
        project = os.environ.get(provider.folder_env, "") if provider.folder_env else ""
        if project and not model.startswith("gpt://"):
            model = f"gpt://{project}/{model}"
        return cls(
            provider=provider_name,
            model=model,
            base_url=os.environ.get("AGENT_BASE_URL")
            or (os.environ.get(provider.base_url_env, "") if provider.base_url_env else "")
            or provider.base_url,
            # Для Ollama и прокси GigaChat ключ не нужен, но клиент OpenAI требует непустую строку.
            api_key=os.environ.get(provider.key_env, "") if provider.key_env else "ollama",
            effort=os.environ.get("AGENT_EFFORT", "medium"),
            mode=Mode(os.environ.get("AGENT_MODE", Mode.READ_ONLY)),
            # Предохранитель от зацикливания: столько обращений к модели за один вопрос.
            max_iterations=_int_env("AGENT_MAX_ITERATIONS", 8),
            # Предохранитель для пакета Opti24: тарифицируемых вызовов API за сессию.
            max_billable_calls=_int_env("AGENT_MAX_BILLABLE_CALLS", 10),
            opti24_env_file=Path(
                os.environ.get("OPTI24_ENV_FILE") or default_env_file("demo")
            ).expanduser(),
            audit_log=Path(os.environ.get("AGENT_AUDIT_LOG", "./agent-audit.jsonl")),
            project=project,
            tool_routing=_choice_env("AGENT_TOOL_ROUTING", "auto", {"auto", "on", "off"}),
            verify_answer=_choice_env("AGENT_VERIFY_ANSWER", "off", {"on", "off"}) == "on",
        )
