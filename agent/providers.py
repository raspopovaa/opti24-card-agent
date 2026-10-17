"""Провайдеры моделей: Claude и любые модели с OpenAI-совместимым API.

Китайские провайдеры (DeepSeek, Qwen, Kimi, GLM) отдают OpenAI-совместимый
интерфейс, поэтому для них нужен один общий цикл (``llm_openai.py``). Меняется
только адрес, ключ и имя модели.

Названия моделей у провайдеров меняются каждые несколько месяцев. Значения
ниже — стартовые; актуальное имя берите в документации провайдера и задавайте
переменной ``AGENT_MODEL``. Адрес можно переопределить через ``AGENT_BASE_URL``.

Важно про данные: запросы уходят на серверы провайдера. На DEMO это безопасно.
С рабочим договором (номера карт, водители, суммы) учтите требования закона
о персональных данных и политику вашей компании, либо используйте локальную
модель через Ollama: данные не покидают компьютер.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Provider:
    kind: str  # "anthropic" или "openai"
    default_model: str
    key_env: str  # переменная окружения с ключом ("" — ключ не нужен)
    base_url: str = ""
    note: str = ""


PROVIDERS: dict[str, Provider] = {
    "claude": Provider("anthropic", "claude-haiku-5-5", "ANTHROPIC_API_KEY", note="Anthropic"),
    "deepseek": Provider(
        "openai", "deepseek-chat", "DEEPSEEK_API_KEY", "https://api.deepseek.com", "DeepSeek"
    ),
    "qwen": Provider(
        "openai",
        "qwen-plus",
        "DASHSCOPE_API_KEY",
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "Alibaba Qwen (Model Studio, международный адрес)",
    ),
    "kimi": Provider(
        "openai", "kimi-k2-0905-preview", "MOONSHOT_API_KEY", "https://api.moonshot.ai/v1", "Moonshot Kimi"
    ),
    "glm": Provider(
        "openai", "glm-4.6", "ZHIPUAI_API_KEY", "https://open.bigmodel.cn/api/paas/v4", "Zhipu GLM"
    ),
    "ollama": Provider(
        "openai", "qwen3:8b", "", "http://localhost:11434/v1", "Локальная модель через Ollama, бесплатно"
    ),
}


def get_provider(name: str) -> Provider:
    try:
        return PROVIDERS[name]
    except KeyError:
        known = ", ".join(PROVIDERS)
        raise ValueError(f"Неизвестный провайдер {name!r}. Доступны: {known}") from None
