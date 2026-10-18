"""Провайдеры моделей: Claude и любые модели с OpenAI-совместимым API.

Американские (Anthropic Claude, OpenAI), китайские (DeepSeek, Qwen, Kimi, GLM) и
российские (YandexGPT, GigaChat) модели подключаются одинаково. Все, кроме
Claude, отдают OpenAI-совместимый интерфейс, поэтому для них нужен один общий
цикл (``llm_openai.py``). Меняется только адрес, ключ и имя модели. GigaChat
работает через прокси gpt2giga, который запускается рядом с агентом. Любой
другой OpenAI-совместимый API подключается провайдером ``custom``.

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
    # Переменная с ID каталога Yandex Cloud: он входит в имя модели
    # (gpt://<каталог>/<модель>) и передаётся в заголовке проекта.
    folder_env: str = ""
    # Переменная с адресом API, если у провайдера нет общего адреса.
    base_url_env: str = ""


PROVIDERS: dict[str, Provider] = {
    # США
    "claude": Provider("anthropic", "claude-haiku-5-5", "ANTHROPIC_API_KEY", note="Anthropic"),
    "openai": Provider("openai", "gpt-4.1-mini", "OPENAI_API_KEY", "https://api.openai.com/v1", "OpenAI"),
    # Китай
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
    # Россия
    "yandex": Provider(
        "openai",
        "yandexgpt/latest",
        "YANDEX_API_KEY",
        "https://ai.api.cloud.yandex.net/v1",
        "YandexGPT (Yandex AI Studio)",
        folder_env="YANDEX_FOLDER_ID",
    ),
    "gigachat": Provider(
        "openai", "GigaChat-2-Max", "", "http://localhost:8090/v1", "GigaChat через локальный прокси gpt2giga"
    ),
    # Любой OpenAI-совместимый API: адрес и модель задаются AGENT_BASE_URL и AGENT_MODEL.
    "custom": Provider("openai", "", "AGENT_API_KEY", "", "Любой OpenAI-совместимый API"),
    # Локально
    "ollama": Provider(
        "openai", "qwen3:8b", "", "http://localhost:11434/v1", "Локальная модель через Ollama, бесплатно"
    ),
    # vLLM: OpenAI-совместимый сервер на своём компьютере или в сети. Модель задаётся при запуске сервера.
    "vllm": Provider("openai", "", "", "http://localhost:8000/v1", "Локальный сервер vLLM, бесплатно"),
}


def get_provider(name: str) -> Provider:
    try:
        return PROVIDERS[name]
    except KeyError:
        known = ", ".join(PROVIDERS)
        raise ValueError(f"Неизвестный провайдер {name!r}. Доступны: {known}") from None
