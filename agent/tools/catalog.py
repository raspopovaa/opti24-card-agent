"""Каталог инструментов для выбора по темам: ядро и инструменты, меняющие данные.

Какие инструменты открыть для вопроса, решает тема из матрицы (agent/topics.toml). Ядро видно
всегда: оно дешёвое и нужно почти в каждом разговоре. Тест проверяет, что каждый инструмент
агента входит в ядро или хотя бы в одну тему.
"""

from __future__ import annotations

CORE: tuple[str, ...] = ("get_account_context", "lookup_knowledge", "get_api_usage")

# Изменяют данные: в режиме read_only не показываются (изменение всё равно отклонил бы
# ApprovalGate, а схемы только зря занимали бы место и подталкивали бы модель к попытке).
# Действия, которые нельзя отменить: их выполняют только после подтверждения человеком.
# Удаление лимитов и правил не восстанавливает прежние ID и настройки; заказ отчёта списывает
# тарифицируемые запросы из пакета. Остальные изменения (блокировка, комментарий, лимиты) меняются
# обратно другой операцией, выполняются сразу и записываются в журнал.
IRREVERSIBLE_ACTIONS: frozenset[str] = frozenset(
    {"remove_limit", "remove_region_limit", "remove_restriction", "order_fuel_report"}
)

WRITE_TOOLS: frozenset[str] = frozenset(
    {
        "block_cards",
        "set_card_comment",
        "set_limit",
        "set_region_limit",
        "set_restriction",
        "remove_limit",
        "remove_region_limit",
        "remove_restriction",
        "order_fuel_report",
    }
)
