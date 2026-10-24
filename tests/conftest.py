import pytest


@pytest.fixture(autouse=True)
def _agent_db(tmp_path, monkeypatch):
    # Локальная база агента в тестах — своя для каждого теста, не data/agent.db в репозитории.
    monkeypatch.setenv("AGENT_DB_PATH", str(tmp_path / "agent.db"))


@pytest.fixture(autouse=True)
def _knowledge_index(tmp_path, monkeypatch):
    # Векторы базы знаний — во временной папке; эмбеддинги из окружения разработчика не берутся.
    monkeypatch.setenv("AGENT_KNOWLEDGE_DB", str(tmp_path / "knowledge.db"))
    monkeypatch.delenv("AGENT_EMBEDDING_MODEL", raising=False)
