import pytest


@pytest.fixture(autouse=True)
def _agent_db(tmp_path, monkeypatch):
    # Локальная база агента в тестах — своя для каждого теста, не data/agent.db в репозитории.
    monkeypatch.setenv("AGENT_DB_PATH", str(tmp_path / "agent.db"))
