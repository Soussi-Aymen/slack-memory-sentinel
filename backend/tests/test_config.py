"""Cognee config contract: env-backed settings, idempotent configure_cognee()."""

from __future__ import annotations

import contextdrift.config as config_mod
from contextdrift.config import configure_cognee, has_api_key, settings


def test_settings_exposes_contract_fields(monkeypatch):
    monkeypatch.setenv("VECTOR_DB_URL", "http://example-qdrant:6333")
    monkeypatch.setenv("DATA_PATH", "/tmp/corpus.json")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")

    assert settings.vector_db_url == "http://example-qdrant:6333"
    assert settings.data_path == "/tmp/corpus.json"
    assert settings.llm_model == "gpt-4o-mini"
    assert settings.embedding_model == "text-embedding-3-small"


def test_has_api_key_rejects_placeholder(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-your-key-here")
    assert has_api_key() is False
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live-not-a-placeholder")
    assert has_api_key() is True
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert has_api_key() is False


def test_configure_cognee_is_safely_callable_twice(monkeypatch):
    """Second call must not raise. No Qdrant or OpenAI network traffic."""
    monkeypatch.setenv("VECTOR_DB_URL", "http://qdrant:6333")
    configure_cognee()
    configure_cognee()
    assert config_mod._adapter_registered is True

    from cognee.infrastructure.databases.vector.config import get_vectordb_config

    vector_cfg = get_vectordb_config()
    assert vector_cfg.vector_db_provider == "qdrant"
    assert vector_cfg.vector_dataset_database_handler == "qdrant"


def test_configure_cognee_forwards_openai_key_to_cognee(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live-not-a-placeholder")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    configure_cognee()

    from cognee.infrastructure.databases.vector.embeddings.config import get_embedding_config
    from cognee.infrastructure.llm.config import get_llm_config

    assert get_llm_config().llm_api_key == "sk-live-not-a-placeholder"
    assert get_embedding_config().embedding_api_key == "sk-live-not-a-placeholder"
