"""Cognee runtime configuration for ContextDrift.

Adapter registration order is load-bearing: the Qdrant community adapter
must be registered before any ``cognee.config`` call, otherwise Cognee
silently falls back to an unsupported-provider error at query time.
"""

from __future__ import annotations

import os
from pathlib import Path

_APP_ROOT = Path("/app") if Path("/app").is_dir() else Path.cwd()

_adapter_registered = False


class _Settings:
    """Live env-backed settings. Re-reads so tests can monkeypatch os.environ."""

    @property
    def vector_db_url(self) -> str:
        return os.environ.get("VECTOR_DB_URL", "http://qdrant:6333")

    @property
    def data_path(self) -> str:
        return os.environ.get("DATA_PATH", str(_APP_ROOT / "data" / "mock_slack_data.json"))

    @property
    def llm_model(self) -> str:
        return os.environ.get("LLM_MODEL", "gpt-4o-mini")

    @property
    def embedding_model(self) -> str:
        return os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")


settings = _Settings()


def has_api_key() -> bool:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    return bool(key) and not key.startswith("sk-your-key")


def configure_cognee() -> None:
    """Register the Qdrant adapter, then apply Cognee runtime settings.

    Safe to call more than once: adapter registration runs a single time,
    and the subsequent ``cognee.config`` writes are idempotent setters.
    """
    global _adapter_registered

    # Register BEFORE any cognee.config call. Adapter 0.4.0 registers on
    # import of the ``register`` module (it is not callable); newer adapters
    # export a ``register()`` function — support both.
    if not _adapter_registered:
        from cognee_community_vector_adapter_qdrant import register as register_qdrant

        if callable(register_qdrant):
            register_qdrant()
        _adapter_registered = True

    import cognee

    cognee.config.system_root_directory(str(_APP_ROOT / ".cognee_system"))
    cognee.config.data_root_directory(str(_APP_ROOT / ".data_storage"))
    cognee.config.set_relational_db_config({"db_provider": "sqlite"})
    cognee.config.set_vector_db_config(
        {
            "vector_db_provider": "qdrant",
            "vector_db_url": settings.vector_db_url,
            "vector_db_key": os.environ.get("VECTOR_DB_KEY", ""),
        }
    )
    cognee.config.set_llm_model(settings.llm_model)
    cognee.config.set_embedding_model(settings.embedding_model)
    cognee.config.set_embedding_dimensions(1536)
