"""Integration tests: live Qdrant when the compose network is up.

HTTP compare/ingest wiring lives in ``test_web.py`` (TestClient, mocked search).
This module hits the real vector DB so a Qdrant 1.19 named-vector regression
fails here instead of in the demo UI.
"""

from __future__ import annotations

import os
from urllib.error import URLError
from urllib.request import urlopen

import pytest

from contextdrift.naive import CHUNK_COLLECTION, VECTOR_NAME


def _vector_db_url() -> str:
    return os.environ.get("VECTOR_DB_URL", "http://qdrant:6333").rstrip("/")


def _qdrant_up() -> bool:
    try:
        with urlopen(_vector_db_url() + "/readyz", timeout=1.5) as resp:  # noqa: S310
            return 200 <= getattr(resp, "status", 200) < 300
    except (URLError, OSError, TimeoutError, ValueError):
        return False


pytestmark = pytest.mark.skipif(not _qdrant_up(), reason="Qdrant is not reachable")


def _client():
    from qdrant_client import QdrantClient

    return QdrantClient(url=_vector_db_url())


def test_chunk_collection_exists_with_named_text_vector():
    info = _client().get_collection(CHUNK_COLLECTION)
    vectors = info.config.params.vectors
    assert isinstance(vectors, dict)
    assert VECTOR_NAME in vectors
    assert vectors[VECTOR_NAME].size == 1536


def test_query_points_with_using_text_returns_points():
    hits = _client().query_points(
        collection_name=CHUNK_COLLECTION,
        query=[0.0] * 1536,
        using=VECTOR_NAME,
        limit=3,
        with_payload=True,
    )
    assert hits.points is not None
    assert len(hits.points) >= 1
    payload = hits.points[0].payload or {}
    raw = str(payload.get("text") or payload.get("content") or "")
    assert raw


def test_query_points_without_using_is_rejected():
    from qdrant_client.http.exceptions import UnexpectedResponse

    with pytest.raises(UnexpectedResponse) as caught:
        _client().query_points(
            collection_name=CHUNK_COLLECTION,
            query=[0.0] * 1536,
            limit=1,
            with_payload=True,
        )
    assert "Not existing vector name" in str(caught.value)


def test_unknown_collection_errors():
    from qdrant_client.http.exceptions import UnexpectedResponse

    with pytest.raises((UnexpectedResponse, ValueError)):
        _client().query_points(
            collection_name="does_not_exist_contextdrift",
            query=[0.0] * 1536,
            using=VECTOR_NAME,
            limit=1,
        )
