"""Naive Qdrant baseline tests. Mocked client and embeddings — no network."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from contextdrift.ingest import format_message
from contextdrift.metrics import _current_phase
from contextdrift.naive import (
    CHUNK_COLLECTION,
    INGEST_FIRST,
    find_chunk_collection,
    naive_vector_search,
    shape_results,
)

INCIDENT = {
    "channel": "incident-alerts",
    "user": "@alex",
    "timestamp": "2026-08-11T09:14:02Z",
    "text": "Checkout is returning 504s, ticket CHECKOUT-504.",
}
DECOY = {
    "channel": "general",
    "user": "@pat",
    "timestamp": "2026-08-11T09:20:00Z",
    "text": "504 gateway timeout on the marketing site.",
}
BLOCKER = {
    "channel": "backend-dev",
    "user": "@marcus",
    "timestamp": "2026-08-11T12:01:00Z",
    "text": "blocking the release, rolling back to sync checks.",
}


def test_find_chunk_collection_prefers_exact_name():
    assert find_chunk_collection(["Entity_name", CHUNK_COLLECTION, "Document"]) == CHUNK_COLLECTION
    assert find_chunk_collection(["slack_DocumentChunk_text"]) == "slack_DocumentChunk_text"
    assert find_chunk_collection(["Entity_name", "other"]) is None
    assert find_chunk_collection([]) is None


def test_shape_results_sorts_by_score_and_parses_prefix():
    hits = [
        SimpleNamespace(score=0.41, payload={"text": format_message(DECOY)}),
        {"score": 0.93, "payload": {"text": format_message(INCIDENT)}},
        SimpleNamespace(score=0.62, payload={"text": format_message(BLOCKER)}),
    ]
    shaped = shape_results(hits)
    assert [row["score"] for row in shaped] == [0.93, 0.62, 0.41]
    assert shaped[0] == {
        "text": INCIDENT["text"],
        "score": 0.93,
        "channel": "incident-alerts",
        "user": "@alex",
    }
    assert shaped[1]["channel"] == "backend-dev"
    assert shaped[1]["user"] == "@marcus"
    assert shaped[2]["channel"] == "general"
    assert "timestamp" not in shaped[0]


class _FakeCollections:
    def __init__(self, names: list[str]):
        self.collections = [SimpleNamespace(name=name) for name in names]


class _FakeQdrant:
    def __init__(self, names: list[str], points: list[object]):
        self.names = names
        self.points = points
        self.queries: list[dict] = []

    def get_collections(self):
        return _FakeCollections(self.names)

    def query_points(self, **kwargs):
        self.queries.append(kwargs)
        return SimpleNamespace(points=self.points)


@pytest.mark.asyncio
async def test_naive_vector_search_shapes_mocked_qdrant_hits(monkeypatch):
    points = [
        SimpleNamespace(score=0.41, payload={"text": format_message(DECOY)}),
        SimpleNamespace(score=0.93, payload={"text": format_message(INCIDENT)}),
        SimpleNamespace(score=0.62, payload={"text": format_message(BLOCKER)}),
    ]
    client = _FakeQdrant([CHUNK_COLLECTION], points)
    phases: list[str] = []

    async def fake_embed(query: str) -> list[float]:
        phases.append(_current_phase.get())
        assert query == "was the checkout 504 bug fixed?"
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr("contextdrift.naive.configure_cognee", lambda: None)
    monkeypatch.setattr("contextdrift.naive.register_metrics", lambda: None)
    monkeypatch.setattr("contextdrift.naive.cost_since", lambda _mark: 0.0004)
    monkeypatch.setattr("contextdrift.naive._qdrant_client", lambda: client)
    monkeypatch.setattr("contextdrift.naive._embed_query", fake_embed)

    result = await naive_vector_search("was the checkout 504 bug fixed?", top_k=3)

    assert tuple(result) == ("results", "cost_usd", "latency_ms", "error")
    assert result["error"] is None
    assert result["cost_usd"] == pytest.approx(0.0004)
    assert result["latency_ms"] >= 0.0
    assert [row["score"] for row in result["results"]] == [0.93, 0.62, 0.41]
    assert result["results"][0]["channel"] == "incident-alerts"
    assert result["results"][0]["user"] == "@alex"
    assert result["results"][0]["text"] == INCIDENT["text"]
    assert phases == ["naive_embed"]
    assert client.queries[0]["collection_name"] == CHUNK_COLLECTION
    assert client.queries[0]["query"] == [0.1, 0.2, 0.3]
    assert client.queries[0]["limit"] == 3


@pytest.mark.asyncio
async def test_naive_vector_search_missing_collection_is_ingest_first(monkeypatch):
    client = _FakeQdrant(["Entity_name"], [])

    async def fake_embed(_query: str) -> list[float]:
        raise AssertionError("must not embed when the chunk collection is missing")

    monkeypatch.setattr("contextdrift.naive.configure_cognee", lambda: None)
    monkeypatch.setattr("contextdrift.naive.register_metrics", lambda: None)
    monkeypatch.setattr("contextdrift.naive._qdrant_client", lambda: client)
    monkeypatch.setattr("contextdrift.naive._embed_query", fake_embed)

    result = await naive_vector_search("anything")
    assert result["error"] == INGEST_FIRST
    assert result["results"] == []
    assert result["cost_usd"] >= 0.0
    assert result["latency_ms"] >= 0.0
    assert client.queries == []


@pytest.mark.asyncio
async def test_naive_vector_search_never_raises(monkeypatch):
    monkeypatch.setattr("contextdrift.naive.configure_cognee", lambda: None)
    monkeypatch.setattr("contextdrift.naive.register_metrics", lambda: None)

    def boom() -> None:
        raise RuntimeError("qdrant went sideways")

    monkeypatch.setattr("contextdrift.naive._qdrant_client", boom)

    result = await naive_vector_search("anything")
    assert result["error"] == "qdrant went sideways"
    assert result["results"] == []
