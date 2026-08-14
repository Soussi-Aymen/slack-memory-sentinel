"""Graph-recall contract tests. Fabricated RecallResponse objects — no network."""

from __future__ import annotations

from types import SimpleNamespace

import cognee
import pytest
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.modules.search.types import SearchType

from contextdrift.ingest import format_message
from contextdrift.metrics import _current_phase
from contextdrift.search import (
    INGEST_FIRST,
    parse_slack_prefix,
    query_slack_memory,
    unwrap_chunks,
    unwrap_graph_answer,
)

CORPUS_RECORD = {
    "channel": "releases",
    "user": "@jordan",
    "timestamp": "2026-08-12T16:40:11Z",
    "text": "hotfix PR #405 merged and deployed to production.",
}

GRAPH_TEXT = (
    "The checkout 504 is resolved. Sarah moved fraud checks to an async worker "
    "and Jordan shipped hotfix PR #405."
)
PREFIXED_CHUNK = (
    "[Channel: #backend-dev] @sarah (2026-08-11T10:02:18Z): "
    "Moving fraud checks onto an async queue worker."
)


def test_parse_slack_prefix_roundtrips_format_message():
    parsed = parse_slack_prefix(format_message(CORPUS_RECORD))
    assert parsed == {
        "text": CORPUS_RECORD["text"],
        "channel": CORPUS_RECORD["channel"],
        "user": CORPUS_RECORD["user"],
        "timestamp": CORPUS_RECORD["timestamp"],
    }


def test_parse_slack_prefix_without_provenance_keeps_raw_text():
    parsed = parse_slack_prefix("no channel prefix here")
    assert parsed == {
        "text": "no channel prefix here",
        "channel": "",
        "user": "",
        "timestamp": "",
    }


def test_unwrap_graph_answer_from_fabricated_recall_objects():
    hits = [
        SimpleNamespace(source="graph", text=GRAPH_TEXT),
        {"answer": "Marcus's rollback was superseded."},
    ]
    assert unwrap_graph_answer(hits) == (f"{GRAPH_TEXT}\nMarcus's rollback was superseded.")
    assert unwrap_graph_answer(GRAPH_TEXT) == GRAPH_TEXT
    assert unwrap_graph_answer(None) == ""
    assert unwrap_graph_answer([]) == ""
    assert unwrap_graph_answer([[SimpleNamespace(text="nested")]]) == "nested"


def test_unwrap_chunks_parses_prefix_from_mixed_shapes():
    hits = [
        SimpleNamespace(source="graph", text=PREFIXED_CHUNK),
        {"text": format_message(CORPUS_RECORD)},
        "plain evidence with no prefix",
        SimpleNamespace(text=""),
    ]
    chunks = unwrap_chunks(hits)
    assert chunks[0] == {
        "text": "Moving fraud checks onto an async queue worker.",
        "channel": "backend-dev",
        "user": "@sarah",
        "timestamp": "2026-08-11T10:02:18Z",
    }
    assert chunks[1]["channel"] == "releases"
    assert chunks[1]["user"] == "@jordan"
    assert chunks[1]["timestamp"] == CORPUS_RECORD["timestamp"]
    assert chunks[1]["text"] == CORPUS_RECORD["text"]
    assert chunks[2] == {
        "text": "plain evidence with no prefix",
        "channel": "",
        "user": "",
        "timestamp": "",
    }
    assert len(chunks) == 3


@pytest.mark.asyncio
async def test_query_slack_memory_unwraps_mocked_recall(monkeypatch):
    calls: list[object] = []
    phases: list[str] = []

    async def fake_recall(query_text, query_type=None, **kwargs):
        calls.append({"query": query_text, "query_type": query_type, "kwargs": kwargs})
        phases.append(_current_phase.get())
        if query_type is SearchType.GRAPH_COMPLETION:
            return [SimpleNamespace(source="graph", text=GRAPH_TEXT)]
        if query_type is SearchType.CHUNKS:
            return [SimpleNamespace(source="graph", text=PREFIXED_CHUNK)]
        raise AssertionError(f"unexpected search type: {query_type}")

    monkeypatch.setattr("contextdrift.search.configure_cognee", lambda: None)
    monkeypatch.setattr("contextdrift.search.register_metrics", lambda: None)
    monkeypatch.setattr("contextdrift.search.cost_since", lambda _mark: 0.007)
    monkeypatch.setattr(cognee, "recall", fake_recall)

    result = await query_slack_memory("was the checkout 504 bug fixed?")

    assert tuple(result) == (
        "graph_answer",
        "evidence_chunks",
        "cost_usd",
        "latency_ms",
        "error",
    )
    assert result["error"] is None
    assert result["graph_answer"] == GRAPH_TEXT
    assert result["evidence_chunks"] == [
        {
            "text": "Moving fraud checks onto an async queue worker.",
            "channel": "backend-dev",
            "user": "@sarah",
            "timestamp": "2026-08-11T10:02:18Z",
        }
    ]
    assert result["cost_usd"] == pytest.approx(0.007)
    assert result["latency_ms"] >= 0.0
    assert [call["query_type"] for call in calls] == [
        SearchType.GRAPH_COMPLETION,
        SearchType.CHUNKS,
    ]
    assert all(call["query"] == "was the checkout 504 bug fixed?" for call in calls)
    assert all(call["kwargs"]["datasets"] == ["slack"] for call in calls)
    assert phases == ["graph_recall", "graph_recall"]


@pytest.mark.asyncio
async def test_query_slack_memory_collection_missing_is_ingest_first(monkeypatch):
    async def fake_recall(*_args, **_kwargs):
        raise CollectionNotFoundError("Collection 'DocumentChunk_text' not found!")

    monkeypatch.setattr("contextdrift.search.configure_cognee", lambda: None)
    monkeypatch.setattr("contextdrift.search.register_metrics", lambda: None)
    monkeypatch.setattr(cognee, "recall", fake_recall)

    result = await query_slack_memory("anything")
    assert result["error"] == INGEST_FIRST
    assert result["graph_answer"] == ""
    assert result["evidence_chunks"] == []
    assert result["cost_usd"] >= 0.0
    assert result["latency_ms"] >= 0.0


@pytest.mark.asyncio
async def test_query_slack_memory_never_raises(monkeypatch):
    async def fake_recall(*_args, **_kwargs):
        raise RuntimeError("vector adapter went sideways")

    monkeypatch.setattr("contextdrift.search.configure_cognee", lambda: None)
    monkeypatch.setattr("contextdrift.search.register_metrics", lambda: None)
    monkeypatch.setattr(cognee, "recall", fake_recall)

    result = await query_slack_memory("anything")
    assert result["error"] == "vector adapter went sideways"
    assert result["graph_answer"] == ""
    assert result["evidence_chunks"] == []
