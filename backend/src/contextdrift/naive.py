"""Naive cosine baseline: embed the query and hit Qdrant directly."""

from __future__ import annotations

import time
from typing import Any

from contextdrift.config import configure_cognee, settings
from contextdrift.metrics import cost_since, marker, phase, register_metrics
from contextdrift.search import parse_slack_prefix

CHUNK_COLLECTION = "DocumentChunk_text"
INGEST_FIRST = "No memory yet — ingest the Slack corpus first."


def _result(
    *,
    results: list[dict] | None = None,
    cost_usd: float = 0.0,
    latency_ms: float = 0.0,
    error: str | None = None,
) -> dict:
    return {
        "results": results if results is not None else [],
        "cost_usd": cost_usd,
        "latency_ms": latency_ms,
        "error": error,
    }


def find_chunk_collection(names: list[str]) -> str | None:
    """Prefer exact ``DocumentChunk_text``, else the first name that contains it."""
    if CHUNK_COLLECTION in names:
        return CHUNK_COLLECTION
    for name in names:
        if CHUNK_COLLECTION in name:
            return name
    return None


def _hit_payload(hit: Any) -> dict:
    payload = getattr(hit, "payload", None)
    if payload is None and isinstance(hit, dict):
        payload = hit.get("payload")
    return payload if isinstance(payload, dict) else {}


def _hit_score(hit: Any) -> float:
    if isinstance(hit, dict):
        return float(hit.get("score") or 0.0)
    return float(getattr(hit, "score", 0.0) or 0.0)


def shape_results(hits: list[Any]) -> list[dict]:
    """Turn Qdrant scored points into ``{text, score, channel, user}``, score desc."""
    shaped: list[dict] = []
    for hit in hits:
        payload = _hit_payload(hit)
        raw = payload.get("text") or payload.get("content") or ""
        parsed = parse_slack_prefix(str(raw))
        shaped.append(
            {
                "text": parsed["text"],
                "score": _hit_score(hit),
                "channel": parsed["channel"],
                "user": parsed["user"],
            }
        )
    shaped.sort(key=lambda row: row["score"], reverse=True)
    return shaped


def _collection_names(client: Any) -> list[str]:
    collections = client.get_collections()
    listed = getattr(collections, "collections", collections)
    names: list[str] = []
    for item in listed or []:
        if isinstance(item, str):
            names.append(item)
        else:
            name = getattr(item, "name", None)
            if name is None and isinstance(item, dict):
                name = item.get("name")
            if name:
                names.append(str(name))
    return names


def _query_hits(client: Any, collection: str, vector: list[float], top_k: int) -> list[Any]:
    response = client.query_points(
        collection_name=collection,
        query=vector,
        limit=top_k,
        with_payload=True,
    )
    points = getattr(response, "points", response)
    return list(points or [])


async def _embed_query(query: str) -> list[float]:
    import litellm

    response = await litellm.aembedding(model=settings.embedding_model, input=[query])
    data = response.data[0]
    if isinstance(data, dict):
        return list(data["embedding"])
    return list(data.embedding)


def _qdrant_client() -> Any:
    from qdrant_client import QdrantClient

    return QdrantClient(url=settings.vector_db_url)


async def naive_vector_search(query: str, top_k: int = 3) -> dict:
    """Embed ``query`` and return the top-k cosine hits. Never raises."""
    configure_cognee()
    register_metrics()

    cost_mark = marker()
    started = time.perf_counter()
    try:
        client = _qdrant_client()
        collection = find_chunk_collection(_collection_names(client))
        if collection is None:
            return _result(
                cost_usd=cost_since(cost_mark),
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error=INGEST_FIRST,
            )
        with phase("naive_embed"):
            vector = await _embed_query(query)
        hits = _query_hits(client, collection, vector, top_k)
        return _result(
            results=shape_results(hits),
            cost_usd=cost_since(cost_mark),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error=None,
        )
    except Exception as exc:
        return _result(
            cost_usd=cost_since(cost_mark),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error=str(exc) or type(exc).__name__,
        )
