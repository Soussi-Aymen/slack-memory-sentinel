"""Graph recall over the Slack corpus via Cognee ``recall()``."""

from __future__ import annotations

import re
import time
from typing import Any

from contextdrift.config import configure_cognee
from contextdrift.metrics import cost_since, marker, phase, register_metrics

DATASET_NAME = "slack"
INGEST_FIRST = "No memory yet — ingest the Slack corpus first."

_PREFIX = re.compile(
    r"^\[Channel: #(?P<channel>[^\]]+)\]\s+"
    r"(?P<user>\S+)\s+"
    r"\((?P<timestamp>[^)]+)\):\s*"
    r"(?P<body>.*)$",
    re.DOTALL,
)


def _item_text(item: Any) -> str:
    """Pull a renderable string out of a RecallResponse-shaped object."""
    if item is None:
        return ""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("text", "answer", "content"):
            value = item.get(key)
            if isinstance(value, str) and value:
                return value
        return ""
    for attr in ("text", "answer", "content"):
        value = getattr(item, attr, None)
        if isinstance(value, str) and value:
            return value
    return ""


def unwrap_graph_answer(results: Any) -> str:
    """Defensively flatten ``list[RecallResponse]`` (or nearby shapes) to a string."""
    if results is None:
        return ""
    if isinstance(results, str):
        return results.strip()
    if isinstance(results, list):
        parts = [unwrap_graph_answer(item) for item in results]
        return "\n".join(part for part in parts if part).strip()
    return _item_text(results).strip()


def parse_slack_prefix(raw: str) -> dict[str, str]:
    """Split ``[Channel: #x] @user (ts): body`` into provenance fields."""
    text = raw if isinstance(raw, str) else str(raw or "")
    match = _PREFIX.match(text.strip())
    if not match:
        return {"text": text, "channel": "", "user": "", "timestamp": ""}
    return {
        "text": match.group("body"),
        "channel": match.group("channel"),
        "user": match.group("user"),
        "timestamp": match.group("timestamp"),
    }


def unwrap_chunks(results: Any) -> list[dict[str, str]]:
    """Turn CHUNKS recall hits into ``{text, channel, user, timestamp}`` dicts."""
    if results is None:
        return []
    if isinstance(results, str):
        parsed = parse_slack_prefix(results)
        return [parsed] if parsed["text"] or parsed["channel"] else []
    if not isinstance(results, list):
        results = [results]
    chunks: list[dict[str, str]] = []
    for item in results:
        if isinstance(item, list):
            chunks.extend(unwrap_chunks(item))
            continue
        parsed = parse_slack_prefix(_item_text(item))
        if parsed["text"] or parsed["channel"]:
            chunks.append(parsed)
    return chunks


def _result(
    *,
    graph_answer: str = "",
    evidence_chunks: list[dict[str, str]] | None = None,
    cost_usd: float = 0.0,
    latency_ms: float = 0.0,
    error: str | None = None,
) -> dict:
    return {
        "graph_answer": graph_answer,
        "evidence_chunks": evidence_chunks if evidence_chunks is not None else [],
        "cost_usd": cost_usd,
        "latency_ms": latency_ms,
        "error": error,
    }


async def query_slack_memory(query: str) -> dict:
    """Recall a graph answer plus chunk evidence. Never raises."""
    configure_cognee()
    register_metrics()

    import cognee
    from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
    from cognee.modules.search.types import SearchType

    cost_mark = marker()
    started = time.perf_counter()
    try:
        with phase("graph_recall"):
            graph_hits = await cognee.recall(
                query,
                query_type=SearchType.GRAPH_COMPLETION,
                datasets=[DATASET_NAME],
            )
            chunk_hits = await cognee.recall(
                query,
                query_type=SearchType.CHUNKS,
                datasets=[DATASET_NAME],
            )
        return _result(
            graph_answer=unwrap_graph_answer(graph_hits),
            evidence_chunks=unwrap_chunks(chunk_hits),
            cost_usd=cost_since(cost_mark),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error=None,
        )
    except CollectionNotFoundError:
        return _result(
            cost_usd=cost_since(cost_mark),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error=INGEST_FIRST,
        )
    except Exception as exc:
        return _result(
            cost_usd=cost_since(cost_mark),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error=str(exc) or type(exc).__name__,
        )
