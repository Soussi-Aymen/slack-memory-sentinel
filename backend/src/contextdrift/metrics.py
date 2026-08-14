"""In-process LiteLLM cost, token, and latency tracker.

Every Cognee LLM call routes through litellm. A CustomLogger success callback
captures model, tokens, USD cost and latency into a thread-safe ring buffer
tagged with the current ``phase()`` label. Cognee's ``SessionModelUsage`` table
is the fallback when no callback events arrive (callback registration missed).
"""

from __future__ import annotations

import math
import sqlite3
import threading
import time
from collections import defaultdict, deque
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from litellm.integrations.custom_logger import CustomLogger

PhaseName = Literal["ingest", "graph_recall", "naive_embed"]

SNAPSHOT_KEYS = (
    "total_cost_usd",
    "by_phase",
    "by_model",
    "tokens_in",
    "tokens_out",
    "p50_latency_ms",
    "call_count",
)

_BUFFER_MAX = 8192
_UNATTRIBUTED = "unattributed"

_current_phase: ContextVar[str] = ContextVar("contextdrift_phase", default=_UNATTRIBUTED)
_lock = threading.Lock()
_buffer: deque[_CallRecord] = deque(maxlen=_BUFFER_MAX)
_logger: LiteLLMMetricsLogger | None = None


@dataclass(frozen=True, slots=True)
class _CallRecord:
    recorded_at: float
    phase: str
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: float


def _empty_snapshot() -> dict[str, Any]:
    return {
        "total_cost_usd": 0.0,
        "by_phase": {},
        "by_model": {},
        "tokens_in": 0,
        "tokens_out": 0,
        "p50_latency_ms": 0.0,
        "call_count": 0,
    }


def _latency_ms(start_time: Any, end_time: Any) -> float:
    if start_time is None or end_time is None:
        return 0.0
    try:
        delta = end_time - start_time
    except TypeError:
        return 0.0
    seconds = delta.total_seconds() if hasattr(delta, "total_seconds") else float(delta)
    return max(0.0, seconds * 1000.0)


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _usage_tokens(kwargs: dict[str, Any], response_obj: Any) -> tuple[int, int]:
    usage = getattr(response_obj, "usage", None)
    if usage is None and isinstance(response_obj, dict):
        usage = response_obj.get("usage")
    if usage is not None:
        if isinstance(usage, dict):
            prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
            completion = usage.get("completion_tokens", usage.get("output_tokens"))
        else:
            prompt = getattr(usage, "prompt_tokens", None)
            if prompt is None:
                prompt = getattr(usage, "input_tokens", None)
            completion = getattr(usage, "completion_tokens", None)
            if completion is None:
                completion = getattr(usage, "output_tokens", None)
        if prompt is not None or completion is not None:
            return _as_int(prompt), _as_int(completion)

    std = kwargs.get("standard_logging_object") or {}
    if isinstance(std, dict):
        return _as_int(std.get("prompt_tokens")), _as_int(std.get("completion_tokens"))
    return 0, 0


def _response_cost_usd(kwargs: dict[str, Any]) -> float:
    cost = kwargs.get("response_cost")
    if cost not in (None, 0, 0.0):
        return _as_float(cost)
    std = kwargs.get("standard_logging_object") or {}
    if isinstance(std, dict):
        cost = std.get("response_cost")
        if cost not in (None, 0, 0.0):
            return _as_float(cost)
        breakdown = std.get("cost_breakdown") or {}
        if isinstance(breakdown, dict):
            return _as_float(breakdown.get("total_cost"))
    return _as_float(kwargs.get("response_cost"))


def _model_name(kwargs: dict[str, Any]) -> str:
    std = kwargs.get("standard_logging_object") or {}
    if isinstance(std, dict):
        name = std.get("model") or std.get("model_group")
        if name:
            return str(name)
    return str(kwargs.get("model") or "unknown")


def _record(
    *,
    model: str,
    tokens_in: int,
    tokens_out: int,
    cost_usd: float,
    latency_ms: float,
    phase: str | None = None,
    recorded_at: float | None = None,
) -> None:
    event = _CallRecord(
        recorded_at=time.perf_counter() if recorded_at is None else recorded_at,
        phase=phase if phase is not None else _current_phase.get(),
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
    )
    with _lock:
        _buffer.append(event)


def _p50(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * 0.5
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _aggregate(records: list[_CallRecord]) -> dict[str, Any]:
    if not records:
        return _empty_snapshot()
    by_phase: dict[str, float] = defaultdict(float)
    by_model: dict[str, float] = defaultdict(float)
    tokens_in = 0
    tokens_out = 0
    total = 0.0
    latencies: list[float] = []
    for event in records:
        by_phase[event.phase] += event.cost_usd
        by_model[event.model] += event.cost_usd
        tokens_in += event.tokens_in
        tokens_out += event.tokens_out
        total += event.cost_usd
        latencies.append(event.latency_ms)
    return {
        "total_cost_usd": total,
        "by_phase": dict(by_phase),
        "by_model": dict(by_model),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "p50_latency_ms": _p50(latencies),
        "call_count": len(records),
    }


def _cognee_system_roots() -> list[Path]:
    roots: list[Path] = []
    for candidate in (
        Path("/app/.cognee_system"),
        Path(".cognee_system"),
        Path("backend/.cognee_system"),
    ):
        if candidate.exists():
            roots.append(candidate)
    return roots


def _read_session_model_usage() -> list[Any]:
    """Best-effort rows from Cognee's ``SessionModelUsage`` table. Never raises.

    The class is imported so the fallback stays coupled to Cognee's schema
    (``tokens_in``, ``tokens_out``, ``cost_usd`` per model). The read itself is a
    readonly sqlite query so ``snapshot()`` stays synchronous and never opens a
    nested event loop inside FastAPI.
    """
    try:
        from cognee.modules.session_lifecycle.models import SessionModelUsage as _Usage
    except Exception:
        _Usage = None  # noqa: N806 — table may still exist on disk

    table = getattr(_Usage, "__tablename__", "session_model_usage")
    for root in _cognee_system_roots():
        for db_path in (*root.rglob("*.sqlite"), *root.rglob("*.db")):
            try:
                conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
                conn.row_factory = sqlite3.Row
                try:
                    rows = conn.execute(
                        f"SELECT model, tokens_in, tokens_out, cost_usd FROM {table}"
                    ).fetchall()
                finally:
                    conn.close()
            except Exception:
                continue
            if rows:
                return list(rows)
    return []


def _row_field(row: Any, name: str) -> Any:
    if isinstance(row, dict):
        return row.get(name)
    keys = getattr(row, "keys", None)
    if callable(keys):
        try:
            if name in keys():
                return row[name]
        except Exception:
            pass
    return getattr(row, name, None)


def _snapshot_from_usage_rows(rows: list[Any]) -> dict[str, Any]:
    if not rows:
        return _empty_snapshot()
    by_model: dict[str, float] = defaultdict(float)
    tokens_in = 0
    tokens_out = 0
    total = 0.0
    for row in rows:
        model = str(_row_field(row, "model") or "unknown")
        tin = _as_int(_row_field(row, "tokens_in"))
        tout = _as_int(_row_field(row, "tokens_out"))
        cost = _as_float(_row_field(row, "cost_usd"))
        by_model[model] += cost
        tokens_in += tin
        tokens_out += tout
        total += cost
    return {
        "total_cost_usd": total,
        "by_phase": {},
        "by_model": dict(by_model),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "p50_latency_ms": 0.0,
        "call_count": 0,
    }


class LiteLLMMetricsLogger(CustomLogger):
    """Records successful LiteLLM calls into the process-local ring buffer."""

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            kwargs = kwargs or {}
            tokens_in, tokens_out = _usage_tokens(kwargs, response_obj)
            _record(
                model=_model_name(kwargs),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=_response_cost_usd(kwargs),
                latency_ms=_latency_ms(start_time, end_time),
            )
        except Exception:
            return

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.log_success_event(kwargs, response_obj, start_time, end_time)


def register_metrics() -> None:
    """Idempotently attach ``LiteLLMMetricsLogger`` to ``litellm.callbacks``."""
    global _logger
    import litellm

    with _lock:
        if _logger is None:
            _logger = LiteLLMMetricsLogger()
        callbacks = list(getattr(litellm, "callbacks", None) or [])
        if _logger not in callbacks:
            litellm.callbacks = [*callbacks, _logger]


@contextmanager
def phase(name: PhaseName) -> Iterator[str]:
    """Tag every callback recorded inside this block with ``name``."""
    token = _current_phase.set(str(name))
    try:
        yield str(name)
    finally:
        _current_phase.reset(token)


def snapshot() -> dict[str, Any]:
    """Aggregate the ring buffer, or SessionModelUsage when no events arrived."""
    with _lock:
        records = list(_buffer)
    if records:
        return _aggregate(records)
    return _snapshot_from_usage_rows(_read_session_model_usage())


def marker() -> float:
    """Return a monotonic timestamp for a later ``cost_since`` delta."""
    return time.perf_counter()


def cost_since(marker: float) -> float:
    """USD spent on buffered calls recorded at or after ``marker``."""
    with _lock:
        return sum(event.cost_usd for event in _buffer if event.recorded_at >= marker)


def _reset() -> None:
    """Drop buffered events. Tests only."""
    with _lock:
        _buffer.clear()
