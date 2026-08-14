"""Unit tests for the LiteLLM cost tracker. Fabricated events only — no network."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from contextdrift import metrics


@pytest.fixture(autouse=True)
def _clean_buffer():
    metrics._reset()
    yield
    metrics._reset()


def _kwargs(
    *,
    model: str = "gpt-4o-mini",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    cost_usd: float = 0.001,
) -> tuple[dict, SimpleNamespace]:
    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    )
    kwargs = {"model": model, "response_cost": cost_usd}
    return kwargs, response


def _times(latency_ms: float) -> tuple[datetime, datetime]:
    start = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)
    end = start + timedelta(milliseconds=latency_ms)
    return start, end


def test_snapshot_keys_match_the_frozen_contract():
    snap = metrics.snapshot()
    assert tuple(snap) == metrics.SNAPSHOT_KEYS
    assert snap == {
        "total_cost_usd": 0.0,
        "by_phase": {},
        "by_model": {},
        "tokens_in": 0,
        "tokens_out": 0,
        "p50_latency_ms": 0.0,
        "call_count": 0,
    }


def test_register_metrics_is_idempotent():
    import litellm

    metrics.register_metrics()
    first = list(litellm.callbacks)
    metrics.register_metrics()
    second = list(litellm.callbacks)
    assert metrics._logger in first
    assert first.count(metrics._logger) == 1
    assert second.count(metrics._logger) == 1


def test_log_success_event_aggregates_tokens_cost_and_latency():
    logger = metrics.LiteLLMMetricsLogger()
    kwargs, response = _kwargs(
        model="gpt-4o-mini", prompt_tokens=100, completion_tokens=40, cost_usd=0.002
    )
    logger.log_success_event(kwargs, response, *_times(80))

    snap = metrics.snapshot()
    assert snap["call_count"] == 1
    assert snap["tokens_in"] == 100
    assert snap["tokens_out"] == 40
    assert snap["total_cost_usd"] == pytest.approx(0.002)
    assert snap["by_model"] == {"gpt-4o-mini": pytest.approx(0.002)}
    assert snap["p50_latency_ms"] == pytest.approx(80.0)
    assert snap["by_phase"] == {"unattributed": pytest.approx(0.002)}


def test_phase_context_manager_tags_events():
    logger = metrics.LiteLLMMetricsLogger()
    with metrics.phase("ingest"):
        kwargs, response = _kwargs(cost_usd=0.01, prompt_tokens=50, completion_tokens=10)
        logger.log_success_event(kwargs, response, *_times(20))
    with metrics.phase("graph_recall"):
        kwargs, response = _kwargs(
            model="gpt-4o-mini", cost_usd=0.003, prompt_tokens=30, completion_tokens=20
        )
        logger.log_success_event(kwargs, response, *_times(40))
    with metrics.phase("naive_embed"):
        kwargs, response = _kwargs(
            model="text-embedding-3-small",
            cost_usd=0.0001,
            prompt_tokens=8,
            completion_tokens=0,
        )
        logger.log_success_event(kwargs, response, *_times(12))

    snap = metrics.snapshot()
    assert snap["call_count"] == 3
    assert snap["tokens_in"] == 88
    assert snap["tokens_out"] == 30
    assert snap["total_cost_usd"] == pytest.approx(0.0131)
    assert snap["by_phase"]["ingest"] == pytest.approx(0.01)
    assert snap["by_phase"]["graph_recall"] == pytest.approx(0.003)
    assert snap["by_phase"]["naive_embed"] == pytest.approx(0.0001)
    assert snap["by_model"]["gpt-4o-mini"] == pytest.approx(0.013)
    assert snap["by_model"]["text-embedding-3-small"] == pytest.approx(0.0001)


def test_p50_latency_is_the_median_of_fabricated_events():
    logger = metrics.LiteLLMMetricsLogger()
    for latency in (10.0, 30.0, 50.0, 70.0, 90.0):
        kwargs, response = _kwargs(cost_usd=0.0)
        logger.log_success_event(kwargs, response, *_times(latency))
    assert metrics.snapshot()["p50_latency_ms"] == pytest.approx(50.0)


def test_marker_and_cost_since_isolate_a_window():
    logger = metrics.LiteLLMMetricsLogger()
    kwargs, response = _kwargs(cost_usd=0.4)
    logger.log_success_event(kwargs, response, *_times(5))

    mark = metrics.marker()
    kwargs, response = _kwargs(cost_usd=0.15)
    logger.log_success_event(kwargs, response, *_times(5))
    kwargs, response = _kwargs(cost_usd=0.05)
    logger.log_success_event(kwargs, response, *_times(5))

    assert metrics.cost_since(mark) == pytest.approx(0.20)
    assert metrics.snapshot()["total_cost_usd"] == pytest.approx(0.60)


@pytest.mark.asyncio
async def test_async_log_success_event_records_the_same_shape():
    logger = metrics.LiteLLMMetricsLogger()
    kwargs, response = _kwargs(cost_usd=0.007, prompt_tokens=12, completion_tokens=3)
    await logger.async_log_success_event(kwargs, response, *_times(33))
    snap = metrics.snapshot()
    assert snap["call_count"] == 1
    assert snap["total_cost_usd"] == pytest.approx(0.007)
    assert snap["tokens_in"] == 12
    assert snap["tokens_out"] == 3
    assert snap["p50_latency_ms"] == pytest.approx(33.0)


def test_malformed_event_does_not_raise():
    logger = metrics.LiteLLMMetricsLogger()
    logger.log_success_event(None, None, None, None)
    snap = metrics.snapshot()
    assert snap["call_count"] == 1
    assert snap["total_cost_usd"] == 0.0
    assert snap["by_model"] == {"unknown": 0.0}


def test_cost_falls_back_to_standard_logging_object():
    logger = metrics.LiteLLMMetricsLogger()
    kwargs = {
        "model": "gpt-4o-mini",
        "response_cost": 0.0,
        "standard_logging_object": {
            "model": "gpt-4o-mini",
            "prompt_tokens": 9,
            "completion_tokens": 2,
            "cost_breakdown": {"total_cost": 0.0042},
        },
    }
    logger.log_success_event(kwargs, SimpleNamespace(), *_times(1))
    snap = metrics.snapshot()
    assert snap["total_cost_usd"] == pytest.approx(0.0042)
    assert snap["tokens_in"] == 9
    assert snap["tokens_out"] == 2


def test_session_model_usage_fallback_when_buffer_is_empty(monkeypatch):
    rows = [
        SimpleNamespace(model="gpt-4o-mini", tokens_in=200, tokens_out=80, cost_usd=0.012),
        SimpleNamespace(
            model="text-embedding-3-small", tokens_in=40, tokens_out=0, cost_usd=0.0004
        ),
    ]
    monkeypatch.setattr(metrics, "_read_session_model_usage", lambda: rows)

    snap = metrics.snapshot()
    assert snap["call_count"] == 0
    assert snap["p50_latency_ms"] == 0.0
    assert snap["by_phase"] == {}
    assert snap["tokens_in"] == 240
    assert snap["tokens_out"] == 80
    assert snap["total_cost_usd"] == pytest.approx(0.0124)
    assert snap["by_model"]["gpt-4o-mini"] == pytest.approx(0.012)
    assert snap["by_model"]["text-embedding-3-small"] == pytest.approx(0.0004)


def test_session_model_usage_schema_is_importable():
    from cognee.modules.session_lifecycle.models import SessionModelUsage

    assert SessionModelUsage.__tablename__ == "session_model_usage"
    for column in ("tokens_in", "tokens_out", "cost_usd", "model"):
        assert hasattr(SessionModelUsage, column)


@pytest.mark.asyncio
async def test_phase_label_is_isolated_across_concurrent_tasks():
    logger = metrics.LiteLLMMetricsLogger()

    async def run(name: str, cost: float) -> None:
        with metrics.phase(name):  # type: ignore[arg-type]
            await asyncio.sleep(0.01)
            kwargs, response = _kwargs(cost_usd=cost)
            logger.log_success_event(kwargs, response, *_times(1))

    await asyncio.gather(
        run("ingest", 0.10),
        run("graph_recall", 0.03),
        run("naive_embed", 0.01),
    )
    snap = metrics.snapshot()
    assert snap["by_phase"]["ingest"] == pytest.approx(0.10)
    assert snap["by_phase"]["graph_recall"] == pytest.approx(0.03)
    assert snap["by_phase"]["naive_embed"] == pytest.approx(0.01)


def test_buffer_events_win_over_the_fallback(monkeypatch):
    monkeypatch.setattr(
        metrics,
        "_read_session_model_usage",
        lambda: [SimpleNamespace(model="other", tokens_in=999, tokens_out=999, cost_usd=9.99)],
    )
    logger = metrics.LiteLLMMetricsLogger()
    kwargs, response = _kwargs(cost_usd=0.001, prompt_tokens=1, completion_tokens=1)
    logger.log_success_event(kwargs, response, *_times(4))

    snap = metrics.snapshot()
    assert snap["call_count"] == 1
    assert snap["total_cost_usd"] == pytest.approx(0.001)
    assert "other" not in snap["by_model"]
