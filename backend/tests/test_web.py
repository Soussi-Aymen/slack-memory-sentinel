"""httpx tests for the htmx shell. Search is mocked — no Qdrant or LLM."""

from __future__ import annotations

from fastapi.testclient import TestClient

from contextdrift.web import create_app

EMPTY_SNAP = {
    "total_cost_usd": 0.0,
    "by_phase": {},
    "by_model": {},
    "tokens_in": 0,
    "tokens_out": 0,
    "p50_latency_ms": 0.0,
    "call_count": 0,
}


async def _unused_search(_query: str, top_k: int = 3):
    raise AssertionError("search must not run in shell tests")


async def _noop_startup() -> None:
    return None


def _client(monkeypatch) -> TestClient:
    monkeypatch.setattr("contextdrift.web.qdrant_reachable", lambda: True)
    monkeypatch.setattr("contextdrift.web.cognee_configured", lambda: True)
    monkeypatch.setattr("contextdrift.web.has_api_key", lambda: True)
    monkeypatch.setattr("contextdrift.web.register_metrics", lambda: None)
    monkeypatch.setattr("contextdrift.web.run_startup", _noop_startup)
    monkeypatch.setattr("contextdrift.web.snapshot", lambda: EMPTY_SNAP)
    monkeypatch.setattr("contextdrift.web.naive_vector_search", _unused_search)
    monkeypatch.setattr("contextdrift.web.query_slack_memory", _unused_search)
    return TestClient(create_app())


def test_health_ok(monkeypatch):
    with _client(monkeypatch) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_renders_status_and_presets(monkeypatch):
    with _client(monkeypatch) as client:
        response = client.get("/")
    assert response.status_code == 200
    body = response.text
    assert "Qdrant up" in body
    assert "Cognee configured" in body
    assert "API key present" in body
    assert "was the checkout 504 bug fixed?" in body
    assert "is the checkout release still blocked?" in body
    assert 'hx-post="/compare"' in body
    assert 'hx-get="/metrics"' in body
    assert 'hx-trigger="load, every 2s"' in body
    assert 'src="/static/htmx.min.js"' in body
    assert "cdn.jsdelivr" not in body
    assert "unpkg.com" not in body
    assert "channel-preview" not in body
    assert "slack-msg" not in body
    assert "analysis console" in body.lower() or "knowledge-graph recall" in body


def test_metrics_fragment(monkeypatch):
    snap = {
        **EMPTY_SNAP,
        "total_cost_usd": 0.12,
        "by_phase": {"ingest": 0.1, "graph_recall": 0.02},
        "by_model": {"gpt-4o-mini": 0.12},
        "tokens_in": 80,
        "tokens_out": 20,
        "p50_latency_ms": 40.0,
        "call_count": 3,
    }
    monkeypatch.setattr("contextdrift.web.qdrant_reachable", lambda: True)
    monkeypatch.setattr("contextdrift.web.cognee_configured", lambda: True)
    monkeypatch.setattr("contextdrift.web.has_api_key", lambda: True)
    monkeypatch.setattr("contextdrift.web.register_metrics", lambda: None)
    monkeypatch.setattr("contextdrift.web.run_startup", _noop_startup)
    monkeypatch.setattr("contextdrift.web.snapshot", lambda: snap)
    monkeypatch.setattr("contextdrift.web.naive_vector_search", _unused_search)
    monkeypatch.setattr("contextdrift.web.query_slack_memory", _unused_search)
    with TestClient(create_app()) as client:
        response = client.get("/metrics")
    assert response.status_code == 200
    body = response.text
    assert "$0.1200" in body
    assert "ingest" in body
    assert "graph_recall" in body
    assert "gpt-4o-mini" in body
    assert "undefined" in body
    assert 'hx-trigger="every 2s"' in body


def test_integrations_page(monkeypatch):
    with _client(monkeypatch) as client:
        response = client.get("/integrations")
    assert response.status_code == 200
    assert "Connect Slack" in response.text
    assert "/slack/connect" in response.text


def test_slack_commands_route_is_mounted(monkeypatch):
    with _client(monkeypatch) as client:
        response = client.post("/api/v1/slack/commands")
    assert response.status_code != 404
    assert response.status_code in (401, 422, 500)
