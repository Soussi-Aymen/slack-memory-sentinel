"""httpx tests for the htmx shell. Search is mocked — no Qdrant or LLM."""

from __future__ import annotations

from fastapi.testclient import TestClient

from contextdrift.web import create_app, slack_ready, slack_status

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


def _client(monkeypatch, *, naive=_unused_search, graph=_unused_search) -> TestClient:
    monkeypatch.setattr("contextdrift.web.qdrant_reachable", lambda: True)
    monkeypatch.setattr("contextdrift.web.cognee_configured", lambda: True)
    monkeypatch.setattr("contextdrift.web.has_api_key", lambda: True)
    monkeypatch.setattr("contextdrift.web.register_metrics", lambda: None)
    monkeypatch.setattr("contextdrift.web.run_startup", _noop_startup)
    monkeypatch.setattr("contextdrift.web.snapshot", lambda: EMPTY_SNAP)
    monkeypatch.setattr("contextdrift.web.naive_vector_search", naive)
    monkeypatch.setattr("contextdrift.web.query_slack_memory", graph)
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
    body = response.text
    assert "SLACK_SIGNING_SECRET" in body
    assert "missing" in body
    assert "Slack is not configured" in body
    assert 'href="/slack/connect"' not in body


def test_integrations_ready_shows_connect(monkeypatch):
    monkeypatch.setenv("SLACK_CLIENT_ID", "cid")
    monkeypatch.setenv("SLACK_CLIENT_SECRET", "csecret")
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "ssecret")
    monkeypatch.setenv(
        "SLACK_REDIRECT_URI",
        "https://example.ngrok-free.app/api/v1/integrations/slack/callback",
    )
    monkeypatch.setenv("SLACK_FRONTEND_BASE_URL", "http://localhost:8000")
    monkeypatch.setenv("INTEGRATION_CREDENTIALS_KEY", "dGVzdGtleXRlc3RrZXl0ZXN0a2V5dGVzdA==")
    with _client(monkeypatch) as client:
        response = client.get("/integrations")
    assert response.status_code == 200
    assert 'href="/slack/connect"' in response.text
    assert "Slack is not configured" not in response.text


def test_slack_connect_without_secret_explains(monkeypatch):
    with _client(monkeypatch) as client:
        response = client.get("/slack/connect")
    assert response.status_code == 503
    assert "SLACK_SIGNING_SECRET" in response.text


def test_slack_commands_route_is_mounted(monkeypatch):
    with _client(monkeypatch) as client:
        response = client.post("/api/v1/slack/commands")
    assert response.status_code != 404
    assert response.status_code in (401, 422, 500)


def test_slack_status_treats_placeholders_as_missing(monkeypatch):
    monkeypatch.setenv("SLACK_CLIENT_ID", "cid")
    monkeypatch.setenv("SLACK_CLIENT_SECRET", "")
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "   ")
    monkeypatch.setenv(
        "SLACK_REDIRECT_URI",
        "https://YOUR_NGROK_HOST/api/v1/integrations/slack/callback",
    )
    monkeypatch.setenv("SLACK_FRONTEND_BASE_URL", "http://localhost:8000")
    monkeypatch.setenv("INTEGRATION_CREDENTIALS_KEY", "key_here")
    status = slack_status()
    assert status["SLACK_CLIENT_ID"] is True
    assert status["SLACK_CLIENT_SECRET"] is False
    assert status["SLACK_SIGNING_SECRET"] is False
    assert status["SLACK_REDIRECT_URI"] is False
    assert status["SLACK_FRONTEND_BASE_URL"] is True
    assert status["INTEGRATION_CREDENTIALS_KEY"] is False
    assert slack_ready() is False


def test_compare_renders_both_panels(monkeypatch):
    async def fake_naive(query: str, top_k: int = 3):
        assert query == "was the checkout 504 bug fixed?"
        return {
            "results": [
                {
                    "text": "Checkout is returning 504s, ticket CHECKOUT-504.",
                    "score": 0.93,
                    "channel": "incident-alerts",
                    "user": "@alex",
                }
            ],
            "cost_usd": 0.0004,
            "latency_ms": 12.0,
            "error": None,
        }

    async def fake_graph(query: str):
        assert query == "was the checkout 504 bug fixed?"
        return {
            "graph_answer": "Resolved by hotfix PR #405.",
            "evidence_chunks": [
                {
                    "text": "hotfix PR #405 merged and deployed to production.",
                    "channel": "releases",
                    "user": "@jordan",
                    "timestamp": "2026-08-12T16:40:11Z",
                }
            ],
            "cost_usd": 0.007,
            "latency_ms": 40.0,
            "error": None,
        }

    with _client(monkeypatch, naive=fake_naive, graph=fake_graph) as client:
        response = client.post("/compare", data={"query": "was the checkout 504 bug fixed?"})
    assert response.status_code == 200
    body = response.text
    assert "Naive Vector Search" in body
    assert "incident-alerts" in body
    assert "CHECKOUT-504" in body
    assert "PR #405" in body
    assert "releases" in body
    assert "$0.0004" in body
    assert "$0.0070" in body


def test_compare_empty_query_uses_the_preset(monkeypatch):
    seen: list[str] = []

    async def fake_naive(query: str, top_k: int = 3):
        seen.append(query)
        return {"results": [], "cost_usd": 0.0, "latency_ms": 1.0, "error": None}

    async def fake_graph(query: str):
        seen.append(query)
        return {
            "graph_answer": "",
            "evidence_chunks": [],
            "cost_usd": 0.0,
            "latency_ms": 1.0,
            "error": None,
        }

    with _client(monkeypatch, naive=fake_naive, graph=fake_graph) as client:
        response = client.post("/compare", data={"query": "  ", "preset": ""})
    assert response.status_code == 200
    assert seen == ["was the checkout 504 bug fixed?", "was the checkout 504 bug fixed?"]
    assert "No hits." in response.text


def test_compare_renders_naive_error_panel(monkeypatch):
    async def fake_naive(query: str, top_k: int = 3):
        return {
            "results": [],
            "cost_usd": 0.0,
            "latency_ms": 8.0,
            "error": 'Wrong input: Not existing vector name error: ""',
        }

    async def fake_graph(query: str):
        return {
            "graph_answer": "",
            "evidence_chunks": [],
            "cost_usd": 0.0,
            "latency_ms": 8.0,
            "error": None,
        }

    with _client(monkeypatch, naive=fake_naive, graph=fake_graph) as client:
        response = client.post("/compare", data={"query": "was the checkout 504 bug fixed?"})
    assert response.status_code == 200
    assert "Not existing vector name" in response.text
    assert "ContextDrift Memory Sentinel" in response.text


def test_compare_unexpected_raise_is_error_fragment(monkeypatch):
    async def boom(query: str, top_k: int = 3):
        raise RuntimeError("compare exploded")

    async def fake_graph(query: str):
        return {
            "graph_answer": "",
            "evidence_chunks": [],
            "cost_usd": 0.0,
            "latency_ms": 0.0,
            "error": None,
        }

    with _client(monkeypatch, naive=boom, graph=fake_graph) as client:
        response = client.post("/compare", data={"query": "x"})
    assert response.status_code == 500
    assert "compare exploded" in response.text


def test_ingest_fragment(monkeypatch):
    async def fake_ingest(progress_callback=None):
        if progress_callback:
            progress_callback("configured")
            progress_callback("remembered")
        return {"message_count": 12, "cost_usd": 0.042, "duration_s": 1.5}

    monkeypatch.setattr("contextdrift.web.run_ingestion", fake_ingest)
    with _client(monkeypatch) as client:
        response = client.post("/ingest")
    assert response.status_code == 200
    assert "Ingested 12 messages" in response.text
    assert "$0.0420" in response.text
    assert "configured → remembered" in response.text
