# ContextDrift — Cross-Channel Slack Memory Sentinel

Engineering context fragments across Slack channels: an outage is reported in `#incident-alerts`, diagnosed in `#backend-dev`, and shipped in `#releases`. Ask a vector search "was the checkout bug fixed?" and it returns the panic, never the fix — because the message announcing the fix shares almost no vocabulary with the question.

ContextDrift builds a typed knowledge graph over Slack history with [Cognee](https://www.cognee.ai/), retrieves through [Qdrant](https://www.qdrant.tech/), and traces the dependency chain end to end: outage to root cause to async migration to the release that resolved it. Every claim carries provenance back to the original Slack message (channel, author, timestamp).

It also detects **drift** — decisions that vector search still ranks highly but that were later reversed. A blocker from three hours ago is not current state, and a memory layer that cannot tell the difference will confidently hand you a stale answer.

![ContextDrift comparing naive Qdrant vector search against Cognee knowledge-graph recall, with live cost per query](docs/screenshot.png)

## Architecture

![ContextDrift architecture: htmx browser, FastAPI + Cognee in Docker, Qdrant, and OpenAI](docs/architecture.png)

```mermaid
flowchart LR
  Data["data/mock_slack_data.json<br/>12 adversarial messages"] --> Ingest["ingest.py<br/>remember(list)"]
  Ingest --> Kuzu["Kuzu graph<br/>entities + relations"]
  Ingest --> Qdrant["Qdrant<br/>DocumentChunk_text"]
  Web["web.py POST /compare"] --> Naive["naive.py<br/>raw cosine top-3"]
  Web --> Graph["search.py<br/>recall GRAPH_COMPLETION"]
  Naive --> Qdrant
  Graph --> Qdrant
  Graph --> Kuzu
  Metrics["metrics.py<br/>litellm callback"] -.captures.-> Ingest
  Metrics -.captures.-> Graph
  Metrics -.captures.-> Naive
  Metrics --> Monitor["GET /metrics<br/>hx-trigger every 2s"]
```

The corpus is Slack-shaped (`data/mock_slack_data.json`, 12 messages). Answers also land in Slack through Cognee's official app (`/cognee-ask`, `/cognee-remember`, `/cognee-link`) once the workspace is connected at `/integrations`. The Qdrant community adapter still declares `cognee==1.4.2`; a uv `override-dependencies` pin to `cognee==1.5.0.dev1` is what makes that official Slack stack available — empirically `register` + `remember` + `recall` round-trip on 1.5.0.dev1. Bulk channel history is still our ingest pipeline: the official app omits `channels:history`.

## Compose topology

```mermaid
flowchart LR
  subgraph compose [docker compose]
    Backend["backend :8000<br/>uvicorn --reload"]
    QdrantSvc["qdrant :6333"]
  end
  Backend -->|"http://qdrant:6333"| QdrantSvc
  QdrantSvc -.->|bind mount| Storage["./qdrant_storage"]
  Backend -.->|"ro mount"| DataDir["./data"]
  Backend -.->|"ro mount"| FrontDir["./frontend"]
  Backend -.->|named volumes| Vols["cognee_system<br/>cognee_data"]
```

Inside the backend container Qdrant is `http://qdrant:6333`, not `localhost`. `.env.example` ships `VECTOR_DB_URL=http://qdrant:6333` for the containerized path, with a commented `http://localhost:6333` for host-side runs. A wrong value here surfaces as an adapter error and reads like a Cognee bug.

## Quickstart

Requires Docker Compose and an OpenAI API key (`gpt-4o-mini` + `text-embedding-3-small`).

```bash
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. Leave `VECTOR_DB_URL=http://qdrant:6333`.

```bash
docker compose up --build
```

Open [http://localhost:8000](http://localhost:8000). Click **Ingest corpus**, wait for the one-shot `remember()` pipeline to finish, then **Compare** on the preset query `was the checkout 504 bug fixed?`.

Native Slack (optional): copy `slack-app-manifest.yml`, replace `YOUR_NGROK_HOST`, create the app at api.slack.com, fill the `SLACK_*` keys in `.env`, run `ngrok http 8000`, then open [http://localhost:8000/integrations](http://localhost:8000/integrations) and **Connect Slack**. In the workspace run `/cognee-link`, then `/cognee-ask was the checkout 504 bug fixed?`.

Optional CLI ingest (same pipeline, no UI):

```bash
docker compose run --rm backend uv run python -m contextdrift.ingest
```

Qdrant answers on [http://localhost:6333/readyz](http://localhost:6333/readyz). The app healthcheck is [http://localhost:8000/health](http://localhost:8000/health).

## Why the baseline is a real Qdrant query

The left panel is not a simulated straw man. `naive.py` embeds the query with `text-embedding-3-small` and hits the same Qdrant `DocumentChunk_text` collection Cognee wrote during ingest, returning cosine top-3 with scores.

Graph traversal is the only variable: both panels share the collection and the embedding model. The adversarial corpus is built so lexical overlap points at the incident report, a marketing-site 504 decoy, and a superseded blocker — never at the lexically disjoint hotfix in `#releases`.

## How cost is measured

Every Cognee LLM call routes through LiteLLM. `metrics.py` registers a `CustomLogger` success callback that records model, tokens in/out, cost in USD, and latency into an in-process ring buffer tagged by phase (`ingest`, `graph_recall`, `naive_embed`). Cognee's `SessionModelUsage` table is the fallback if no callback events arrive.

The number that sells the pitch is **cost per correct answer** — undefined for the baseline, because its numerator is zero. Panels show absolute spend per query side by side, plus the one-time ingest cost. Qdrant is self-hosted, so retrieval is $0 marginal and 100% of spend is LLM tokens.

The monitor fragment at `GET /metrics` is polled every 2s via `hx-trigger`.

## 2-minute demo script

Rehearse once with a timer. Bracketed values are filled from the live `/metrics` panel.

**0:00–0:20 — The problem.** "An outage gets reported in `#incident-alerts`, diagnosed in `#backend-dev`, and shipped in `#releases`. Three channels, three days, one story. Ask any vector search 'was the checkout bug fixed?' and it hands you the panic, never the fix — because the fix doesn't share a single keyword with the question."

**0:20–0:50 — The baseline, live.** Run the preset query. Point at the left panel. "This is real Qdrant search, not a straw man — same collection, same embedding model as our system, top three by cosine. Result one is the original 504 report. Result two is a *different* 504, on the marketing site, totally unrelated. Result three is Marcus saying he's blocking the release and rolling back. Every hit is lexically perfect and the answer is wrong — and note the third one is a decision that was reversed three hours later."

**0:50–1:25 — ContextDrift.** Point right. "Same query, same Qdrant, plus Cognee's knowledge graph. It traces outage to root cause to the async migration to the release that shipped it, and tells you it's resolved by PR #405. Every claim carries a provenance chip back to the Slack message — channel, author, timestamp. And it flags Marcus's blocker as superseded, because the graph knows a later decision overrode it. That's the drift: your memory said 'blocked' long after reality said 'shipped.'"

**1:25–1:50 — Cost.** Switch to the monitoring panel. "We instrumented every LLM call through litellm. Naive search costs [X] per query. ContextDrift costs [Y]. Yes, that's [N] times more — and it's the only one that's right, so cost per *correct* answer for the baseline is undefined. Qdrant is self-hosted, so retrieval is $0 marginal; every cent here is tokens. One-time ingest for the whole corpus was [Z]."

**1:50–2:00 — Close.** "Point it at Jira or your codebase and nothing changes but the loader. The memory layer is the product."

If asked why the Qdrant adapter declared `cognee==1.4.2`: a declared pin is not proof of incompatibility. We forced `cognee==1.5.0.dev1` with uv `override-dependencies` and verified `register` + `remember` + `recall` against a scratch Qdrant collection.

## Future work

- **LangChain / LangGraph / LangSmith.** Cognee already owns orchestration; a state machine around a single `recall()` adds no signal. LiteLLM already yields model, tokens, cost, and latency, so LangSmith would duplicate the callback with an extra API key and network dependency.
- **Loaders** for Jira and the codebase — the graph and cost tracker stay the same.
