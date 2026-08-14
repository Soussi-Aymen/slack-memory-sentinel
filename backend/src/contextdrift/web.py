"""FastAPI + htmx UI: side-by-side recall vs naive search, live cost monitor."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from functools import wraps
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import Response

from contextdrift.config import configure_cognee, has_api_key, settings
from contextdrift.ingest import run_ingestion
from contextdrift.metrics import register_metrics, snapshot
from contextdrift.naive import naive_vector_search
from contextdrift.search import query_slack_memory

PRESET_QUERIES = (
    ("checkout-fixed", "was the checkout 504 bug fixed?"),
    ("release-blocked", "is the checkout release still blocked?"),
)

Handler = Callable[..., Awaitable[Response]]


def _frontend_root() -> Path:
    container = Path("/app/frontend")
    if (container / "templates").is_dir():
        return container
    return Path(__file__).resolve().parents[3] / "frontend"


def _load_corpus() -> list[dict[str, Any]]:
    path = Path(settings.data_path)
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []


def qdrant_reachable() -> bool:
    url = settings.vector_db_url.rstrip("/") + "/readyz"
    try:
        with urlopen(url, timeout=1.0) as resp:  # noqa: S310 — operator-controlled VECTOR_DB_URL
            return 200 <= getattr(resp, "status", 200) < 300
    except (URLError, OSError, TimeoutError, ValueError):
        return False


def cognee_configured() -> bool:
    try:
        configure_cognee()
    except Exception:
        return False
    return True


def _status() -> dict[str, bool]:
    return {
        "qdrant": qdrant_reachable(),
        "cognee": cognee_configured(),
        "api_key": has_api_key(),
    }


def _usd(value: float) -> str:
    return f"${value:,.4f}"


def _correct_answer_cost(snap: dict[str, Any]) -> str:
    by_phase = snap.get("by_phase") or {}
    graph = float(by_phase.get("graph_recall") or 0.0)
    if graph <= 0:
        return "—"
    return _usd(graph)


def create_app() -> FastAPI:
    frontend = _frontend_root()
    templates = Jinja2Templates(directory=str(frontend / "templates"))
    templates.env.filters["usd"] = _usd

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        register_metrics()
        yield

    app = FastAPI(title="ContextDrift", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=str(frontend / "static")), name="static")

    def error_fragment(request: Request, exc: BaseException) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "_error.html",
            {"message": str(exc) or type(exc).__name__},
            status_code=500,
        )

    def catch_errors(handler: Handler) -> Handler:
        @wraps(handler)
        async def wrapped(request: Request, *args: Any, **kwargs: Any) -> Response:
            try:
                return await handler(request, *args, **kwargs)
            except Exception as exc:
                return error_fragment(request, exc)

        return wrapped

    @app.get("/health")
    @catch_errors
    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.get("/", response_class=HTMLResponse)
    @catch_errors
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "status": _status(),
                "messages": _load_corpus(),
                "presets": PRESET_QUERIES,
                "default_query": PRESET_QUERIES[0][1],
            },
        )

    @app.post("/compare", response_class=HTMLResponse)
    @catch_errors
    async def compare(
        request: Request,
        query: str = Form(""),
        preset: str = Form(""),
    ) -> HTMLResponse:
        text = (query or "").strip() or (preset or "").strip() or PRESET_QUERIES[0][1]
        naive, graph = await asyncio.gather(
            naive_vector_search(text),
            query_slack_memory(text),
        )
        return templates.TemplateResponse(
            request,
            "_comparison.html",
            {"query": text, "naive": naive, "graph": graph},
        )

    @app.post("/ingest", response_class=HTMLResponse)
    @catch_errors
    async def ingest(request: Request) -> HTMLResponse:
        progress: list[str] = []
        result = await run_ingestion(progress_callback=progress.append)
        return templates.TemplateResponse(
            request,
            "_ingest.html",
            {"result": result, "progress": progress},
        )

    @app.get("/metrics", response_class=HTMLResponse)
    @catch_errors
    async def metrics_fragment(request: Request) -> HTMLResponse:
        snap = snapshot()
        return templates.TemplateResponse(
            request,
            "_metrics.html",
            {
                "snap": snap,
                "correct_answer_cost": _correct_answer_cost(snap),
            },
        )

    return app
