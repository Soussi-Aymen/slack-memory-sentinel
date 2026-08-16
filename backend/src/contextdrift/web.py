"""FastAPI + htmx UI: side-by-side recall vs naive search, live cost monitor."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from functools import wraps
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
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

SLACK_ENV_KEYS = (
    "SLACK_CLIENT_ID",
    "SLACK_CLIENT_SECRET",
    "SLACK_SIGNING_SECRET",
    "SLACK_REDIRECT_URI",
    "SLACK_FRONTEND_BASE_URL",
    "INTEGRATION_CREDENTIALS_KEY",
)

Handler = Callable[..., Awaitable[Response]]


def _env_configured(name: str) -> bool:
    value = os.environ.get(name, "").strip()
    if not value:
        return False
    return "YOUR_NGROK" not in value and not value.endswith("_here")


def slack_status() -> dict[str, bool]:
    """Which Slack/OAuth env vars are set. Values are never returned."""
    return {name: _env_configured(name) for name in SLACK_ENV_KEYS}


def slack_ready() -> bool:
    return all(slack_status().values())


def _frontend_root() -> Path:
    container = Path("/app/frontend")
    if (container / "templates").is_dir():
        return container
    return Path(__file__).resolve().parents[3] / "frontend"


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


async def run_startup() -> None:
    """Register Cognee settings and apply 1.5 schema migrations.

    Tests monkeypatch this so the htmx shell never touches sqlite or Qdrant.
    """
    configure_cognee()
    from cognee.run_migrations import run_migrations

    await run_migrations()


def _mount_slack_routes(app: FastAPI) -> None:
    """Official Cognee Slack + OAuth routers (1.5.0.dev1)."""
    import cognee.modules.integrations.slack  # noqa: F401 — registers SlackIntegration
    from cognee.api.v1.integrations.routers import get_integrations_router
    from cognee.api.v1.slack.routers import get_slack_channels_router, get_slack_router

    app.include_router(get_slack_router(), prefix="/api/v1/slack", tags=["slack"])
    app.include_router(get_slack_channels_router(), prefix="/api/v1/slack", tags=["slack"])
    app.include_router(
        get_integrations_router(),
        prefix="/api/v1/integrations",
        tags=["integrations"],
    )


def create_app() -> FastAPI:
    frontend = _frontend_root()
    templates = Jinja2Templates(directory=str(frontend / "templates"))
    templates.env.filters["usd"] = _usd

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        register_metrics()
        await run_startup()
        yield

    app = FastAPI(title="ContextDrift", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=str(frontend / "static")), name="static")
    _mount_slack_routes(app)

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
                "presets": PRESET_QUERIES,
                "default_query": PRESET_QUERIES[0][1],
            },
        )

    @app.get("/integrations", response_class=HTMLResponse)
    @catch_errors
    async def integrations(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "integrations.html",
            {
                "status": _status(),
                "slack_env": slack_status(),
                "slack_ready": slack_ready(),
                "slack_outcome": request.query_params.get("slack", ""),
            },
        )

    @app.get("/slack/connect")
    @catch_errors
    async def slack_connect(request: Request) -> Response:
        missing = [name for name, ok in slack_status().items() if not ok]
        if missing:
            return templates.TemplateResponse(
                request,
                "_error.html",
                {
                    "message": (
                        "Slack is not configured. Paste Client ID, Client Secret, "
                        "and Signing Secret from api.slack.com → your app → "
                        "Basic Information into .env, set SLACK_REDIRECT_URI to "
                        "the ngrok callback URL, then restart. Missing: " + ", ".join(missing)
                    )
                },
                status_code=503,
            )
        from cognee.modules.integrations.oauth_flow import make_state
        from cognee.modules.integrations.registry import get_integration
        from cognee.modules.users.methods import get_default_user

        configure_cognee()
        integration = get_integration("slack")
        user = await get_default_user()
        state = make_state(user.id, signing_secret=integration.state_signing_secret())
        return RedirectResponse(integration.authorize_url(state))

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
