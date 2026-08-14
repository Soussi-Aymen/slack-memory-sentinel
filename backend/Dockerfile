FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies resolve in their own cached layer so editing src never re-syncs
# the whole cognee tree.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-editable

COPY src ./src
# Editable so the bind-mounted src is live and uvicorn --reload suffices.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "contextdrift.web:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "/app/src"]
