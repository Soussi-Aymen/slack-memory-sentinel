FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_COMPILE_BYTECODE=0 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONUNBUFFERED=1 \
    UV_NO_SYNC=1 \
    PATH="/app/.venv/bin:$PATH" \
    ENABLE_BACKEND_ACCESS_CONTROL=false

WORKDIR /app

# Dependencies resolve in their own cached layer so editing src never re-syncs
# the whole cognee tree. Cognee 1.5.0.dev1 declares LanceDB as its default
# vector store (~430MB of lancedb/pylance/pyarrow); we use Qdrant, and the
# LanceDB adapter is imported only when that provider is selected, so drop it.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-editable \
        --no-install-package lancedb \
        --no-install-package pylance \
        --no-install-package pyarrow \
        --no-install-package lance-namespace \
        --no-install-package lance-namespace-urllib3-client

COPY src ./src
# Editable so the bind-mounted src is live and uvicorn --reload suffices.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen \
        --no-install-package lancedb \
        --no-install-package pylance \
        --no-install-package pyarrow \
        --no-install-package lance-namespace \
        --no-install-package lance-namespace-urllib3-client

RUN useradd --uid 1000 --create-home --home-dir /home/app app \
    && mkdir -p /app/.cognee_system /app/.data_storage \
    && chown -R 1000:1000 /app /home/app

COPY entrypoint.sh /entrypoint.sh
RUN chmod 755 /entrypoint.sh

ENV HOME=/home/app

EXPOSE 8000

# Starts as root only to chown bind-mounted named volumes, then drops to uid 1000.
ENTRYPOINT ["/entrypoint.sh"]
CMD ["uv", "run", "--no-sync", "uvicorn", "contextdrift.web:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "/app/src"]
