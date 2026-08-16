#!/bin/sh
set -e
# Named volumes created by an earlier root run stay root-owned. Fix that,
# then drop to uid 1000 before uvicorn starts.
mkdir -p /app/.cognee_system /app/.data_storage
if [ "$(id -u)" = 0 ]; then
  chown -R 1000:1000 /app/.cognee_system /app/.data_storage
  exec setpriv --reuid=1000 --regid=1000 --init-groups -- "$@"
fi
exec "$@"
