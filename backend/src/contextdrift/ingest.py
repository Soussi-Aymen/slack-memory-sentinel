"""Ingest the Slack corpus into Cognee in a single batched ``remember()`` call."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from contextdrift.config import configure_cognee, settings
from contextdrift.metrics import cost_since, marker, phase, register_metrics

DATASET_NAME = "slack"


def format_message(record: dict) -> str:
    """Render one Slack record as a provenance-prefixed string for Cognee."""
    return (
        f"[Channel: #{record['channel']}] {record['user']} "
        f"({record['timestamp']}): {record['text']}"
    )


def _notify(progress_callback: Callable[[str], Any] | None, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)


async def run_ingestion(progress_callback: Callable[[str], Any] | None = None) -> dict:
    """Reset Cognee, load the Slack corpus, and remember every message in one call.

    Returns keys ``message_count``, ``cost_usd``, ``duration_s``.
    """
    configure_cognee()
    register_metrics()
    _notify(progress_callback, "configured")

    import cognee

    await cognee.forget(everything=True)
    await cognee.prune.prune_system(metadata=True)
    _notify(progress_callback, "reset")

    data_path = Path(settings.data_path)
    records = json.loads(data_path.read_text(encoding="utf-8"))
    messages = [format_message(record) for record in records]
    _notify(progress_callback, f"loaded {len(messages)} messages")

    started = time.perf_counter()
    cost_mark = marker()
    with phase("ingest"):
        await cognee.remember(messages, dataset_name=DATASET_NAME)
    duration_s = time.perf_counter() - started
    _notify(progress_callback, "remembered")

    return {
        "message_count": len(messages),
        "cost_usd": cost_since(cost_mark),
        "duration_s": duration_s,
    }


def main() -> None:
    result = asyncio.run(run_ingestion(progress_callback=print))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
