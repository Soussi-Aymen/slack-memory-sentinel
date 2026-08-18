"""Ingest contract tests. format_message and a mocked pipeline — no LLM calls."""

from __future__ import annotations

import json
from pathlib import Path

import cognee
import pytest

from contextdrift.ingest import format_message, run_ingestion

CORPUS_RECORD = {
    "channel": "backend-dev",
    "user": "@sarah",
    "timestamp": "2026-08-11T10:02:18Z",
    "text": "Moving fraud checks onto an async queue worker.",
}


def test_format_message_matches_frozen_prefix():
    assert format_message(CORPUS_RECORD) == (
        "[Channel: #backend-dev] @sarah (2026-08-11T10:02:18Z): "
        "Moving fraud checks onto an async queue worker."
    )


def test_format_message_roundtrips_a_real_corpus_record():
    container = Path("/app/data/mock_slack_data.json")
    path = (
        container
        if container.is_file()
        else Path(__file__).resolve().parents[2] / "data" / "mock_slack_data.json"
    )
    records = json.loads(path.read_text(encoding="utf-8"))
    formatted = format_message(records[0])
    assert formatted.startswith(f"[Channel: #{records[0]['channel']}] {records[0]['user']} (")
    assert formatted.endswith(f"): {records[0]['text']}")
    assert records[0]["timestamp"] in formatted


@pytest.mark.asyncio
async def test_run_ingestion_single_remember_and_return_keys(monkeypatch, tmp_path):
    corpus = [CORPUS_RECORD, {**CORPUS_RECORD, "channel": "releases", "user": "@jordan"}]
    data_file = tmp_path / "mock_slack_data.json"
    data_file.write_text(json.dumps(corpus), encoding="utf-8")
    monkeypatch.setenv("DATA_PATH", str(data_file))

    calls: dict[str, object] = {}

    async def fake_forget(**kwargs):
        calls["forget"] = kwargs

    async def fake_prune_system(*, metadata=False, **kwargs):
        calls["prune"] = {"metadata": metadata, **kwargs}

    async def fake_remember(data, dataset_name=None):
        calls["remember"] = {"data": data, "dataset_name": dataset_name}

    monkeypatch.setattr("contextdrift.ingest.configure_cognee", lambda: None)
    monkeypatch.setattr("contextdrift.ingest.register_metrics", lambda: None)
    monkeypatch.setattr("contextdrift.ingest.cost_since", lambda _mark: 0.042)
    monkeypatch.setattr(cognee, "forget", fake_forget)
    monkeypatch.setattr(cognee.prune, "prune_system", fake_prune_system)
    monkeypatch.setattr(cognee, "remember", fake_remember)

    progress: list[str] = []
    result = await run_ingestion(progress_callback=progress.append)

    assert tuple(result) == ("message_count", "cost_usd", "duration_s")
    assert result["message_count"] == 2
    assert result["cost_usd"] == pytest.approx(0.042)
    assert result["duration_s"] >= 0.0
    assert calls["forget"] == {"everything": True}
    assert calls["prune"] == {"metadata": True}
    remembered = calls["remember"]
    assert remembered["dataset_name"] == "slack"
    assert remembered["data"] == [format_message(record) for record in corpus]
    assert "configured" in progress
    assert "reset" in progress
    assert "remembered" in progress


@pytest.mark.asyncio
async def test_run_ingestion_empty_corpus_still_remembers_an_empty_batch(monkeypatch, tmp_path):
    data_file = tmp_path / "empty.json"
    data_file.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("DATA_PATH", str(data_file))

    remembered: dict[str, object] = {}

    async def fake_forget(**kwargs):
        return None

    async def fake_prune_system(*, metadata=False, **kwargs):
        return None

    async def fake_remember(data, dataset_name=None):
        remembered["data"] = data
        remembered["dataset_name"] = dataset_name

    monkeypatch.setattr("contextdrift.ingest.configure_cognee", lambda: None)
    monkeypatch.setattr("contextdrift.ingest.register_metrics", lambda: None)
    monkeypatch.setattr("contextdrift.ingest.cost_since", lambda _mark: 0.0)
    monkeypatch.setattr(cognee, "forget", fake_forget)
    monkeypatch.setattr(cognee.prune, "prune_system", fake_prune_system)
    monkeypatch.setattr(cognee, "remember", fake_remember)

    result = await run_ingestion()
    assert result["message_count"] == 0
    assert remembered["data"] == []
    assert remembered["dataset_name"] == "slack"


@pytest.mark.asyncio
async def test_run_ingestion_missing_file_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "missing.json"))
    monkeypatch.setattr("contextdrift.ingest.configure_cognee", lambda: None)
    monkeypatch.setattr("contextdrift.ingest.register_metrics", lambda: None)

    async def fake_forget(**kwargs):
        return None

    async def fake_prune_system(*, metadata=False, **kwargs):
        return None

    monkeypatch.setattr(cognee, "forget", fake_forget)
    monkeypatch.setattr(cognee.prune, "prune_system", fake_prune_system)

    with pytest.raises(FileNotFoundError):
        await run_ingestion()
