"""Adversarial Slack corpus: schema, channel coverage, keyword-overlap invariant."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

REQUIRED_FIELDS = ("channel", "user", "timestamp", "text")
REQUIRED_CHANNELS = {
    "incident-alerts",
    "backend-dev",
    "releases",
    "general",
    "random",
}
MESSAGE_COUNT = 12

# The demo query that naive cosine search is supposed to fail on.
DEMO_QUERY = "was the checkout 504 bug fixed?"

STOPWORDS = {
    "was",
    "the",
    "a",
    "an",
    "on",
    "in",
    "to",
    "of",
    "and",
    "is",
    "are",
    "we",
    "why",
    "for",
    "with",
    "this",
    "that",
    "it",
    "yet",
}


def _dataset_path() -> Path:
    container = Path("/app/data/mock_slack_data.json")
    if container.is_file():
        return container
    return Path(__file__).resolve().parents[2] / "data" / "mock_slack_data.json"


def _load_messages() -> list[dict]:
    payload = json.loads(_dataset_path().read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    return payload


def significant_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {token for token in tokens if token not in STOPWORDS and len(token) > 2}


def test_dataset_has_twelve_messages_with_required_schema():
    messages = _load_messages()
    assert len(messages) == MESSAGE_COUNT
    for record in messages:
        assert set(REQUIRED_FIELDS) <= set(record)
        assert record["channel"] in REQUIRED_CHANNELS
        assert record["user"].startswith("@")
        assert record["text"].strip()
        datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))


def test_dataset_covers_all_demo_channels():
    channels = {record["channel"] for record in _load_messages()}
    assert REQUIRED_CHANNELS <= channels


def test_incident_carries_query_keywords_resolution_does_not():
    """Load-bearing invariant: naive retrieval is supposed to miss the fix.

    The incident report shares many significant tokens with the demo query;
    the #releases hotfix shares none. If someone edits the corpus and the
    premise breaks, this test fails before the demo does.
    """
    messages = _load_messages()
    incident = next(
        record
        for record in messages
        if record["channel"] == "incident-alerts" and "CHECKOUT-504" in record["text"]
    )
    resolution = next(record for record in messages if record["channel"] == "releases")
    query_keywords = significant_tokens(DEMO_QUERY)
    incident_overlap = query_keywords & significant_tokens(incident["text"])
    resolution_overlap = query_keywords & significant_tokens(resolution["text"])

    assert query_keywords <= significant_tokens(incident["text"]), (
        f"incident is missing query keywords: {query_keywords - incident_overlap}"
    )
    assert not resolution_overlap, (
        f"resolution leaked query keywords {resolution_overlap}; naive search would find the fix"
    )


def test_lexical_decoy_and_superseded_blocker_are_present():
    messages = _load_messages()
    decoy = next(
        record for record in messages if record["channel"] == "general" and "504" in record["text"]
    )
    assert "marketing" in decoy["text"].lower()

    blocker = next(record for record in messages if record["user"] == "@marcus")
    assert "blocking the release" in blocker["text"]
    assert "sync" in blocker["text"].lower()

    override = next(
        record for record in messages if record["user"] == "@sarah" and "Override" in record["text"]
    )
    assert override["timestamp"] > blocker["timestamp"]
