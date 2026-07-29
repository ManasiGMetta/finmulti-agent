"""
sentiment_tool.py

Mock-first wrapper around finsent-tf, same pattern as finagent's
src/tools/sentiment_tool.py. This one returns a plain dict (LangGraph
state update) instead of a dataclass, to match graph node conventions.

To wire up the real model, replace the body of `get_sentiment()` with
a call into finsent-tf's inference entrypoint (see finagent's
sentiment_tool.py for the exact wiring pattern — same approach here).
"""

from __future__ import annotations

import hashlib
from datetime import date


def get_sentiment(ticker: str) -> dict:
    """Returns a dict with sentiment_score, sentiment_label, sentiment_source."""
    seed = f"{ticker}-{date.today().isoformat()}"
    digest = hashlib.sha256(seed.encode()).hexdigest()
    raw = int(digest[:8], 16) / 0xFFFFFFFF
    score = round((raw * 2) - 1, 3)
    label = "positive" if score > 0.15 else "negative" if score < -0.15 else "neutral"

    return {
        "sentiment_score": score,
        "sentiment_label": label,
        "sentiment_source": "finsent-tf (MOCK)",
    }
