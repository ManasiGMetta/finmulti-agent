"""
forecast_tool.py

Mock-first wrapper around finforecast-tf. Same pattern as finagent's
src/tools/forecast_tool.py — returns a plain dict for the LangGraph
state update.

To wire up the real model, replace the body of `get_forecast()` with
a call into finforecast-tf's prediction entrypoint (see finagent's
forecast_tool.py for the exact wiring pattern).
"""

from __future__ import annotations

import hashlib
from datetime import date


def get_forecast(ticker: str) -> dict:
    """Returns a dict with forecast_direction, forecast_confidence, forecast_baseline_direction."""
    seed = f"{ticker}-{date.today().isoformat()}-forecast"
    digest = hashlib.sha256(seed.encode()).hexdigest()
    confidence = 0.5 + (int(digest[:4], 16) / 0xFFFF) * 0.3
    predicted = "up" if int(digest[4], 16) % 2 == 0 else "down"
    baseline = "up" if int(digest[5], 16) % 2 == 0 else "down"

    return {
        "forecast_direction": predicted,
        "forecast_confidence": round(confidence, 3),
        "forecast_baseline_direction": baseline,
        "forecast_source": "finforecast-tf (MOCK)",
    }
