"""
report.py

Formats final graph state as readable text or JSON.
"""

from __future__ import annotations

import json

from src.state import AgentState


def to_json(state: AgentState, indent: int = 2) -> str:
    return json.dumps(dict(state), indent=indent)


def to_text(state: AgentState) -> str:
    lines = [
        f"=== Multi-Agent Financial Report: {state.get('ticker')} ===",
        f"Query: {state.get('query')}",
        f"Steps taken: {' -> '.join(state.get('steps_taken', []))}",
        "",
        "--- Sentiment ({}) ---".format(state.get("sentiment_source")),
        f"Score: {state.get('sentiment_score')}  |  Label: {state.get('sentiment_label')}",
        "",
        "--- Research ({}) ---".format(state.get("research_source")),
        str(state.get("research_answer")),
        "",
        "--- Forecast ({}) ---".format(state.get("forecast_source")),
        f"Direction: {state.get('forecast_direction')}  |  Confidence: {state.get('forecast_confidence')}  |  Baseline: {state.get('forecast_baseline_direction')}",
        "",
        "--- Bull case ---",
        str(state.get("bull_case")),
        "",
        "--- Bear case ---",
        str(state.get("bear_case")),
        "",
        "--- Judge's verdict (confidence: {}) ---".format(state.get("final_confidence")),
        str(state.get("final_verdict")),
    ]
    return "\n".join(lines)
