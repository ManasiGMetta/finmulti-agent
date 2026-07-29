"""
state.py

Shared state passed between every node in the graph. LangGraph nodes
each receive the full state and return a partial update (a dict of the
keys they changed) which gets merged in.

Keeping this as one flat TypedDict (rather than nested dataclasses)
matches LangGraph's expected state shape and keeps merges simple.
"""

from __future__ import annotations

from typing import Optional, TypedDict


class AgentState(TypedDict, total=False):
    # --- input ---
    ticker: str
    query: str

    # --- supervisor routing ---
    next_step: str                  # which node to run next, set by supervisor
    steps_taken: list[str]          # audit trail of every node that ran
    routing_reasoning: list[str]    # supervisor's stated reasoning at each hop

    # --- specialist outputs (populated as agents run) ---
    sentiment_score: Optional[float]
    sentiment_label: Optional[str]
    sentiment_source: Optional[str]

    research_answer: Optional[str]
    research_citations: Optional[list[str]]
    research_source: Optional[str]

    forecast_direction: Optional[str]
    forecast_confidence: Optional[float]
    forecast_baseline_direction: Optional[str]
    forecast_source: Optional[str]

    # --- debate ---
    bull_case: Optional[str]
    bear_case: Optional[str]

    # --- final output ---
    final_verdict: Optional[str]
    final_confidence: Optional[str]   # "low" / "medium" / "high", judge's own call
