"""
specialists.py

Thin LangGraph node wrappers around the mock-first data tools
(src/tools/). These nodes don't need an LLM — they just fetch data
and record that the step ran, for the supervisor's audit trail.
"""

from __future__ import annotations

from src.state import AgentState
from src.tools.forecast_tool import get_forecast
from src.tools.research_tool import get_research
from src.tools.sentiment_tool import get_sentiment


def _mark_done(state: AgentState, step_name: str) -> list[str]:
    return state.get("steps_taken", []) + [step_name]


def sentiment_node(state: AgentState) -> dict:
    result = get_sentiment(state["ticker"])
    result["steps_taken"] = _mark_done(state, "sentiment")
    return result


def research_node(state: AgentState) -> dict:
    result = get_research(state["ticker"], state["query"])
    result["steps_taken"] = _mark_done(state, "research")
    return result


def forecast_node(state: AgentState) -> dict:
    result = get_forecast(state["ticker"])
    result["steps_taken"] = _mark_done(state, "forecast")
    return result
