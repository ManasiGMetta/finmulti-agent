"""
debate.py

The bull and bear agents both see the exact same evidence (sentiment,
research, forecast) but are prompted to argue opposite conclusions.
This is the clearest "multi-agent" moment in the graph: two distinct
reasoning processes over identical inputs, producing different
framings for the judge to weigh.

Both run in a single graph node here for simplicity (one node, two LLM
calls) rather than as two separate graph nodes — they don't need to see
each other's output, so there's no ordering dependency between them.
"""

from __future__ import annotations

from src.llm import get_llm
from src.state import AgentState


def _evidence_block(state: AgentState) -> str:
    return f"""Ticker: {state.get('ticker')}
Sentiment: {state.get('sentiment_label')} (score {state.get('sentiment_score')})
Research context: {state.get('research_answer')}
Forecast: predicted direction '{state.get('forecast_direction')}' with confidence {state.get('forecast_confidence')}, naive baseline direction '{state.get('forecast_baseline_direction')}'"""


def debate_node(state: AgentState) -> dict:
    llm = get_llm()
    evidence = _evidence_block(state)

    bull_prompt = f"""{evidence}

Argue the bull case (optimistic outlook) for this stock based ONLY on
the evidence above. Be specific about which pieces of evidence support
optimism, and note honestly if the evidence is weak. 2-3 sentences."""

    bear_prompt = f"""{evidence}

Argue the bear case (pessimistic outlook) for this stock based ONLY on
the evidence above. Be specific about which pieces of evidence support
caution, and note honestly if the evidence is weak. 2-3 sentences."""

    bull_response = llm.invoke(bull_prompt)
    bear_response = llm.invoke(bear_prompt)

    return {
        "bull_case": bull_response.content.strip(),
        "bear_case": bear_response.content.strip(),
        "steps_taken": state.get("steps_taken", []) + ["debate"],
    }
