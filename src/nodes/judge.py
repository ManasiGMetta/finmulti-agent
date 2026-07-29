"""
judge.py

The judge sees the bull case, bear case, and underlying evidence, and
produces a final verdict plus a self-reported confidence level. This
is where the multi-agent debate actually gets resolved into a single
answer — the judge's job is weighing, not just concatenating.
"""

from __future__ import annotations

from src.llm import get_llm
from src.state import AgentState


def _judge_prompt(state: AgentState) -> str:
    return f"""Ticker: {state.get('ticker')}
User question: {state.get('query')}

Bull case: {state.get('bull_case')}

Bear case: {state.get('bear_case')}

Forecast model prediction: '{state.get('forecast_direction')}' with confidence {state.get('forecast_confidence')}

As the judge, weigh the bull and bear cases against the forecast
confidence and produce a final verdict. Be honest about uncertainty —
do not manufacture false confidence. Respond in 2-4 sentences, then on
a new line state your own confidence in this verdict as exactly one
word: low, medium, or high."""


def judge_node(state: AgentState) -> dict:
    llm = get_llm()
    response = llm.invoke(_judge_prompt(state))
    text = response.content.strip()

    # Try to pull the trailing confidence word off the last line;
    # fall back to "medium" if the LLM didn't follow the format exactly.
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    confidence = "medium"
    verdict = text
    if lines:
        last = lines[-1].lower()
        for level in ("low", "medium", "high"):
            if level in last and len(last) < 20:
                confidence = level
                verdict = "\n".join(lines[:-1]).strip()
                break

    return {
        "final_verdict": verdict,
        "final_confidence": confidence,
        "steps_taken": state.get("steps_taken", []) + ["judge"],
    }
