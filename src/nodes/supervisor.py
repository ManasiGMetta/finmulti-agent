"""
supervisor.py

The supervisor is what makes this multi-agent rather than a fixed
pipeline (contrast with finagent, which always runs sentiment -> RAG
-> forecast -> summary in that exact order). Here, an LLM looks at the
current state and decides which node to run next — including deciding
to skip a step, or finish early.

Routing contract: the LLM is asked to respond with exactly one of:
  "sentiment", "research", "forecast", "debate", "judge", "finish"
That single word becomes state["next_step"], which src/graph.py's
conditional edge reads to decide where to go.
"""

from __future__ import annotations

from src.llm import get_llm
from src.state import AgentState

VALID_STEPS = {"sentiment", "research", "forecast", "debate", "judge", "finish"}


def _build_routing_prompt(state: AgentState) -> str:
    done = state.get("steps_taken", [])
    return f"""You are a supervisor routing a financial analysis agent workflow.

Ticker: {state.get('ticker')}
User query: {state.get('query')}

Steps already completed: {done or 'none'}

Available data so far:
- sentiment_score: {state.get('sentiment_score')}
- research_answer: {'present' if state.get('research_answer') else 'missing'}
- forecast_direction: {state.get('forecast_direction')}
- bull_case: {'present' if state.get('bull_case') else 'missing'}
- bear_case: {'present' if state.get('bear_case') else 'missing'}
- final_verdict: {'present' if state.get('final_verdict') else 'missing'}

Decide which specialist should run next. Rules:
- Run "sentiment", "research", and "forecast" before "debate" (debate needs their data)
- Run "debate" before "judge" (judge needs both bull and bear cases)
- Once final_verdict is present, respond "finish"
- Respond with exactly ONE WORD from this set, nothing else: sentiment, research, forecast, debate, judge, finish

Which specialist should run next?"""


def supervisor_node(state: AgentState) -> dict:
    llm = get_llm()
    prompt = _build_routing_prompt(state)
    response = llm.invoke(prompt)

    raw = response.content.strip().lower()
    # Be defensive: LLMs sometimes wrap the answer in extra words.
    next_step = next((w for w in VALID_STEPS if w in raw), "finish")

    reasoning_log = state.get("routing_reasoning", [])
    reasoning_log = reasoning_log + [f"-> {next_step}"]

    return {
        "next_step": next_step,
        "routing_reasoning": reasoning_log,
    }
