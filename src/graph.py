"""
graph.py

Builds the LangGraph StateGraph. Structure:

    supervisor -> (conditional) -> sentiment  -> supervisor
                                 -> research   -> supervisor
                                 -> forecast   -> supervisor
                                 -> debate     -> supervisor
                                 -> judge      -> supervisor
                                 -> finish     -> END

Every specialist node routes back to the supervisor rather than to a
fixed next step — the supervisor re-evaluates state after each hop and
decides what happens next. This is what makes routing dynamic rather
than a hardcoded pipeline (contrast with finagent's fixed sequence).

A max-hop guard prevents infinite loops if the LLM's routing logic
ever gets stuck (e.g. repeatedly re-running a step) — not expected in
normal operation, but cheap insurance for anything LLM-decided.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.nodes.debate import debate_node
from src.nodes.judge import judge_node
from src.nodes.specialists import forecast_node, research_node, sentiment_node
from src.nodes.supervisor import supervisor_node
from src.state import AgentState

MAX_HOPS = 12


def _route_from_supervisor(state: AgentState) -> str:
    """Conditional edge: read next_step and also enforce the hop guard."""
    if len(state.get("steps_taken", [])) >= MAX_HOPS:
        return "finish"
    return state.get("next_step", "finish")


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("sentiment", sentiment_node)
    graph.add_node("research", research_node)
    graph.add_node("forecast", forecast_node)
    graph.add_node("debate", debate_node)
    graph.add_node("judge", judge_node)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        _route_from_supervisor,
        {
            "sentiment": "sentiment",
            "research": "research",
            "forecast": "forecast",
            "debate": "debate",
            "judge": "judge",
            "finish": END,
        },
    )

    # Every specialist reports back to the supervisor for the next decision.
    for node_name in ("sentiment", "research", "forecast", "debate", "judge"):
        graph.add_edge(node_name, "supervisor")

    return graph.compile()
