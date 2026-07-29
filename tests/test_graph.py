"""
test_graph.py

Tests run entirely against FakeLLM (no ANTHROPIC_API_KEY required, no
network calls) — this exercises the graph structure, routing logic,
and state merging, not the quality of real LLM reasoning.
"""

from __future__ import annotations

import os

import pytest

# Ensure tests never accidentally hit the real API even if a key is
# present in the environment they're run in.
os.environ.pop("ANTHROPIC_API_KEY", None)

from src.graph import build_graph
from src.llm import FakeLLM, get_llm
from src.report import to_json, to_text


@pytest.fixture
def initial_state():
    return {
        "ticker": "AAPL",
        "query": "Should I be worried about this stock?",
        "steps_taken": [],
        "routing_reasoning": [],
    }


def test_get_llm_falls_back_to_fake_without_api_key():
    llm = get_llm()
    assert isinstance(llm, FakeLLM)


def test_graph_runs_to_completion(initial_state):
    graph = build_graph()
    final_state = graph.invoke(initial_state)
    assert final_state["next_step"] == "finish"


def test_graph_visits_all_specialists_in_order(initial_state):
    graph = build_graph()
    final_state = graph.invoke(initial_state)
    steps = final_state["steps_taken"]
    # sentiment/research/forecast must precede debate, debate before judge
    assert steps.index("sentiment") < steps.index("debate")
    assert steps.index("research") < steps.index("debate")
    assert steps.index("forecast") < steps.index("debate")
    assert steps.index("debate") < steps.index("judge")


def test_graph_populates_all_specialist_data(initial_state):
    graph = build_graph()
    final_state = graph.invoke(initial_state)
    assert final_state["sentiment_score"] is not None
    assert final_state["research_answer"] is not None
    assert final_state["forecast_direction"] in ("up", "down")


def test_graph_produces_bull_and_bear_cases(initial_state):
    graph = build_graph()
    final_state = graph.invoke(initial_state)
    assert final_state["bull_case"]
    assert final_state["bear_case"]
    assert final_state["bull_case"] != final_state["bear_case"]


def test_graph_produces_final_verdict_with_valid_confidence(initial_state):
    graph = build_graph()
    final_state = graph.invoke(initial_state)
    assert final_state["final_verdict"]
    assert final_state["final_confidence"] in ("low", "medium", "high")


def test_graph_respects_max_hop_guard(initial_state):
    # Even if routing were to loop, the graph must terminate.
    graph = build_graph()
    final_state = graph.invoke(initial_state)
    assert len(final_state["steps_taken"]) <= 12


def test_mock_sources_are_clearly_labeled(initial_state):
    graph = build_graph()
    final_state = graph.invoke(initial_state)
    assert "MOCK" in final_state["sentiment_source"]
    assert "MOCK" in final_state["research_source"]
    assert "MOCK" in final_state["forecast_source"]


def test_to_text_contains_all_sections(initial_state):
    graph = build_graph()
    final_state = graph.invoke(initial_state)
    text = to_text(final_state)
    for section in ("Sentiment", "Research", "Forecast", "Bull case", "Bear case", "verdict"):
        assert section in text


def test_to_json_round_trips_ticker(initial_state):
    import json

    graph = build_graph()
    final_state = graph.invoke(initial_state)
    parsed = json.loads(to_json(final_state))
    assert parsed["ticker"] == "AAPL"
