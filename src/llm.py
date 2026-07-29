"""
llm.py

LLM access point for every node that needs real reasoning (supervisor
routing, bull/bear debate, judge synthesis). Sentiment/research/forecast
stay mock-first data tools (see src/tools/) — this file is specifically
for the nodes where an LLM's judgment is the actual point.

Two implementations:
  - get_real_llm()  -> ChatAnthropic, requires ANTHROPIC_API_KEY
  - FakeLLM         -> canned, deterministic responses for tests/CI,
                       so the test suite runs with zero API key and
                       zero network calls.

get_llm() picks the real client if ANTHROPIC_API_KEY is set, else
warns and falls back to FakeLLM. This mirrors finagent's mock-first
pattern, but for LLM calls instead of data tools.
"""

from __future__ import annotations

import os
import warnings


class FakeLLM:
    """
    Deterministic stand-in for a real chat model. Returns a canned
    response based on simple keyword matching against the prompt, so
    graph logic (routing, debate structure, synthesis) can be tested
    without any API key or network access.
    """

    def invoke(self, prompt: str):
        text = self._respond(prompt)
        return _FakeResponse(text)

    @staticmethod
    def _respond(prompt: str) -> str:
        p = prompt.lower()
        if "which specialist" in p:
            # Supervisor routing prompt: walk through steps in a fixed
            # order based on what's still missing, so the fake behaves
            # like a real (if unimaginative) router for graph testing.
            if "sentiment_score: none" in p:
                return "sentiment"
            if "research_answer: missing" in p:
                return "research"
            if "forecast_direction: none" in p:
                return "forecast"
            if "bull_case: missing" in p or "bear_case: missing" in p:
                return "debate"
            if "final_verdict: missing" in p:
                return "judge"
            return "finish"
        # NOTE: check judge/synthesis BEFORE bull/bear — the judge prompt
        # contains the literal substrings "bull case" and "bear case"
        # (it quotes both arguments back), so it would otherwise be
        # misrouted into the bull/bear branch below.
        if "as the judge" in p or "final verdict" in p:
            return (
                "[FAKE] Weighing both cases and the forecast confidence, "
                "the evidence is mixed; a moderate-confidence view is "
                "warranted rather than a strong conviction either way.\n"
                "medium"
            )
        if "argue the bull case" in p:
            return (
                "[FAKE] The optimistic case rests on positive sentiment "
                "momentum and constructive retrieved context; if the "
                "forecast confirms an upward direction, conviction is higher."
            )
        if "argue the bear case" in p:
            return (
                "[FAKE] The pessimistic case highlights any negative "
                "sentiment signal, mixed or cautionary retrieved context, "
                "and forecast uncertainty as reasons for caution."
            )
        return "[FAKE] No matching canned response for this prompt."


class _FakeResponse:
    """Mimics the .content attribute LangChain chat model responses expose."""

    def __init__(self, content: str):
        self.content = content


def get_real_llm(model: str = "claude-sonnet-4-6", temperature: float = 0.3):
    """
    Build a real ChatAnthropic client. Requires ANTHROPIC_API_KEY to be
    set in the environment. Raises if the key is missing so callers can
    fall back cleanly.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set.")

    from langchain_anthropic import ChatAnthropic

    return ChatGoogleGenerativeAI(model=model, temperature=temperature)


def get_llm():
    """
    Return a real LLM if ANTHROPIC_API_KEY is set, else fall back to
    FakeLLM with a warning. This is the single call site every node
    should use — keeps the real/fake decision in one place.
    """
    try:
        return get_real_llm()
    except RuntimeError:
        warnings.warn(
            "ANTHROPIC_API_KEY not set — using FakeLLM. Routing, debate, "
            "and synthesis will use canned responses, not real reasoning. "
            "Set ANTHROPIC_API_KEY to run this agent for real.",
            stacklevel=2,
        )
        return FakeLLM()
