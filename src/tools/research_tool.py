"""
research_tool.py

Mock-first wrapper around rag-financial-qa. Same pattern as
finagent's src/tools/rag_tool.py — returns a plain dict for the
LangGraph state update.

To wire up the real system, replace the body of `get_research()` with
a call into rag-financial-qa's query entrypoint (see finagent's
rag_tool.py for the exact wiring pattern).
"""

from __future__ import annotations


def get_research(ticker: str, query: str) -> dict:
    """Returns a dict with research_answer, research_citations, research_source."""
    answer = (
        f"[MOCK] Retrieved context for {ticker} regarding: '{query}'. "
        f"Recent filings and commentary show a mixed picture — no single "
        f"dominant narrative. Replace this with a real answer once "
        f"rag-financial-qa is wired up."
    )
    citations = [
        f"[MOCK] {ticker} 10-Q filing, most recent quarter",
        f"[MOCK] {ticker} earnings call transcript",
    ]
    return {
        "research_answer": answer,
        "research_citations": citations,
        "research_source": "rag-financial-qa (MOCK)",
    }
