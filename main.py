"""
main.py

CLI entrypoint for the multi-agent financial analysis graph.

Usage:
    python main.py --ticker AAPL
    python main.py --ticker AAPL --query "Should I be worried about this stock?"
    python main.py --ticker AAPL --format json

Requires ANTHROPIC_API_KEY in the environment for real LLM-driven
routing/debate/synthesis. Without it, runs on FakeLLM (canned
responses) with a warning — useful for seeing the graph structure and
running tests, not for real analysis.
"""

from __future__ import annotations

import argparse

from src.graph import build_graph
from src.report import to_json, to_text


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-agent financial analysis.")
    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--format", type=str, choices=["text", "json"], default="text")
    return parser.parse_args()


def main():
    args = parse_args()
    query = args.query or f"What is currently driving {args.ticker}'s stock performance?"

    graph = build_graph()
    final_state = graph.invoke(
        {
            "ticker": args.ticker,
            "query": query,
            "steps_taken": [],
            "routing_reasoning": [],
        }
    )

    if args.format == "json":
        print(to_json(final_state))
    else:
        print(to_text(final_state))


if __name__ == "__main__":
    main()
