# finmulti-agent

A genuine multi-agent system, built with **LangGraph**, that dynamically
routes between specialist agents (sentiment, research, forecast), runs
a **bull vs. bear debate**, and has a **judge agent** synthesize a
final verdict — rather than calling tools in a fixed sequence.

This is the architectural step up from [`finagent`](https://github.com/ManasiGMetta/finmulti-agent/blob/finagent): `finagent` is one orchestrator calling three tools in a hardcoded
order. Here, an LLM-driven **supervisor** decides what runs next based
on the current state, and the debate/judge pattern introduces actual
multi-agent coordination — two agents reasoning over the same evidence
toward different conclusions, resolved by a third.

| Repo                                                                     | Type                       | Role                                            |
| ------------------------------------------------------------------------ | -------------------------- | ----------------------------------------------- |
| [`finsent-tf`](https://github.com/ManasiGMetta/finmulti-agent/blob/finsent-tf)             | Discriminative NLP         | Sentiment scoring                               |
| [`rag-financial-qa`](https://github.com/ManasiGMetta/finmulti-agent/blob/rag-financial-qa) | Generative / RAG           | Retrieval-augmented Q&A                         |
| [`finforecast-tf`](https://github.com/ManasiGMetta/finmulti-agent/blob/finforecast-tf)     | Time-series                | Direction forecasting                           |
| [`finagent`](https://github.com/ManasiGMetta/finmulti-agent/blob/finagent)                 | Single-agent orchestration | Fixed-sequence tool calling                     |
| `finmulti-agent` (this repo)                                             | **Multi-agent**            | Dynamic routing + debate + judge, via LangGraph |

## Architecture

```
                 ┌─────────────┐
     ┌──────────▶│ supervisor  │◀──────────┐
     │           │  (LLM router)│           │
     │           └──────┬──────┘           │
     │                  │                   │
     │     decides next step based on       │
     │        current state, then:          │
     │                  │                   │
┌────┴────┬─────────┬───┴────┬────────┬─────┴────┐
│sentiment│ research│forecast│ debate │  judge   │
│ (mock)  │ (mock)  │ (mock) │(LLM x2)│  (LLM)   │
└─────────┴─────────┴────────┴────────┴──────────┘
                                           │
                                         finish
                                           │
                                          END
```

Every specialist node reports back to the supervisor, which
re-evaluates state and picks the next step — including deciding to
finish. This is a **supervisor pattern**, one of the standard
multi-agent topologies (as opposed to a fixed pipeline or a
peer-to-peer handoff pattern).

### The debate step is the core multi-agent moment

The bull and bear agents see identical evidence (sentiment, research,
forecast) but are prompted toward opposite conclusions. The judge then
weighs both — this is meaningfully different from just aggregating
tool outputs, because it forces two separate reasoning passes over the
same facts and a third pass to resolve the disagreement.

## Design choices

- **LLM-based routing.** The supervisor is a real LLM call, not
if/else logic — it reads the current state as a prompt and returns
which node to run next. This is the point of the project: showing
the routing decision can be delegated to a model, not hardcoded.
- **Mock-first data tools, LLM-required reasoning.** Sentiment,
research, and forecast stay mock-first (same pattern as `finagent`)
since they're data lookups, not judgment calls. Routing, debate, and
synthesis genuinely need an LLM — mocking those would defeat the
point of the project — so those three nodes call a real model via
**Groq** (`langchain-groq`), currently `llama-3.3-70b-versatile`.
Groq was chosen deliberately for its free tier, so the project can
be run and iterated on without a paid API budget.
- **`FakeLLM` for tests only.** The test suite (`tests/test_graph.py`)
runs against a deterministic `FakeLLM` so CI/tests need zero API key
and zero network access. This tests graph *structure* (routing order,
state merging, termination) — it does not test reasoning quality,
which can only be judged by actually running it with a real key.
- **Max-hop guard.** Since routing is LLM-decided, `src/graph.py` caps
total hops at 12 as cheap insurance against the router looping —
not expected in normal operation, but worth having since the control
flow is no longer fully deterministic.

## Quickstart

```
pip install -r requirements.txt
cp .env.example .env   # then add your GROQ_API_KEY (free tier: https://console.groq.com/keys)

python main.py --ticker AAPL
python main.py --ticker AAPL --query "Should I be worried about this stock?"
python main.py --ticker AAPL --format json
```

`.env` is loaded automatically at runtime via `python-dotenv` — no
need to `export` the key manually in your shell.

**Without an API key**, it still runs — but on `FakeLLM` canned
responses (a warning prints for each LLM-backed node). Useful for
seeing the graph structure or running tests; not for real analysis.

## Testing

```
pytest tests/ -v
```

10 tests, all against `FakeLLM` — no API key or network access
required. Covers: routing order (specialists before debate, debate
before judge), state population, bull/bear divergence, valid judge
confidence levels, hop-guard termination, and output formatting.

## Wiring up real data tools

Same pattern as `finagent`: `src/tools/sentiment_tool.py`, `research_tool.py`, and `forecast_tool.py` are mock-first with a
documented swap-in point for the real `finsent-tf`, `rag-financial-qa`, and `finforecast-tf` repos (see `finagent`'s
tool files for the exact wiring convention this follows).

## Repo structure

```
finmulti-agent/
├── main.py                     # CLI entrypoint
├── src/
│   ├── state.py                  # shared LangGraph state (TypedDict)
│   ├── llm.py                    # real Groq client + FakeLLM for tests
│   ├── graph.py                  # StateGraph wiring + conditional routing
│   ├── report.py                 # text + JSON formatting
│   ├── nodes/
│   │   ├── supervisor.py           # LLM-based dynamic router
│   │   ├── specialists.py          # sentiment/research/forecast nodes
│   │   ├── debate.py               # bull + bear agents
│   │   └── judge.py                # synthesis + final verdict
│   └── tools/
│       ├── sentiment_tool.py       # mock-first, finsent-tf wiring point
│       ├── research_tool.py        # mock-first, rag-financial-qa wiring point
│       └── forecast_tool.py        # mock-first, finforecast-tf wiring point
├── tests/
│   └── test_graph.py             # 10 tests against FakeLLM
├── requirements.txt
├── .env.example
└── README.md
```

## Honest limitations

- The supervisor's routing, while genuinely LLM-driven, is prompted
toward a fairly linear path (specialists → debate → judge) by
design — it's not exploring wildly different strategies per query.
A more ambitious version would let the supervisor skip steps (e.g.,
skip forecast entirely if sentiment is already decisive) or call a
specialist twice; the current prompt allows this but doesn't
strongly encourage it.
- The judge's confidence level is self-reported by the LLM, not
calibrated against any ground truth — treat it as the model's stated
uncertainty, not a validated probability.
- As with `finforecast-tf`, nothing here should be read as investment
advice; this is an architecture demonstration, not a trading system.

## Possible extensions

- Let the supervisor genuinely skip or reorder steps based on early
signal (e.g., strongly negative sentiment triggers an early bear-only
path)
- Add a second debate round where bull/bear see each other's argument
and respond, before the judge weighs in
- Stream intermediate steps to a UI so the routing decisions are
visible as they happen, not just in the final report
- Add LangSmith or similar tracing to inspect actual routing decisions
across many runs, not just the final state
