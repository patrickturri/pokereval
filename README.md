# PokerEval

A poker testbed for frontier-model research engineering.

> An end-to-end RL research pipeline that evaluates and improves LLM poker
> reasoning — a typed environment, a verifiable **and** unverifiable eval suite,
> an exact CFR/Nash ground truth on Kuhn & Leduc, and (in later phases) a
> synthetic-data flywheel + RLVR training run.

## Why poker?

Poker is an **imperfect-information** game with an *exact* notion of correct
play on its small variants: solve Kuhn/Leduc with CFR and you have a Nash policy
to grade any decision against, plus **exploitability** — the single number
saying how many chips per round a best-responder could win against a strategy.
That gives a rare combination for evaluating LLM reasoning:

- a **verifiable reward** (probability mass the Nash policy puts on the chosen
  action) with no judge in the loop, and
- a **whole-policy metric** (exploitability) that can't be gamed spot-by-spot,

alongside an **unverifiable** signal (an LLM-judge rubric scoring *reasoning
quality*) so we can study where the soft signal agrees with ground truth.

## What's here today (Phase 1)

One typed contract runs through everything:

```
State  ──render──▶  prompt  ──LLM──▶  text  ──parse──▶  Decision  ──grade──▶  Score
```

| Component | Module | What it does |
|---|---|---|
| Interface | `interface/types.py`, `interface/prompts.py` | Pydantic `State`/`Decision`/`Score`; prompt rendering + robust `ACTION:` parsing |
| Engine | `engine/kuhn_leduc.py` | OpenSpiel wrapper enumerating every Kuhn/Leduc decision node |
| Solver | `solver/nash.py` | CFR → Nash policy, exploitability, deterministic-policy exploitability |
| Graders | `graders/verifiable.py`, `graders/judge.py` | Exact Nash/reference grader + LLM-judge rubric grader |
| Eval | `eval/spots.py`, `eval/runner.py`, `eval/leaderboard.py` | Spot sets, the run loop, Markdown leaderboard |
| CLI | `cli.py` | `pokereval run --variant {kuhn,leduc,holdem}` |

Everything is model-agnostic behind a one-method `LLMClient` protocol, so the
entire pipeline runs **offline with no API keys** against a built-in
`FakeClient`. Swapping in `AnthropicClient`/`OpenAIClient` produces a real
frontier-model leaderboard.

## Quickstart

```bash
pip install -e ".[dev]"          # pure-Python core + pytest
pip install -e ".[openspiel]"    # optional: CFR/exploitability on Kuhn/Leduc

make test                        # full suite; OpenSpiel tests skip if pyspiel absent
make eval-leduc                  # exact-grade a FakeClient on Leduc, print leaderboard
make eval-holdem                 # curated Hold'em spots, no OpenSpiel needed
```

Example (the default credential-free `FakeClient`, which always folds — a useful
sanity baseline):

```
$ python -m pokereval.cli run --variant leduc --iterations 500
| Model | N | Errors | Mean Verifiable | Exact-Match | Mean Judge | Exploitability |
|---|---|---|---|---|---|---|
| fake-fold | 936 | 312 | 0.504 | 0.532 | — | 3.5625 |
```

- **Mean Verifiable** — average Nash probability mass on the chosen action.
- **Exact-Match** — how often the chosen action is the Nash-modal one.
- **Exploitability** — chips/round a best responder wins against the model's
  *whole* policy (lower is better; Nash ≈ 0). The `fake-fold` baseline is
  trivially exploitable, which is exactly what we want a baseline to show.

## Roadmap

1. **Environment + eval suite** *(this phase)* — engine, both graders, exact
   ground truth, reproducible leaderboard. **Next:** run ≥3 real frontier
   models and write up where they systematically misplay.
2. **Synthetic-data flywheel + RLVR** — generate spots, label via CFR, target
   Phase-1 failure modes, RLVR-train a small open model, measure the
   exploitability drop.
3. **Agent harness + memory + served demo** — a session-playing agent with
   memory, behind an API + playable web UI.

Design spec:
[`docs/superpowers/specs/2026-06-20-poker-llm-research-eng-design.md`](docs/superpowers/specs/2026-06-20-poker-llm-research-eng-design.md)
· Phase-1 plan:
[`docs/superpowers/plans/2026-06-20-phase1-environment-eval-suite.md`](docs/superpowers/plans/2026-06-20-phase1-environment-eval-suite.md)

## Stack

OpenSpiel (Kuhn/Leduc + CFR + exploitability) · PokerKit (No-Limit Hold'em) ·
`verifiers` (Prime Intellect) · Tinker · Qwen 2.5 · Weights & Biases · vLLM ·
FastAPI · Next.js
