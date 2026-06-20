# PokerEval

A poker testbed for frontier-model research engineering.

> End-to-end RL research pipeline that evaluates and improves LLM poker reasoning —
> a `verifiers`-compatible environment, a verifiable/unverifiable eval suite, a
> synthetic adversarial-data flywheel, an RLVR training run on Tinker, and a
> memory-equipped agent served behind a playable demo.

## Status

🚧 Early development. Design spec:
[`docs/superpowers/specs/2026-06-20-poker-llm-research-eng-design.md`](docs/superpowers/specs/2026-06-20-poker-llm-research-eng-design.md)

## Phases

1. **Environment + Eval suite** — poker engine wrapper, graders (verifiable +
   LLM-judge), static eval set, frontier-model leaderboard.
2. **Synthetic data flywheel + RLVR** — adversarial data generation, RLVR training
   of a small open model on Tinker, exploitability/eval improvement.
3. **Agent harness + memory + demo** — session-playing agent with memory and
   self-improvement, served behind an API + playable web UI.

## Stack

OpenSpiel (Kuhn/Leduc + CFR + exploitability) · PokerKit (No-Limit Hold'em) ·
`verifiers` (Prime Intellect) · Tinker · Qwen 2.5 · Weights & Biases · vLLM ·
FastAPI · Next.js
