# PokerEval

A poker testbed for frontier-model research engineering.

> End-to-end RL research pipeline that evaluates and improves LLM poker reasoning —
> a `verifiers`-compatible environment, a verifiable/unverifiable eval suite, a
> synthetic adversarial-data flywheel, an RLVR training run on Tinker, and a
> memory-equipped agent served behind a playable demo.

## Status

🚧 Phase 1 (environment + eval suite) is implemented and reproducible.
Design spec:
[`docs/superpowers/specs/2026-06-20-poker-llm-research-eng-design.md`](docs/superpowers/specs/2026-06-20-poker-llm-research-eng-design.md) ·
Phase 1 methodology + findings:
[`docs/phase1-findings.md`](docs/phase1-findings.md)

## Quickstart

```bash
make install          # editable install (OpenSpiel + model SDKs)
make test             # full test suite
make leaderboard      # credential-free baseline leaderboard, all variants
```

Run a real frontier model (needs `ANTHROPIC_API_KEY`):

```bash
make leaderboard PROVIDER=anthropic MODEL=claude-opus-4-8
# or per-variant, with the reasoning judge and JSON output for archival:
python -m pokereval.cli run --variant leduc --provider anthropic \
  --model claude-opus-4-8 --judge --format json
```

The leaderboard reports, per model: mean Nash-probability mass, exact-match
rate, whole-policy **exploitability** (the headline metric — chips/hand a
best-responder extracts; 0 = Nash), an optional LLM-judge reasoning score, and
parse-error rate.

### Synthetic data (Phase 2, in progress)

Generate CFR/Nash-labeled spots tagged by failure mode (the spots where
frontier models are predicted to deviate — mixed strategies, facing-a-bet
decisions). These double as verifiable-reward RLVR training data and a targeted
eval set:

```bash
make synth VARIANT=kuhn TAG=mixed OUT=data/kuhn_mixed.jsonl
# or: python -m pokereval.cli synth --variant leduc --out data/leduc_synth.jsonl
```

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
