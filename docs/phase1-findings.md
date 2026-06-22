# Phase 1 — How well do frontier LLMs reason about imperfect-information poker?

**Status:** harness complete and reproducible; frontier-model numbers pending an
API-key run. This document is the methodology + results template; the tables are
filled by running the leaderboard against real models (see *Reproducing* below).

## Question

Imperfect-information poker is a clean probe of a capability that matters for
frontier models: reasoning correctly under hidden information and quantifiable
uncertainty. Kuhn and Leduc poker have *exact* game-theoretic solutions, so we
can grade a model's decisions against ground truth rather than vibes — and we can
measure how *exploitable* a model's whole strategy is, not just whether
individual answers look plausible.

> Where do frontier LLMs systematically deviate from Nash-optimal play, and does
> their stated reasoning track the quality of their decisions?

## What we measure

| Metric | Games | Meaning |
|---|---|---|
| **Mean verifiable** | Kuhn, Leduc | Average Nash probability mass on the chosen action (1.0 = always picks a move the equilibrium plays with prob 1). |
| **Exact-match rate** | Kuhn, Leduc | Fraction of decisions equal to the equilibrium's most-likely action. |
| **Exploitability** | Kuhn, Leduc | Value (chips/hand) a best-responding opponent extracts from the model's *whole deterministic policy*. **0 = unexploitable (Nash).** This is the headline number. |
| **Reference-match** | Hold'em | Fraction of curated "obvious" spots where the model picks the clearly-correct action. |
| **Mean judge** | all | LLM-judge rubric score (0–1) on *reasoning quality*, independent of whether the action was correct. |
| **Parse-error rate** | all | Fraction of responses where no legal `ACTION:` line could be parsed. A capability signal in itself. |

The verifiable graders (Nash probability for Kuhn/Leduc, reference action for
Hold'em) are exact and side-effect-free. The judge grader is the
*unverifiable*-domain counterpart — it scores the *argument*, letting us ask
whether a model that plays well also reasons well, and where the two diverge.

## Method

1. **Spots.** Kuhn/Leduc decision nodes are enumerated exhaustively from the
   OpenSpiel game tree (every unique information state). Hold'em uses a small,
   hand-curated set of uncontroversial spots so the reference signal is sound.
2. **Solve.** CFR produces the Nash policy and per-info-state action
   probabilities; exploitability is computed via OpenSpiel best-response.
3. **Run.** Each model sees a rendered State, thinks, and emits `ACTION: <a> [size]`.
4. **Grade.** Verifiable grader scores against Nash/reference; the optional judge
   grades reasoning against a rubric.
5. **Aggregate.** Per-model summary + whole-policy exploitability → leaderboard.

Everything is one typed interface — `State → Decision → Score` — shared by every
component (engine, solver, graders, runner, leaderboard).

## Baseline (always-fold)

A degenerate `fake-fold` client (always returns `ACTION: fold`) anchors the
scale. Folding is illegal when not facing a bet, which is why ~half of Leduc
nodes register as parse/legal errors — the baseline is *meant* to look bad. Its
non-zero exploitability and mean-verifiable just reflect that "fold" coincides
with the Nash action at some nodes by chance.

| Variant | N | Errors | Mean verifiable | Exact-match | Exploitability |
|---|---|---|---|---|---|
| Kuhn  | 12  | 6   | 0.52 | 0.50 | 0.92 |
| Leduc | 936 | 312 | 0.48 | 0.50 | 3.56 |
| Hold'em | 6 | 3 | 0.67 | 0.00 | — |

*(Generated with `make leaderboard ITERATIONS=2000`; small variance run-to-run
because CFR is seeded by iteration count, not RNG.)*

## Frontier-model leaderboard

Run with real keys (`make leaderboard PROVIDER=anthropic MODEL=claude-opus-4-8`,
etc.). Target ≥3 models spanning at least one Anthropic model, one OpenAI model,
and one open model served via an OpenAI-compatible endpoint.

### Results so far

**claude-opus-4-8** (adaptive thinking; CFR at 2000 iterations):

| Variant | N | Errors | Mean verifiable | Exact-match | Exploitability | Mean judge |
|---|---|---|---|---|---|---|
| Kuhn    | 12  | 0 | 0.750 | 0.750 | **0.167** | — |
| Leduc   | 936 | 0 | 0.527 | 0.545 | **2.765** | — |
| Hold'em | 6   | 0 | 1.000 | n/a   | —          | 0.667 |

For scale, the always-fold baseline scores exploitability ≈ 0.92 (Kuhn) and
≈ 3.56 (Leduc); Nash is 0 in both.

Reading these:

- **Kuhn confirms the headline hypothesis.** Opus picks the modal Nash action
  75% of the time, but its *whole policy* is still exploitable at **0.167
  chips/hand** — far better than the always-fold baseline (0.92) yet well above
  Nash (0). It plays the obvious nodes correctly and collapses the mixed
  (bluff/call-frequency) nodes to pure strategies, which a best-responder
  punishes. This is exactly the failure mode the Phase-2 flywheel targets.
- **Leduc is where it breaks down.** On the harder game Opus picks the modal
  Nash action only **54.5%** of the time and its policy stays exploitable at
  **2.765 chips/hand** — only ~22% better than always-folding (3.56), and a
  world away from Nash. The capability gap between trivial (Kuhn) and merely
  small (Leduc) imperfect-information games is large: more betting structure and
  the public card overwhelm its equilibrium reasoning. This is the headline
  failure mode.
- **Hold'em: 6/6 curated reference actions correct** (mean-verifiable 1.000),
  with a judge reasoning score of 0.667 — the decisions are right and the stated
  reasoning is decent but doesn't hit every rubric criterion. (Exact-match is
  Nash-only and n/a here.)
- **0 parse errors across all 954 decisions** — Opus reliably emits the
  `ACTION:` format; the eval signal is clean, not noise from formatting.

| Model | Variant | Mean verifiable | Exact-match | Exploitability | Mean judge | Parse-err |
|---|---|---|---|---|---|---|
| claude-opus-4-8 | kuhn   | 0.750 | 0.750 | 0.167 | — | 0/12 |
| claude-opus-4-8 | leduc  | 0.527 | 0.545 | 2.765 | — | 0/936 |
| claude-opus-4-8 | holdem | 1.000 | n/a   | —     | 0.667 | 0/6 |
| _gpt-…_         | leduc  | | | | | |
| _qwen-…_        | leduc  | | | | | |

### Hypotheses to test (Phase-1 headline)

- **Exploitability ≫ 0 even when exact-match is high.** Expect models to play
  the modal action well but collapse mixed strategies to pure ones, leaving them
  exploitable — bluffing/calling frequencies are where equilibrium needs
  randomization that LLMs don't produce.
- **Judge ↔ ground-truth gap.** Reasoning that *sounds* like pot-odds analysis
  may not correlate with picking the Nash action — measure the correlation.
- **Failure concentrates at facing-a-bet nodes** (fold/call/raise decisions under
  uncertainty) more than at free checks.

These become the targeted failure modes the Phase-2 synthetic-data flywheel
generates against.

## Reproducing

```bash
make install          # editable install (OpenSpiel + SDKs)
make test             # 43 tests; OpenSpiel-marked tests need pyspiel
make leaderboard      # credential-free baseline, all three variants

# Real model (needs ANTHROPIC_API_KEY):
make leaderboard PROVIDER=anthropic MODEL=claude-opus-4-8
# Archive raw numbers:
make leaderboard-json VARIANT=leduc PROVIDER=anthropic MODEL=claude-opus-4-8 > results/leduc-opus.json
```
