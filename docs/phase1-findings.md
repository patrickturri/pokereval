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

### Comparative leaderboard

Two flagships, adaptive/reasoning enabled, CFR at 2000 iterations. Lower
exploitability is better; baseline (always-fold) and Nash bracket the scale.

| Variant | Metric | Baseline | **claude-opus-4-8** | **gemini-3.1-pro** | Nash |
|---|---|---|---|---|---|
| Kuhn  | exploitability ↓ | 0.92 | 0.167 | 0.167 | 0 |
| Kuhn  | exact-match ↑    | 0.50 | 0.750 | 0.750 | 1 |
| Leduc | exploitability ↓ | 3.56 | 2.765 | **2.458** | 0 |
| Leduc | mean-verifiable ↑| 0.48 | 0.527 | **0.603** | 1 |
| Leduc | exact-match ↑    | 0.50 | 0.545 | **0.645** | 1 |
| Hold'em | reference-match ↑ | 0.50 | 1.000 | 1.000 | — |
| Hold'em | judge ↑         | —    | 0.667 | 0.667 | — |

Per-variant N: Kuhn 12, Leduc 936, Hold'em 6. **0 parse errors** for either
model across all 954 decisions per model — the eval signal is clean, not
formatting noise.

(OpenAI's `gpt-5.1` is wired and ready but its key returned `insufficient_quota`,
so it is not yet on the board.)

Reading these:

- **The easy games don't discriminate.** On Kuhn and Hold'em the two flagships
  are identical (Kuhn exploitability 0.167 each; Hold'em 6/6 reference actions
  and judge 0.667 each). Both still leave Kuhn exploitable at 0.167 vs Nash 0 —
  they play obvious nodes correctly but collapse the mixed (bluff/call-frequency)
  nodes to pure strategies, exactly the failure mode the Phase-2 flywheel targets.
- **Leduc is where they separate, and where both break down.** Adding a public
  card and a second betting round, both models fall far short of Nash:
  exploitability 2.46–2.77 vs baseline 3.56 (only 22–31% better than always
  folding) and modal-action accuracy of just 0.55–0.65. **Gemini 3.1 Pro is the
  stronger imperfect-information reasoner here** — ~11% less exploitable (2.458
  vs 2.765) and +10 points of exact-match (0.645 vs 0.545). The capability gap
  between trivial (Kuhn) and merely-small (Leduc) games is the headline result:
  more betting structure overwhelms equilibrium reasoning in *both* frontier
  models, just less so for Gemini.

| Model | Variant | Mean verifiable | Exact-match | Exploitability | Mean judge | Parse-err |
|---|---|---|---|---|---|---|
| claude-opus-4-8  | kuhn   | 0.750 | 0.750 | 0.167 | —     | 0/12 |
| claude-opus-4-8  | leduc  | 0.527 | 0.545 | 2.765 | —     | 0/936 |
| claude-opus-4-8  | holdem | 1.000 | n/a   | —     | 0.667 | 0/6 |
| gemini-3.1-pro   | kuhn   | 0.750 | 0.750 | 0.167 | —     | 0/12 |
| gemini-3.1-pro   | leduc  | 0.603 | 0.645 | 2.458 | —     | 0/936 |
| gemini-3.1-pro   | holdem | 1.000 | n/a   | —     | 0.667 | 0/6 |

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
