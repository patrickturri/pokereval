# PokerEval

[![CI](https://github.com/patrickturri/pokereval/actions/workflows/ci.yml/badge.svg)](https://github.com/patrickturri/pokereval/actions/workflows/ci.yml)

A poker testbed for frontier-model research engineering.

> End-to-end RL research pipeline that evaluates and improves LLM poker reasoning —
> a `verifiers`-compatible environment, a verifiable/unverifiable eval suite, a
> synthetic adversarial-data flywheel, an RLVR training run on Tinker, an
> oracle-free self-play harness, and a credential-free web demo that reproduces
> the headline finding live.

## Why poker?

![Milestones in AI vs. human game-playing — from checkers and chess to Go and, in 2017, no-limit Texas Hold'em](docs/assets/list_games_AI_vs_human.avif)

Most games AI has conquered — checkers, chess, Go — are games of *perfect*
information: both players see the whole board. Poker is different. It is a game
of **imperfect information** (hidden cards, bluffing, mixed strategies), and that
is exactly what makes it a hard, real-world-shaped benchmark.

In 2017, Carnegie Mellon's **Libratus** beat four of the world's top heads-up
no-limit Hold'em professionals over 120,000 hands — the first AI to do so. Its
creators, Tuomas Sandholm and Noam Brown, were explicit that poker was never the
point: *"The techniques in Libratus do not use expert domain knowledge or human
data not specific to poker… thus they apply to a host of imperfect-information
games"* — business negotiation, cybersecurity, pricing, and more. Poker is the
benchmark; reasoning under hidden information is the prize.

PokerEval takes that same lens to **frontier LLMs**: can today's models reason in
an imperfect-information game, and can RL make them *less* exploitable rather than
just more confident? (See the headline finding below — the answer is instructive.)
Background: ["How poker-playing AI Libratus is learning to negotiate better than
any human"](https://qz.com/907896/how-poker-playing-ai-libratus-is-learning-to-negotiate-better-than-any-human),
Quartz.

## Status

✅ **Phase 1** (environment + eval suite) — implemented and reproducible.
[`docs/phase1-findings.md`](docs/phase1-findings.md)

✅ **Phase 2** (synthetic-data flywheel + RLVR on Tinker) — pipeline built,
offline-tested, and **run live** (LoRA fine-tune of `Qwen/Qwen3.5-4B`).
[`docs/phase2-findings.md`](docs/phase2-findings.md)

🟡 **Phase 3** (self-play + demo) — oracle-free self-play RL harness built and
offline-validated (`src/pokereval/rl/selfplay/`, chips-at-showdown reward,
frozen-snapshot opponent, deal-conditioned GRPO), plus a **credential-free web
demo** (`make demo`) that recomputes the collapse live. The live self-play
*training* run on an LLM is gated on Tinker funds — but the **self-play loop
itself is now shown to work offline**: a tabular policy trained by the *same*
loop drives Leduc exploitability from the uniform ~2.37 down to ~1.6, below the
collapsed RLVR policy's 1.75 (`make selfplay-tabular`, no API keys). The harness
now records the full **exploitability-vs-CFR curve across opponent refreshes**
(`eval_curve` in the `rl-selfplay` output), so one funded `--live` run answers the
headline question — *does self-play drive exploitability down?* — end to end; and
a credential-free before/after write-up ships via `python scripts/demo.py`.
[`docs/phase3-selfplay-offline.md`](docs/phase3-selfplay-offline.md)

🧰 **Supporting tooling** (all credential-free, offline) — continuous integration
(GitHub Actions runs the full suite + smoke loops on every push/PR; badge above),
an opt-in faster-converging **CFR+** Nash baseline
([`docs/solver-cfr-plus.md`](docs/solver-cfr-plus.md)), and **exploitability leak
attribution** that localizes *which* decisions a policy bleeds chips on
([`docs/leak-attribution.md`](docs/leak-attribution.md)).

> **Headline result — a reproducible negative finding.** Single-step RLVR against
> a per-decision Nash oracle *raised* the model's per-decision agreement with the
> solver (exact-match **40% → 61%**) but made the whole policy **more exploitable**
> (greedy exploitability **1.05 → 1.75**) — by collapsing onto a deterministic
> modal policy. This held for **two different rewards** (naive Nash-prob *and* an
> entropy-regularized KL-to-Nash whose theoretical optimum is the mixed
> equilibrium), so the failure is **structural**: a bandit framing optimizes each
> decision in isolation and cannot represent the *mixing* a poker equilibrium
> requires. The principled fix is self-play, not a better per-decision reward.

## Quickstart

```bash
make install          # editable install (OpenSpiel + model SDKs)
make test             # full test suite
make leaderboard      # credential-free baseline leaderboard, all variants
make demo             # serve the finding as a local web app → http://localhost:8000
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

Every one of those numbers rests on the CFR solver being correct. It is
validated against Kuhn poker's **closed-form Nash equilibrium** (Kuhn, 1950):
CFR reproduces the analytic strategy — the 3:1 bluff-to-value ratio, the 1/3
indifference-calling frequency — to within ~0.001 at every information state.
Write-up: [`docs/solver-validation.md`](docs/solver-validation.md).

#### Why two metrics? Action-match vs. exploitability

The most-cited LLM poker benchmark, [PokerBench](https://arxiv.org/abs/2501.08328)
(AAAI 2025), scores **action-match** — did the model pick the solver's best
action? PokerEval reports exploitability *too*, because the two disagree. A
credential-free panel of solver-derived policies makes this concrete (no model,
no API keys):

```bash
make metric-divergence    # action-match vs. exploitability across a policy panel
```

Sharpening the Nash policy toward its argmax drives action-match up (0.87 → 1.00)
**while exploitability gets worse** (0.94 → 1.94) — the collapsed `nash_greedy`
policy earns a *perfect* action-match score yet is twice as exploitable as the
Nash mix. This is the Phase-2 RLVR finding reproduced from first principles, with
zero training spend. Full write-up: [`docs/metric-divergence.md`](docs/metric-divergence.md).

#### Where does the leak come from? Exploitability attribution

Exploitability is one aggregate number; it says a policy is exploitable, not
*which decisions* leak. This attributes it across decision nodes by a
counterfactual repair — replace one node with the Nash mix, remeasure
exploitability, read off the drop (no model, no API keys):

```bash
make leak-attribution VARIANT=kuhn    # rank the leaking decisions of the collapsed policy
```

On Kuhn the collapsed (argmax) policy's exploitability **0.167** decomposes onto
exactly the equilibrium mixes it threw away — the biggest leak is *folding the
Queen to a bet* (Nash calls 1/3 of the time), then *never bluffing the Jack*.
Repairing the top 3 nodes closes **77%** of the gap; repairing all of them
recovers Nash exactly. This is the Phase-2 mode collapse localized decision by
decision, tying back to the same 1/3-calling and 3:1 bluff frequencies the solver
is validated against. Full write-up: [`docs/leak-attribution.md`](docs/leak-attribution.md).

### Synthetic data (Phase 2)

Generate CFR/Nash-labeled spots tagged by failure mode (the spots where
frontier models are predicted to deviate — mixed strategies, facing-a-bet
decisions). These double as verifiable-reward RLVR training data and a targeted
eval set:

```bash
make synth VARIANT=kuhn TAG=mixed OUT=data/kuhn_mixed.jsonl
# or: python -m pokereval.cli synth --variant leduc --out data/leduc_synth.jsonl
```

### RLVR training (Phase 2)

The `src/pokereval/rl/` module turns those labeled spots into an RLVR run: a
single-step bandit over Leduc decision nodes, group-relative advantages (GRPO-lite),
and a LoRA fine-tune on Tinker (`importance_sampling`). Two reward designs are
implemented — `nash_prob` (Nash probability of the chosen action) and `kl_nash`
(`log nash(a) − log π(a)`, whose optimum is the Nash *mixed* distribution). Quality
is measured by Leduc **exploitability** vs the CFR solver, both *greedy* (argmax) and
*mixed* (sampled distribution — the correct lens for a mixing game), plus per-decision
Nash-prob and exact-match on a held-out split.

```bash
make rl-smoke    # full loop, offline (mock backend) — no Tinker account needed
# live LoRA run on Tinker (needs a funded account):
python -m pokereval.cli rl-train --variant leduc --preset real --live \
    --reward-mode kl_nash --lr 1e-4 --steps 100
python scripts/demo.py --n-spots 6 --samples 24   # model action dist vs solver Nash
```

**Results (live runs on `Qwen/Qwen3.5-4B`):**

| Metric (held-out Leduc) | Base | After RLVR |
|---|---|---|
| Exact-match w/ Nash action | 40.1% | **61.0%** ⬆ |
| Mean Nash-probability mass | 0.39 | **0.57** ⬆ |
| Greedy exploitability (vs solver) | 1.05 | **1.75** ⬇ worse |
| Mixed exploitability (vs solver) | ~1.40 | ~1.56 ⬇ worse |

The model got better at *matching the solver's single best action* but **more
exploitable overall**, because optimizing a per-decision reward collapses the policy
onto a deterministic mode and discards the mixing Leduc equilibrium needs. Reproduced
across both reward designs — see [`docs/phase2-findings.md`](docs/phase2-findings.md)
for the full analysis. Everything but the live `--live` run is covered by the offline
test suite.

## Phases

1. ✅ **Environment + Eval suite** — poker engine wrapper, graders (verifiable +
   LLM-judge), static eval set, frontier-model leaderboard.
2. ✅ **Synthetic data flywheel + RLVR** — solver-labeled spot generation, RLVR
   LoRA fine-tune of `Qwen/Qwen3.5-4B` on Tinker, exploitability evaluation.
   Outcome: a reproducible mode-collapse / reward-misspecification finding (above).
3. 🟡 **Self-play + demo** — the principled fix to Phase 2: episodic self-play
   (chips-at-showdown reward, no oracle) where mixing can emerge. The harness is
   built and offline-validated; the live *LLM* training run is gated on Tinker
   funds. Ships now: (a) a **credential-free tabular self-play result** — the same
   loop, run on a small tabular policy, drives Leduc exploitability **2.37 → ~1.6**
   offline (below the collapsed RLVR policy's 1.75), confirming the formulation is
   sound independent of the LLM substrate; and (b) a credential-free web demo
   (`make demo`) that reconstructs the collapsed policy from the solver and
   reproduces its 1.75 exploitability live, then shows — decision by decision —
   where committing to one action departs from the mixed equilibrium.

#### Self-play works offline (no API keys, no Tinker)

The Phase-2 finding said the *fix* is self-play. Phase 3 proves the fix offline,
on a tabular policy driven by the **identical** loop (chips-at-showdown reward,
deal-conditioned GRPO, frozen-snapshot opponent), evaluated on the time-average
policy (NFSP-style):

```bash
make selfplay-tabular        # ~30s, CPU, credential-free → before/after JSON
```

| Policy (Leduc, mixed-exploitability lens) | Exploitability vs CFR |
|---|---|
| Uniform random (self-play's starting point) | ~2.37 |
| **Tabular self-play — time-average policy** | **~1.6** ⬇ |
| Collapsed Phase-2 RLVR policy (for reference) | 1.75 |
| Nash equilibrium | 0.0 |

This is the **inverse** of the Phase-2 trajectory: optimizing each decision in
isolation pushed exploitability *up* (1.05 → 1.75); rewarding whole-hand outcomes
pulls it *down* (2.37 → ~1.6), because the policy can now represent — and is
rewarded for — the mixing Leduc requires. The solver is consulted only at eval
time. Full write-up: [`docs/phase3-selfplay-offline.md`](docs/phase3-selfplay-offline.md).

### Demo (Phase 3)

```bash
make demo            # → http://localhost:8000  (offline; no API keys, no Tinker)
```

A self-contained FastAPI app (no node build): solver Nash mix vs. the collapsed
RLVR policy across Leduc's mixed decision nodes, alongside the real recorded
base-vs-RLVR metrics. The first run computes the CFR analysis and caches it; the
committed cache makes subsequent starts instant.

## Stack

OpenSpiel (Kuhn/Leduc + CFR + exploitability) · PokerKit (No-Limit Hold'em) ·
`verifiers` (Prime Intellect) · Tinker (LoRA RL) · `Qwen/Qwen3.5-4B` ·
Weights & Biases · FastAPI (self-contained web demo)
