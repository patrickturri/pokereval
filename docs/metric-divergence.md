# Action-match vs. exploitability: why the standard LLM-poker metric misleads

**TL;DR.** The most-cited LLM poker benchmark, [PokerBench](https://arxiv.org/abs/2501.08328)
(AAAI 2025), scores **action-match**: did the model pick the solver's best action? PokerEval's
Phase-2 result showed that optimizing action-match makes a policy *more* exploitable, not less.
This note turns that single base-vs-RLVR data point into a **controlled, credential-free
demonstration**: a panel of policies derived entirely from the CFR solver — no model, no API
keys, no training run — on which action-match and exploitability **disagree**, and in fact rise
together. Reproduce it with `make metric-divergence`.

## The two metrics

For a policy `π` over a variant's decision nodes:

- **Action-match** — the policy-level analogue of the leaderboard's per-decision `exact_match`.
  For a stochastic policy sampled once per node, the expected exact-match rate is the mean
  probability mass the policy places on the Nash-modal action:
  `action_match(π) = mean_node π(node)[argmax_a nash(node, a)]`.
  This is exactly what an action-match benchmark rewards.
- **Exploitability** — chips/hand a best-responder extracts (0 = Nash), computed by the solver
  (`exploitability_of_distribution` for the mixed policy, `exploitability_of_choices` for its
  argmax collapse).

## The knob: temperature-sharpening the Nash policy

Sharpen the equilibrium per node by `p_i ∝ nash_i ** (1/τ)`:

- **τ = 1** → the policy *is* the Nash mix: exploitability ≈ 0, but action-match < 1 wherever
  Nash mixes (it spreads mass off the modal action *on purpose*).
- **τ → 0** → the policy collapses onto the argmax: action-match → 1.0 **and exploitability
  climbs**, because the mixing that made it unexploitable is gone.

Sharpening never changes which action is modal, so it isolates one variable: *how much mass sits
on the best action*. Watching both metrics as τ falls is the whole experiment.

## The result (Leduc, `make metric-divergence`)

| Policy | τ | Action-Match ↑ | Greedy Expl. | Mixed Expl. |
|---|---|---|---|---|
| nash_mixed | — | 0.866 | 1.9417 | **0.9406** |
| nash_greedy | — | **1.000** | 1.9417 | **1.9417** |
| uniform | — | 0.444 | 1.7500 | 2.3736 |
| anti_nash | — | 0.013 | 4.0542 | 4.0542 |
| sharpened τ=2 | 2 | 0.785 | 1.9417 | 0.9064 |
| sharpened τ=1 | 1 | 0.866 | 1.9417 | 0.9406 |
| sharpened τ=0.5 | 0.5 | 0.916 | 1.9417 | 1.1502 |
| sharpened τ=0.25 | 0.25 | 0.949 | 1.9417 | 1.3813 |
| sharpened τ=0.1 | 0.1 | 0.973 | 1.9417 | 1.5457 |

Two readings of the same table:

1. **`nash_greedy` vs `nash_mixed`** — both encode the *same* per-node best action, so a perfect
   action-match scorer prefers `nash_greedy` (1.00 vs 0.87). Yet `nash_greedy` is **twice as
   exploitable** (1.94 vs 0.94). The policy the metric ranks higher is the worse poker player.
2. **The τ-sweep** — as τ falls from 1.0 to 0.1, action-match rises 0.87 → 0.97 **while mixed
   exploitability rises 0.94 → 1.55**. The two metrics move together, in the wrong direction.

This is the Phase-2 finding (RLVR drove exact-match 40% → 61% and exploitability 1.05 → 1.75)
reproduced from first principles — the sharpened-Nash curve passes straight through the same
region, with no model and no Tinker spend. See [`phase2-findings.md`](phase2-findings.md) for the
live-model version.

## Why it matters

A benchmark that scores action-match rewards *committing harder to the modal action* — which is
precisely what a mixed-strategy equilibrium must **not** do. In an imperfect-information game,
deliberately spreading mass off your best action is what makes you unexploitable; an action-match
metric is blind to it and actively penalizes it. Exploitability is the metric that sees the
collapse. PokerEval reports both, but this panel shows *why you cannot use the cheaper one as a
proxy for the one you care about*.

## Reproduce

```bash
make metric-divergence                      # Leduc panel + verdict (offline)
make metric-divergence VARIANT=kuhn         # smaller game, same story
python -m pokereval.cli metric-divergence --variant leduc --format json   # raw rows
```

Implementation: `src/pokereval/eval/metric_divergence.py` (panel + metrics) and
`src/pokereval/solver/nash.py` (CFR Nash, exploitability). Tested in
`tests/eval/test_metric_divergence.py` — the pure-math core runs without OpenSpiel; the panel
invariants are `pyspiel`-gated.
