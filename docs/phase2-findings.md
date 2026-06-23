# Phase 2 — RLVR on Tinker: Methodology & Findings

## Method
- Single-step bandit over Leduc decision nodes (CFR/Nash-labeled by the synth generator).
- Reward = verifiable signal derived from the solver's Nash strategy.
- Group-relative advantages (GRPO-lite); LoRA fine-tune of `Qwen/Qwen3.5-4B` via Tinker
  `importance_sampling`.
- Base model emits a terse `ACTION: <action>` line (system prompt forces it; reasoning
  models otherwise never reach a parseable action → all-zero reward).
- Metric: Leduc **exploitability** vs the CFR solver (Nash = 0), measured two ways —
  *greedy* (argmax → deterministic policy) and *mixed* (sampled action distribution →
  the correct lens for a game that requires mixing). Held-out 20% split for per-decision
  metrics.

## Results so far

| Run | Reward | Steps / lr | Greedy expl. | Mixed expl. | Nash-prob | Exact-match |
|-----|--------|-----------|--------------|-------------|-----------|-------------|
| Base | — | — | 1.046 | 1.45 | 0.39 | 40.1% |
| 1 | `nash_prob` | 60 / 1e-5 | 1.046 → **1.750** | — | 0.39 → **0.57** | 40% → **61%** |
| 2 | `kl_nash` | 60 / 1e-5 | 1.046 → 1.046 | 1.45 → 1.54 | 0.38 → 0.39 | 39% → 40% |
| 3 | `kl_nash` | 100 / 1e-4 | — (out of funds) | — | — | — |

### The headline finding (real, reproducible)
Run 1 is a clean **reward-misspecification (Goodhart) result**. Rewarding the Nash
probability of the *sampled action* is maximized by collapsing onto the single modal
action in every state. So per-decision agreement with the solver **rose** (exact-match
40% → 61%; the greedy policy converged exactly to a "first-legal-action" deterministic
policy) while whole-policy **exploitability got worse** (1.05 → 1.75) — because Leduc
equilibrium *requires mixing*, and a deterministic policy is more exploitable. The proxy
improved; the true objective regressed.

### The attempted fix
Run 2 switched to a **KL-to-Nash reward** (`log nash(a) − log π(a)`), whose policy-gradient
optimum *is* the Nash mixed distribution (verified offline: feeding the true Nash
distribution to `exploitability_of_distribution` yields ≈0). It did not collapse — but at
`lr=1e-5` it **under-trained** (the entropy term and the Nash term partially cancel, leaving
a weak signal; the policy barely moved). Run 3 (`lr=1e-4`, 100 steps) was intended to give
that correct objective enough signal to bite, but the Tinker balance was exhausted before it
executed — **no data**.

### Status
A genuine exploitability *reduction* has not yet been demonstrated. The two completed runs
show (a) why the naive verifiable reward is the wrong objective for equilibrium, and (b) that
the corrected objective needs more optimization pressure than one low-LR pass. The decisive
higher-LR run is pending Tinker funds.

## Tooling notes
- `pokereval rl-train --variant leduc --preset {smoke,real} [--live] [--reward-mode kl_nash|nash_prob]
  [--lr --steps --temperature --group-size --eval-samples]`
- Pre-flight billing check aborts fast instead of a ~1h retry loop on a dead balance.
- `scripts/demo.py` — prints model action distribution vs solver Nash, spot by spot.

## Reproduce
```bash
make rl-smoke      # offline loop check (mock backend, no account)
# live (needs Tinker balance):
python -m pokereval.cli rl-train --variant leduc --preset real --live \
    --reward-mode kl_nash --lr 1e-4 --steps 100
```

## What's next
1. **Finish the KL win** (needs ~$5–15 Tinker top-up): the `lr=1e-4` run to test whether the
   mixing-preserving reward drives exploitability *down*. Now guarded against fund-exhaustion waste.
2. **Self-play (the proper MDP RL)**: drop the solver-oracle reward entirely; the model plays full
   hands, reward = chips at showdown, equilibrium emerges. The version that generalizes to
   unsolvable poker (how Pluribus works). Larger build.
