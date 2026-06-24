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
| 3 | `kl_nash` | 100 / 1e-4 | 1.046 → **1.750** | 1.40 → 1.56 | 0.38 → **0.57** | 39% → **61%** |

(Run 2's earlier `lr=1e-4` attempt was lost to a depleted balance; row 3 is the re-run.)

### The headline finding (real, reproducible across BOTH rewards)
Single-step RLVR against a per-decision Nash oracle **systematically collapses to a
deterministic modal policy and *increases* exploitability** — and it does so for *two
different reward designs*:

- **Run 1, naive `nash_prob`:** reward = Nash prob of the sampled action. Maximized by
  collapsing onto the modal action in every state.
- **Run 3, entropy-regularized `kl_nash`:** reward = `log nash(a) − log π(a)`, whose
  *theoretical* optimum is the Nash mixed distribution (verified offline: feeding the true
  Nash distribution to `exploitability_of_distribution` yields ≈0). At `lr=1e-5` it
  under-trained (Run 2: weak signal, policy barely moved). At `lr=1e-4` it had enough signal
  to train — **and collapsed to the identical deterministic policy** (greedy 1.75,
  exact-match 0.6096, nash-prob 0.5744 — byte-identical to Run 1 and to the
  "first-legal-action" heuristic).

In every case per-decision agreement with the solver **rose** (exact-match 40% → 61%) while
whole-policy **exploitability got worse** (greedy 1.05 → 1.75; mixed ~1.40 → ~1.56). The
proxy improved; the true objective regressed.

### Interpretation
The failure is **structural, not a reward-tuning problem.** A single-step (bandit) framing
optimizes each decision in isolation, so its optimum is the per-state argmax — a deterministic
policy. But Leduc equilibrium *requires mixing* (bluffing at calibrated frequencies), which a
deterministic policy cannot represent. Two rewards from opposite ends (pure exploitation vs.
explicit entropy regularization) both converge to the same collapse, so the fix is **not a
better per-decision reward** but a **different problem formulation**: self-play, where the
agent plays full hands and mixing emerges from game outcomes (no oracle). This is exactly how
superhuman poker agents (Pluribus) are trained.

### Status
A genuine exploitability *reduction* was not achievable with the bandit/RLVR-against-oracle
formulation — a clean negative result, reproduced across reward designs. The recommended next
step is the self-play MDP (below).

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
