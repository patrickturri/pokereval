# Solver validation — CFR vs. the closed-form Kuhn equilibrium

The CFR solver in [`src/pokereval/solver/nash.py`](../src/pokereval/solver/nash.py)
is the ground truth for everything in this repo: eval exploitability, the
synth/RLVR Nash labels, the metric-divergence panel, and the web demo all trust
it to represent "what a Nash equilibrium does." A bug there would silently
corrupt every result. So rather than assume the solver is correct, we *prove* it
against a game whose equilibrium is known in closed form.

**Kuhn poker** (Kuhn, 1950) is the smallest non-trivial imperfect-information
poker game — one card each from {J, Q, K}, a one-chip ante, a single one-chip
bet — and one of the very few with a fully analytic Nash equilibrium. That makes
it the ideal correctness anchor.

## The closed-form equilibrium

Kuhn's equilibria form a **one-parameter family** indexed by
`α = P(bet | Jack)` for the first player, with `α ∈ [0, 1/3]`. Every other
action probability is pinned by `α` through the game's indifference conditions:

| Decision | Jack | Queen | King |
|---|---|---|---|
| **P0 opens** (check/**bet**) | bet `α` | bet `0` | bet `3α` |
| **P1 facing a bet** (fold/**call**) | call `0` | call `1/3` | call `1` |
| **P1 after a check** (check/**bet**) | bet `1/3` | bet `0` | bet `1` |
| **P0 after check-then-bet** (fold/**call**) | call `0` | call `α + 1/3` | call `1` |

Three relationships are worth calling out because they hold for the *entire*
family, independent of `α`:

- **The bluff-to-value ratio is exactly 3:1.** The first player value-bets the
  King at `3α` and bluffs the Jack at `α` — it bluffs its worst hand at exactly
  one-third the rate it bets its best. Bluff *too little* and a bet always means
  strength (easily exploited by folding); bluff *too much* and a bet means
  nothing (exploited by calling). 3:1 is the balance that makes the opponent
  indifferent.
- **The defender calls a bet with the Queen exactly 1/3 of the time.** Calling
  less invites bluffs; calling more pays off value bets. 1/3 is the frequency
  that makes the bettor indifferent to bluffing.
- **The Queen never opens.** With the middle card, betting can only be called by
  a better hand and folded to by a worse one — checking strictly dominates.

This module encodes the table as
[`analytic_kuhn_strategy(α)`](../src/pokereval/solver/kuhn_equilibrium.py), keyed
by the *same* information-state strings OpenSpiel emits (`"0"`, `"0b"`, `"0p"`,
`"0pb"`, …, with card index `0=J, 1=Q, 2=K`), so it is directly comparable to
the solver's output.

## CFR reproduces it

Running CFR for 8000 iterations and recovering `α` from the solver's own
`P(bet | Jack)` gives **α ≈ 0.202**. Every other probability then matches the
closed form to within ~0.001:

| key | card | context | action | solver | analytic |
|---|---|---|---|--:|--:|
| `0`   | J | open           | bet  | 0.202 | 0.202 |
| `1`   | Q | open           | bet  | 0.001 | 0.000 |
| `2`   | K | open           | bet  | 0.606 | 0.606 |
| `0b`  | J | facing bet     | call | 0.000 | 0.000 |
| `1b`  | Q | facing bet     | call | 0.334 | 0.333 |
| `2b`  | K | facing bet     | call | 1.000 | 1.000 |
| `0p`  | J | after check    | bet  | 0.334 | 0.333 |
| `1p`  | Q | after check    | bet  | 0.000 | 0.000 |
| `2p`  | K | after check    | bet  | 1.000 | 1.000 |
| `0pb` | J | after check-bet | call | 0.000 | 0.000 |
| `1pb` | Q | after check-bet | call | 0.536 | 0.535 |
| `2pb` | K | after check-bet | call | 1.000 | 1.000 |

Note `0.606 ≈ 3 × 0.202` (the 3:1 bluff ratio), `1b` call `≈ 1/3` (the
indifference-calling frequency), and `1pb` call `≈ α + 1/3 = 0.535` — all
emergent from self-play regret minimization, with no equilibrium knowledge baked
into the solver.

## Reproduce

```bash
python -m pytest tests/solver/test_kuhn_equilibrium.py -q
```

[`tests/solver/test_kuhn_equilibrium.py`](../tests/solver/test_kuhn_equilibrium.py)
checks the analytic strategy is a valid distribution and encodes the invariants
(pure Python, no OpenSpiel), then asserts the CFR average policy matches
`analytic_kuhn_strategy(α)` at every information state (deterministic, `abs=0.02`).

```python
from pokereval.solver.kuhn_equilibrium import solver_kuhn_strategy, alpha_of_strategy
s = solver_kuhn_strategy(8000)
print("alpha =", round(alpha_of_strategy(s), 4))   # ≈ 0.2018
print("P(bet|K)/P(bet|J) =", round(s["2"]["bet"] / s["0"]["bet"], 3))  # ≈ 3.0
```

## Why this matters for the rest of the repo

Leduc — the variant the eval suite, RLVR, and self-play harness actually train
and measure on — has *no* closed-form equilibrium, which is exactly why CFR is
needed there. But the solver code path (`solve_nash`, `nash_probs_by_key`,
`exploitability_*`) is shared across variants. Pinning it to the known Kuhn
equilibrium is the strongest available evidence that the Leduc exploitability
numbers — including the headline `1.05 → 1.75` collapse — rest on a correct
solver.

## References

- H. W. Kuhn, "A Simplified Two-Person Poker," *Contributions to the Theory of
  Games*, vol. 1, pp. 97–103, 1950.
- M. Zinkevich, M. Johanson, M. Bowling, C. Piccione, "Regret Minimization in
  Games with Incomplete Information," *NeurIPS*, 2007 — CFR, whose average policy
  converges to a member of the equilibrium family above.
- "Kuhn poker," Wikipedia — standard statement of the one-parameter equilibrium.
