# CFR+ — a faster Nash baseline

**TL;DR.** PokerEval's Nash baseline now offers **CFR+** alongside vanilla CFR
(`solve_nash(game, iters, method="cfr+")`). CFR+ reaches a given exploitability in
roughly **5× fewer iterations** on Leduc, at the same per-iteration cost — so the
solver that defines "ground truth" for every eval, synth label, and the web demo
can be made tighter or cheaper for free. It is the same algorithm family that was
used to *solve* heads-up limit Hold'em.

## Background

Vanilla **CFR** (Counterfactual Regret Minimization; Zinkevich, Johanson, Bowling
& Piccione, "Regret Minimization in Games with Incomplete Information," NeurIPS
2007) converges to a Nash equilibrium in two-player zero-sum games but does so
slowly — average exploitability shrinks at roughly `O(1/√T)`.

**CFR+** (Tammelin, "Solving Large Imperfect Information Games Using CFR+,"
2014, [arXiv:1407.5042](https://arxiv.org/abs/1407.5042)) makes two changes:

1. **Regret-matching⁺** — accumulated regrets are floored at zero each iteration
   instead of being allowed to go negative, so the solver stops "paying down" a
   large negative regret before it can switch back to a newly-good action.
2. **Linear averaging** — later iterates are weighted more heavily in the average
   policy, since they are closer to equilibrium.

These changes give dramatically faster empirical convergence (and the regrets
stay bounded), which is why CFR+ was the engine behind Bowling, Burch, Johanson &
Tammelin, **"Heads-up limit hold'em poker is solved,"** *Science* 348:6218 (2015)
— the first essentially-solved competitively-played imperfect-information game.

## Measured here (Leduc, deterministic)

Exploitability (NashConv / 2; 0 = Nash) of the average policy after *N* iterations,
from `pokereval.solver.nash`:

| Iterations | CFR    | CFR+   |
|-----------:|-------:|-------:|
| 20         | 0.453  | 0.174  |
| 50         | 0.189  | 0.034  |
| 100        | 0.096  | 0.013  |

At 50 iterations CFR+ is already near-Nash where vanilla CFR is ~5× more
exploitable. Reproduce:

```python
from pokereval.engine.kuhn_leduc import load_game
from pokereval.interface.types import GameVariant
from pokereval.solver.nash import solve_nash, nash_exploitability

g = load_game(GameVariant.LEDUC)
for m in ("cfr", "cfr+"):
    print(m, nash_exploitability(g, solve_nash(g, 50, method=m)))
```

The convergence gap is pinned by `tests/solver/test_cfr_plus.py` (offline, no
account): CFR+ at 50 Leduc iterations is asserted to be < half of CFR's
exploitability and < 0.1 absolute, and both solvers reach the same Nash target
given enough iterations.

## Scope

`method` defaults to `"cfr"`, so every existing call (synth labels, eval, demo,
recorded results tables) is byte-for-byte unchanged. CFR+ is opt-in: use it to
tighten the Nash reference cheaply or as a cross-check that the CFR baseline has
actually converged. Both algorithms target the *same* equilibrium; CFR+ just
gets there faster.
