# Exploitability leak attribution: *where* does a policy bleed chips?

**TL;DR.** Exploitability ("a best-responder extracts 1.75 chips/hand from this
policy") is the headline metric in PokerEval — but it is a single aggregate
number. This note localizes it: it attributes a policy's exploitability across
its individual decision nodes, so "more exploitable" becomes a *ranked list of
culpable decisions*. Run it with `make leak-attribution`. On the collapsed
Phase-2 policy the leak concentrates exactly on the **mixed** nodes — the spots
where committing to one action throws away equilibrium mass — and repairing those
few nodes back to the Nash mix recovers most of the gap. Credential-free: no
model, no API keys, no training run.

## The method: counterfactual repair

For a policy `π` and the CFR Nash policy on the same game, the **marginal leak**
of a decision node `s` is

```
leak(s) = E(π) − E(π with node s repaired to the Nash mix)
```

where `E` is OpenSpiel's exact best-response exploitability (`NashConv / 2`; 0 =
Nash). `leak(s)` is the chips/hand a best-responder was extracting *at that one
decision*: replace the policy's behavior there with the equilibrium's and read off
how much the best-responder loses.

Two honest caveats, both handled in the report:

- **Marginal leaks don't sum to the total.** Best-response is non-linear, so
  fixing one node changes how much a best-responder can gain elsewhere. The
  one-at-a-time leaks *rank* the culprits; they are not an additive decomposition.
- **The cumulative curve is the exact accounting.** Repair nodes in ranked order
  and remeasure after each: that curve is exact at every prefix and lands on the
  Nash exploitability once every deviating node is repaired (full repair *is* the
  Nash policy). It answers "how much of the gap do the top-k decisions explain?"

This is the same idea as a *local best response* probe (Lisý & Bowling, 2017):
don't just score the whole strategy — find *where* it is exploitable.

## Worked example: Kuhn poker (`make leak-attribution VARIANT=kuhn`)

Kuhn is small enough to attribute **every** node and check the result against the
game's closed-form equilibrium. The policy under attribution is the **collapsed**
policy — the Nash argmax, i.e. always commit to the single most-likely equilibrium
action — which is the deterministic mode that Phase-2 RLVR collapsed onto.

```
Policy exploitability: 0.1667 (Nash floor 0.0005; gap +0.1661 chips/hand; 5 mixed decision nodes ranked)

| Rank | Info state | Legal | Marginal leak ↓ | Policy → Nash |
|---:|---|---:|---:|---|
| 1 | 1b  | 2 | +0.0554 | {fold:1.00}  → {fold:0.67, call:0.33} |
| 2 | 0p  | 2 | +0.0278 | {check:1.00} → {check:0.67, bet:0.33} |
| 3 | 0   | 2 | +0.0164 | {check:1.00} → {check:0.80, bet:0.20} |
| 4 | 1pb | 2 | +0.0055 | {call:1.00}  → {fold:0.47, call:0.53} |

Repairing the top 3 nodes to Nash closes 77% of the exploitability gap; repairing
all 12 deviating nodes recovers the Nash policy exactly (0.0005).
```

(The ranking pass scores the genuinely-mixed nodes; a converged CFR average policy
leaves negligible noise on the "pure" nodes, so those are skipped from the ranking
but still repaired in the exact endpoint above.)

(Info-state strings are OpenSpiel's: the leading digit is the private card —
`0`=J, `1`=Q, `2`=K — followed by the betting history, `p`=pass/check,
`b`=bet. Action `0` is check/fold, action `1` is bet/call.)

Every top leak is a **textbook Kuhn equilibrium mixing frequency** that the
collapse destroyed:

- **`1b` — Queen facing a bet (the biggest leak).** Kuhn's closed form says call
  with probability **1/3** (the indifference frequency that makes the bettor
  indifferent to bluffing). The collapsed policy folds 100% of the time, so an
  opponent can bet *any* hand into the Queen for free. Repairing this one node is
  a third of the entire leak.
- **`0p`, `0` — bluffing the Jack.** The equilibrium bluffs the worst hand at a
  controlled rate (1/3 after a check, the `α≈0.2` opening bluff first to act). The
  collapse never bluffs, so it is transparent and gives up the value bluffing buys.
- **`1pb` — Queen in the check-bet line.** A near-even 47/53 mix; collapsing it to
  a pure call is the smallest of the four leaks, as the metric should report.

This is the same phenomenon `docs/solver-validation.md` validates from the other
direction: the CFR solver *reproduces* Kuhn's 1/3 calling and 3:1 bluff-to-value
frequencies, and leak attribution shows that **discarding exactly those mixes** is
what makes the collapsed policy exploitable. It is the Phase-2 mode-collapse
finding, localized decision by decision.

## Leduc (`make leak-attribution`)

On Leduc (the default, the larger game where the Phase-2 RLVR result was
recorded) the collapsed policy's exploitability is **1.9417** and the Nash floor
is **0.9406** — the same two numbers the metric-divergence panel reports for
`nash_greedy` and `nash_mixed` ([`docs/metric-divergence.md`](metric-divergence.md)),
an independent cross-check that leak attribution is decomposing the right gap. The
biggest single leaks are the round-1 mixed nodes (Nash splits roughly 0.32/0.68
between calling and raising; the collapse commits 100% to the raise), each worth
about **0.10 chips/hand**. Repairing the top 3 nodes closes **26%** of the gap;
repairing every deviating node recovers the Nash floor exactly (0.9406). The leak
is more diffuse than in Kuhn — a larger game spreads it over more decisions — but
the shape is identical: the collapse bleeds chips precisely where the equilibrium
wanted to mix.

> Leduc mixes at ~582 of its decision nodes, and the default run scores each with
> one exact best-response solve, so it is a heavier offline computation (several
> minutes) — in line with `make demo` / `make selfplay-smoke`. Use `VARIANT=kuhn`
> for the quick, fully-worked look above.

## Reproduce

```bash
make leak-attribution                              # Leduc, collapsed policy (offline)
make leak-attribution VARIANT=kuhn                 # the fully-worked small example above
make leak-attribution POLICY=sharpened             # a τ-sharpened policy instead of the argmax
python -m pokereval.cli leak-attribution --policy nash   # sanity: a Nash policy has zero leak
python -m pokereval.cli leak-attribution --variant leduc --format json   # raw report
```

Implementation: `src/pokereval/eval/leak_attribution.py` (pure-Python `repair` /
`deviating_keys` core + the OpenSpiel-backed `attribute_leaks`) and
`src/pokereval/solver/nash.py` (CFR Nash, exact exploitability). Tested in
`tests/eval/test_leak_attribution.py` — the pure helpers run without OpenSpiel;
the attribution invariants (a Nash policy leaks nothing; full repair recovers the
Nash exploitability exactly) are `pyspiel`-gated and run on Kuhn so they stay fast.

## References

- H. W. Kuhn, "A Simplified Two-Person Poker," *Contributions to the Theory of
  Games*, 1950 — the closed-form equilibrium the worked example checks against.
- V. Lisý & M. Bowling, "Equilibrium Approximation Quality of Current No-Limit
  Poker Bots," 2017 ([arXiv:1612.07547](https://arxiv.org/abs/1612.07547)) —
  *local best response*: localizing where a strategy is exploitable rather than
  only scoring the whole policy.
- M. Zinkevich et al., "Regret Minimization in Games with Incomplete
  Information," NeurIPS 2007 — CFR, the solver underneath every number here.
