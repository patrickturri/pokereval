# PokerEval — the Phase-2 finding, before vs after

Reconstructed offline from the CFR solver and the recorded Tinker run — **no API keys, no Tinker account**. The collapsed policy is rebuilt from the solver and its exploitability recomputed live below.

## Aggregate — RLVR improved the proxy but *worsened* the true objective

| Metric (held-out Leduc) | Base | After RLVR |
|---|---|---|
| Exact-match with Nash action | 39.0% | **61.0%** ⬆ |
| Mean Nash-probability mass | 0.382 | **0.574** ⬆ |
| Greedy exploitability (vs solver) | 1.046 | **1.750** ⬇ worse |
| Mixed exploitability (vs solver) | 1.404 | **1.557** ⬇ worse |

Model: `Qwen/Qwen3.5-4B` · reward: `kl_nash` · source: `rl-leduc-kl-lr1e4.json`

Live re-computation of the collapsed policy's **greedy exploitability**: **1.750** (Nash equilibrium: 0.0), across 936 Leduc info states.

The proxy (exact-match, Nash-prob) went **up** while exploitability got **worse**: single-step RLVR collapsed the policy onto a deterministic mode, discarding the mixing Leduc's equilibrium requires.

## Where it collapsed — decision by decision

Each spot below is a genuinely *mixed* Leduc node (the solver randomizes). **Before** is the Nash mix the equilibrium plays; **after** is the single action the collapsed RLVR policy commits to — and the equilibrium mass that commitment throws away.

### Spot 1 — hero Q, board J, round 2, pot 22
- legal: `fold, call`
- Nash mix (before): `fold=50%, call=50%`
- RLVR commits (after): `fold` — discards **50%** of the equilibrium mass

### Spot 2 — hero Q, board K, round 2, pot 10
- legal: `fold, call, raise`
- Nash mix (before): `fold=51%, call=48%, raise=1%`
- RLVR commits (after): `fold` — discards **49%** of the equilibrium mass

### Spot 3 — hero K, board Q, round 2, pot 10
- legal: `check, raise`
- Nash mix (before): `check=52%, raise=48%`
- RLVR commits (after): `check` — discards **48%** of the equilibrium mass

### Spot 4 — hero Q, board K, round 2, pot 2
- legal: `check, raise`
- Nash mix (before): `check=53%, raise=47%`
- RLVR commits (after): `check` — discards **47%** of the equilibrium mass

### Spot 5 — hero K, board J, round 2, pot 22
- legal: `fold, call`
- Nash mix (before): `fold=53%, call=47%`
- RLVR commits (after): `fold` — discards **47%** of the equilibrium mass

### Spot 6 — hero K, board Q, round 2, pot 14
- legal: `fold, call, raise`
- Nash mix (before): `call=55%, fold=37%, raise=8%`
- RLVR commits (after): `fold` — discards **63%** of the equilibrium mass


---

*Generated offline by `python scripts/demo.py --out docs/demo-before-after.md` — no API keys, no Tinker account. Every number above is recomputed from the CFR solver and the recorded run; the greedy exploitability (1.750) is solved live at generation time.*
