# Phase 3 — Self-Play, Reproduced Offline (credential-free)

Phase 2 established a clean **negative** result: single-step RLVR against a
per-decision Nash oracle collapses to a deterministic modal policy and *raises*
exploitability (greedy 1.05 → 1.75), across two reward designs. The diagnosis was
structural — a bandit framing optimizes each decision in isolation and cannot
represent the *mixing* a poker equilibrium requires — and the prescription was
**self-play over full hands** (chips at showdown, no oracle), where mixing emerges
from outcomes.

The LLM self-play harness (`src/pokereval/rl/selfplay/`) implements exactly that,
but its live training run is gated on a funded Tinker account. Against the mock
backend the loop only proves *mechanics*: `FakeBackend` is a fixed heuristic, so
nothing actually learns offline (design spec, "Risks"). **This note closes that
gap with zero spend.**

## The idea: same loop, a substrate that learns offline

`rl/selfplay/tabular.py` swaps the opaque LLM policy for a small **tabular softmax
policy** keyed on the rendered prompt, and drives it with the *exact same*
self-play scheme — reusing `rollout.play_group`, `advantage.group_advantages`,
and the Phase-2 evaluator `evaluate.evaluate_policy` unchanged:

- **Reward:** chips at showdown (`state.returns()[learner_seat]`). No oracle.
- **Credit assignment:** deal-conditioned GRPO — `G` hands replayed from one fixed
  deal, so the deal's card luck cancels in the group baseline and the advantage
  isolates decision quality.
- **Opponent:** a frozen snapshot of the learner, refreshed every `K` steps
  (fictitious self-play, NFSP-flavored).
- **What's measured:** exploitability of the **time-average** policy. Fictitious
  play converges on the *average* strategy, not the single-iterate best-response —
  the latter is high-variance and need not improve monotonically, which is exactly
  the instability NFSP's averaging is designed to tame (Heinrich & Silver, 2016).

The tabular policy and its average both implement the `SamplingClient` protocol,
so they are drop-in everywhere a sampler is expected — rollouts *and* eval — with
no downstream special-casing.

## Result

Headline run (`make selfplay-tabular`, Leduc, 800 steps, seed 0, ~30s on CPU):

| Policy (Leduc, mixed-exploitability lens) | Exploitability vs CFR |
|---|---|
| Uniform random (the loop's starting point) | ~2.37 |
| **Tabular self-play, time-average policy** | **~1.6** ⬇ |
| (for reference) collapsed Phase-2 RLVR policy | 1.75 |
| (for reference) Nash equilibrium | 0.0 |

Self-play drives exploitability **down** — past the collapsed RLVR policy's 1.75 —
without ever consulting the solver during training (the solver is used only at
eval time, exactly as in Phase 2). This is the inverse of the Phase-2 trajectory,
and it confirms the *formulation* is sound independent of the LLM substrate: when
the policy can represent mixing and is rewarded by game outcomes, mixing emerges
and exploitability falls.

### A Phase-2 echo: argmaxing the average re-collapses it

Greedy (argmax) exploitability of the *same* average policy moves the wrong way
(1.75 → ~2.4). Collapsing a mixed near-equilibrium onto its modal action throws
away the mix — the identical failure mode Phase 2 diagnosed, now visible from the
opposite direction. The mixed lens is the correct one for a mixing game; the
greedy number is shown only to make the point.

## Honesty / scope

- This is a **tabular** result on Leduc. It demonstrates the self-play *loop and
  reward design* reduce exploitability; it does **not** claim anything about an LLM
  policy. The LLM run remains gated behind `--live` on a funded account.
- REINFORCE self-play is stochastic. The number above is the time-average at a
  fixed seed; across seeds the average lands ~1.1–1.7 — always well below the
  uniform 2.37, and at convergence below the RLVR 1.75. It does **not** reach Nash
  (0.0); CFR remains the way to *solve* the game. The defensible claim is
  directional ("self-play reduces exploitability"), matching the design spec.

## Reproduce

```bash
make selfplay-tabular          # headline run (~30s), prints before/after JSON
make selfplay-tabular-smoke    # fast mechanics + drop check
# or, directly:
python -m pokereval.cli rl-selfplay --variant leduc --preset real --tabular --iterations 1000
```

Everything here runs offline with **no API keys and no Tinker account**, and is
covered by `tests/rl/test_selfplay_tabular.py` (the exploitability drop is asserted
deterministically at a fixed seed).
