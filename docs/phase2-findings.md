# Phase 2 — RLVR on Tinker: Methodology & Findings

## Method
- Single-step bandit over Leduc decision nodes (CFR/Nash-labeled by the synth generator).
- Reward = Nash probability of the chosen action (verifiable, dense, in [0, 1]).
- Group-relative advantages (GRPO-lite); LoRA fine-tune via Tinker `importance_sampling`.
- Headline metric: whole-policy Leduc **exploitability** vs the CFR solver (Nash = 0).

## Headline result — PENDING LIVE RUN
> Blocked on Tinker billing (account returns HTTP 402 until a payment method is
> added at https://tinker-console.thinkingmachines.ai/billing/balance). The
> pipeline is built and offline-verified; the numbers below come from a real run.
>
> - Exploitability: base `<X>` → post-RLVR `<Y>` chips/hand (`<Z>%` of the gap to Nash closed)
> - Verifiable-eval mean Nash-prob: `<a>` → `<b>`
> - Exact-match: `<c>` → `<d>`
> - Base model: `<id>` · steps: `<n>` · W&B run: `<url>`

## Reproduce
```bash
make rl-smoke      # offline loop check — mock backend, no Tinker account needed
make rl-train      # live run (needs Tinker billing + W&B)
```

`make rl-smoke` exercises the full sample → reward → advantage → datum → update
loop against a mock sampler and reports before/after exploitability for a
heuristic policy, proving the orchestration end-to-end without any network.
