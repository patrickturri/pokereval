# Phase 3 — Self-Play RL: Design Spec

**Date:** 2026-06-23
**Status:** Approved design, pre-implementation
**Branch context:** `phase1-eval-suite` (Phase 2 RLVR complete; this is the principled follow-up)

## Motivation

Phase 2 established a clean negative result: single-step RLVR against a per-decision
Nash oracle **systematically collapses to a deterministic modal policy and *raises*
exploitability** (greedy 1.05 → 1.75), reproduced across two reward designs. The failure
is structural — a bandit framing optimizes each decision in isolation, so its optimum is
the per-state argmax, but Leduc equilibrium *requires mixing*.

The fix is a different problem formulation, not a better per-decision reward:
**self-play over full hands, reward = chips at showdown, no solver oracle**, where mixing
emerges from game outcomes (how Pluribus is trained). This spec covers building that
harness, validated offline against the mock backend, with the live training run gated
behind `--live` (the Tinker account is currently billing-blocked).

## Settled decisions (from Phase 2 findings + brainstorming)

- **Episodic, sparse reward** = chips at showdown (`state.returns()[seat]`). No oracle.
- **Opponent = frozen snapshot, periodically refreshed** (NFSP-flavored fictitious
  self-play). Learner trains against a frozen copy of its own weights; every K steps the
  opponent is refreshed to the latest learner. Reward is the **learner-seat** return only.
  Initial opponent = the base model (snapshot at step 0). The defensible claim is
  "exploitability vs CFR drops across refreshes," not "reaches Nash."
- **Credit assignment = deal-conditioned GRPO baseline.** Sample G complete hands from the
  *same* dealt cards/board (same chance outcomes); the baseline is the mean learner return
  over that group, so the deal's intrinsic card luck cancels and the advantage isolates
  decision quality. One scalar advantage per hand, fanned out to every learner decision in
  that hand (Leduc hands are 1–3 learner decisions — too short to need discounting or a
  value function).
- **Eval is identical to Phase 2** (`evaluate_policy`, unchanged) — greedy + mixed
  exploitability vs CFR — so the self-play result drops into the same results table.

## Architecture

New package `src/pokereval/rl/selfplay/`, four focused modules. Each communicates through
the existing `SamplingClient` / `Backend` / `build_datum` / `group_advantages` interfaces so
the new code reuses the Phase 2 machinery rather than duplicating it.

### `game.py` — OpenSpiel hand stepper
Thin wrapper over `pyspiel` Leduc. Responsibilities:
- **Deal sampling:** sample chance outcomes to fix a *deal* (both players' hole cards +
  board card). A deal is the unit the GRPO group is conditioned on. The deal is captured as
  the sequence of chance actions so it can be **replayed** to start G fresh hands from
  identical cards.
- **Decision-node `State` construction:** for the current player's decision node, build the
  same `State` the synth generator builds — `hero_cards=[info_key]`, `legal_actions` from
  `os_legal`, `meta={"os_legal": ..., "facing_bet": ...}`, `info_state_key=info_key`. This
  is the existing bridge that makes `render_state` / `parse_decision` work unchanged. Factor
  the construction so `synth/generator.py` and `selfplay/game.py` share one helper (extract
  from the generator if not already reusable) — avoids two divergent copies.
- **Stepping:** apply a chosen `Action` by mapping it back through `os_legal` to the
  OpenSpiel action id and calling `state.child(...)`. Auto-resolve intervening chance nodes
  (e.g. the public board card) by sampling within the fixed deal.
- **Terminal:** expose `is_terminal` and `returns()` (per-seat chips).

### `rollout.py` — one hand, learner vs frozen opponent
`play_hand(deal, learner_sampler, opponent_sampler, learner_seat) -> HandResult` where
`HandResult` carries the learner-seat terminal return and a list of learner decision records
`(prompt_tokens, completion)`:
- At a **learner** node: `render_state` → `learner_sampler.sample(prompt, n=1, temperature)`
  → `parse_decision`. Illegal/unparseable → **fold** (safe legal default; folding usually
  loses → negative advantage → the model learns legality from outcomes, no parser penalty
  reward term). Record `(backend.encode(prompt), completion)`. Step.
- At an **opponent** node: sample from `opponent_sampler` the same way, but **do not record**
  (no gradient through the frozen opponent). Step.
- `play_group(deal, ..., G) -> list[HandResult]`: replays the same deal G times.

### `train.py` — GRPO self-play loop
Mirrors the shape of the existing `rl/train.py` and returns the same `TrainResult`
(`before`/`after` `EvalReport`, `step_rewards`). Per step:
1. Draw B deals (alternating `learner_seat` across deals so both positions are trained).
2. For each deal, `play_group` G hands; collect learner returns `r_1..r_G` and decision
   records.
3. `advs = group_advantages([r_1..r_G])` (deal-conditioned baseline — reuses the existing
   normalizer).
4. For each hand g, for each recorded learner decision, append
   `build_datum(prompt_tokens, comp.tokens, comp.logprobs, advs[g])`.
5. `backend.train_step(data, lr)`.
6. Every `opponent_refresh_every` steps, refresh the frozen opponent via
   `backend.snapshot()`.
7. Periodic eval via the existing `evaluate_policy` (greedy + mixed exploitability vs CFR).
   `step_rewards` logs mean learner return per step (interpretable progress signal).

### Backend extension — `snapshot() -> SamplingClient`
New method on the `Backend` protocol producing a **frozen** sampler bound to the current
weights:
- **`TinkerBackend.snapshot`:** save a LoRA checkpoint of current weights and construct a
  sampling client from that checkpoint (frozen — unaffected by subsequent `train_step`s).
- **`FakeBackend.snapshot`:** return a copy of the current mock sampler.

This is the reusable checkpoint plumbing; it also unblocks the deferred base-vs-trained demo
(previously tracked as option 2) at no extra cost.

## Data flow (one step)

```
for each of B deals:
    deal d = sample_chance()                 # fixed cards + board
    learner_seat = d_index % 2               # alternate positions
    results = play_group(d, learner, frozen_opp, learner_seat, G)
    returns = [hr.learner_return for hr in results]
    advs    = group_advantages(returns)      # (r - mean_d) / std_d  → card luck cancels
    for hr, adv in zip(results, advs):
        for (prompt_tokens, comp) in hr.decisions:
            data.append(build_datum(prompt_tokens, comp.tokens, comp.logprobs, adv))
backend.train_step(data, lr)
if step % opponent_refresh_every == 0:
    frozen_opp = backend.snapshot()
```

No oracle is consulted at any point in training. The solver is used only at eval time, to
measure exploitability — exactly as in Phase 2.

## Configuration

Extend `RLConfig` (or a `SelfPlayConfig` subset) with:
- `deals_per_step` (B) — distinct deals per step.
- `hands_per_deal` (G) — group size for the deal-conditioned baseline.
- `opponent_refresh_every` (K) — steps between frozen-opponent refreshes.
- Reuse existing `learning_rate`, `temperature`, `num_steps`, `eval_*`, `seed`,
  `base_model`, `lora_rank`, `max_tokens`.

Presets `smoke` (tiny: a couple steps, B=2, G=2, K=1 — offline loop check) and `real`
(meaningful B/G/steps for the gated live run).

## CLI / Make / live-gating

- New `rl-selfplay` subcommand mirroring `rl-train`:
  `--variant --preset {smoke,real} --live --wandb` plus overrides
  `--steps --deals --hands --refresh-every --lr --temperature`. Reuse `_preflight_billing`
  so `--live` fails fast on a dead/blocked balance instead of a ~1h retry loop.
- Makefile: `selfplay-smoke` (offline, mock backend, no account) and `selfplay-train`
  (live, needs Tinker billing + W&B).
- Output payload mirrors `rl-train`'s JSON (`before`/`after` EvalReport, `step_rewards`).

## Testing (all offline, no Tinker funds)

OpenSpiel-marked where `pyspiel` is required; pure-Python otherwise.
1. **End-to-end smoke:** `rl-selfplay --preset smoke` (mock backend) runs the full loop,
   produces non-empty training data, and calls `backend.train_step` the expected number of
   times.
2. **Rollout correctness:** a played hand reaches terminal, `learner_return` matches
   `state.returns()[learner_seat]`, learner decisions are collected (opponent decisions are
   not), and an injected illegal/unparseable completion resolves to `fold`.
3. **Deal-conditioned advantage:** for a group where the deal value is constant, the
   baseline cancels it and advantages reflect only return spread (a degenerate all-equal
   group → all-zero advantages, matching `group_advantages`).
4. **Snapshot cadence:** the frozen opponent is refreshed exactly every K steps; between
   refreshes the opponent sampler identity is stable.

## Scope

**In scope:** the full offline-validated self-play harness (`game`/`rollout`/`train`),
`backend.snapshot()` plumbing, eval reuse, CLI/Make wiring, and the test suite above.

**Out of scope (YAGNI / gated):**
- The live training run itself — needs Tinker funds; gated behind `--live`.
- Discounting / value functions — Leduc hands are 1–3 learner decisions; trajectory-level
  scalar advantage is sufficient.
- AIVAT / seat-swap duplicate variance reduction — deal-conditioning already cancels the
  dominant card-luck variance; add only if empirical variance is still too high.

## Risks

- **Live snapshot via Tinker:** depends on checkpoint save + sampling-from-checkpoint working
  concurrently with an active training client. The offline path is unaffected; the live path
  is gated, so this risk cannot block the deliverable.
- **Mock backend does not learn:** `FakeBackend` is a fixed first-legal-action heuristic, so
  offline both learner and opponent behave identically and the snapshot refresh is exercised
  structurally but not behaviorally. The offline build therefore proves the loop's
  *mechanics* (data shapes, cadence, eval wiring, no crashes), **not** a learning result.
  The actual exploitability-reduction claim requires the gated live run on a funded account.

## Success criteria

- `make selfplay-smoke` runs green offline with no account.
- The four test groups pass.
- A live `rl-selfplay --preset real --live` run is ready to execute the moment the Tinker
  balance is topped up, emitting `before`/`after` exploitability in the same format as the
  Phase 2 table — so the headline question (does self-play drive exploitability *down*?) can
  be answered with one funded run.
