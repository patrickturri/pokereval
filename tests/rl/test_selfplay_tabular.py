"""Tests for the credential-free tabular self-play loop.

The pure-Python policy mechanics (legality, the REINFORCE update direction, the
frozen snapshot, the time-average) run with no OpenSpiel. The end-to-end
learning result — exploitability of the average policy drops well below the
uniform baseline — is OpenSpiel-gated and deterministic at a fixed seed.
"""

import dataclasses

import pytest

from pokereval.rl.selfplay.tabular import (
    TabularPolicy,
    FrozenDistSampler,
    _AverageTracker,
    _legal_from_prompt,
    _action_from_text,
    _softmax,
)


def _prompt(legal: str) -> str:
    # Minimal prompt that exposes the one line the policy keys off of.
    return f"You are playing leduc poker.\nLegal actions: {legal}\nACTION: <action>"


# --- pure-Python units ------------------------------------------------------

def test_legal_and_action_parsing():
    assert _legal_from_prompt(_prompt("check, raise")) == ["check", "raise"]
    assert _action_from_text("ACTION: raise") == "raise"
    assert _action_from_text("ACTION: RAISE 2") == "raise"


def test_softmax_temperature_zero_is_argmax():
    assert _softmax([0.0, 2.0, 1.0], 0.0) == [0.0, 1.0, 0.0]
    probs = _softmax([0.0, 0.0], 1.0)
    assert probs == pytest.approx([0.5, 0.5])


def test_sample_returns_only_legal_actions():
    pol = TabularPolicy(seed=1)
    p = _prompt("check, raise")
    comps = pol.sample(p, n=20, temperature=1.0)
    actions = {_action_from_text(c.text) for c in comps}
    assert actions <= {"check", "raise"}


def test_update_shifts_mass_toward_rewarded_action():
    pol = TabularPolicy(seed=0)
    p = _prompt("check, raise")
    before = pol.action_probs(p)["raise"]
    pol.update(p, "raise", advantage=1.0, lr=0.5)
    after = pol.action_probs(p)["raise"]
    assert after > before
    # a negative advantage on the same action moves mass the other way
    pol.update(p, "raise", advantage=-2.0, lr=0.5)
    assert pol.action_probs(p)["raise"] < after


def test_update_ignores_illegal_action():
    pol = TabularPolicy(seed=0)
    p = _prompt("check, raise")
    snapshot = dict(pol.action_probs(p))
    pol.update(p, "fold", advantage=1.0, lr=0.5)  # fold not legal here
    assert pol.action_probs(p) == snapshot


def test_snapshot_is_frozen_against_later_updates():
    pol = TabularPolicy(seed=0)
    p = _prompt("check, raise")
    pol.update(p, "raise", advantage=1.0, lr=0.5)
    frozen = pol.snapshot()
    frozen_prob = frozen.action_probs(p)["raise"]
    # keep training the live policy; the snapshot must not move
    for _ in range(5):
        pol.update(p, "raise", advantage=1.0, lr=0.5)
    assert frozen.action_probs(p)["raise"] == frozen_prob
    assert pol.action_probs(p)["raise"] > frozen_prob


def test_average_tracker_means_two_observations():
    pol = TabularPolicy(seed=0)
    p = _prompt("check, raise")
    pol.action_probs(p)                               # prime the row (uniform)
    avg = _AverageTracker()
    avg.observe(pol)                                  # uniform: raise=0.5
    pol.update(p, "raise", advantage=3.0, lr=0.5)     # push raise up
    avg.observe(pol)
    mean_raise = avg.distribution()[p]["raise"]
    assert 0.5 < mean_raise < pol.action_probs(p)["raise"]


def test_frozen_dist_sampler_greedy_picks_argmax():
    p = _prompt("check, raise")
    s = FrozenDistSampler({p: {"check": 0.2, "raise": 0.8}}, seed=0)
    comps = s.sample(p, n=5, temperature=0.0)
    assert {_action_from_text(c.text) for c in comps} == {"raise"}


def test_frozen_dist_sampler_unseen_prompt_is_uniform_legal():
    s = FrozenDistSampler({}, seed=0)
    p = _prompt("check, raise")
    comps = s.sample(p, n=10, temperature=1.0)
    assert {_action_from_text(c.text) for c in comps} <= {"check", "raise"}


# --- end-to-end learning result (OpenSpiel) ---------------------------------

@pytest.mark.openspiel
def test_tabular_selfplay_reduces_exploitability():
    """The headline offline claim, at a fast deterministic config: the
    time-average self-play policy is markedly less exploitable than the
    uniform-random start."""
    from pokereval.interface.types import GameVariant
    from pokereval.synth.generator import build_labeled_spots
    from pokereval.rl.dataset import split_spots
    from pokereval.rl.config import RLConfig
    from pokereval.rl.selfplay.tabular import tabular_selfplay_train

    variant = GameVariant.LEDUC
    spots = build_labeled_spots(variant, iterations=200)
    train, eval_spots = split_spots(spots, 0.2, 0)
    all_spots = list(train) + list(eval_spots)

    cfg = dataclasses.replace(
        RLConfig(), variant="leduc", num_steps=200, deals_per_step=10,
        hands_per_deal=6, learning_rate=0.4, temperature=1.0,
        opponent_refresh_every=10, eval_samples=16, seed=0,
    )
    result = tabular_selfplay_train(cfg, all_spots, eval_spots)

    assert len(result.step_returns) == cfg.num_steps
    # the loop starts at (near-)uniform exploitability...
    assert result.before.mixed_exploitability > 2.0
    # ...and self-play drives the average policy clearly below it.
    assert result.after.mixed_exploitability < result.before.mixed_exploitability - 0.25


@pytest.mark.openspiel
def test_tabular_smoke_preset_runs():
    """The `tabular_smoke` preset completes the full loop quickly and returns a
    populated result (mechanics check, no exploitability assertion)."""
    from pokereval.interface.types import GameVariant
    from pokereval.synth.generator import build_labeled_spots
    from pokereval.rl.dataset import split_spots
    from pokereval.rl.config import get_preset
    from pokereval.rl.selfplay.tabular import tabular_selfplay_train

    spots = build_labeled_spots(GameVariant.LEDUC, iterations=100)
    train, eval_spots = split_spots(spots, 0.2, 0)
    cfg = dataclasses.replace(get_preset("tabular_smoke"), variant="leduc")
    result = tabular_selfplay_train(cfg, list(train) + list(eval_spots), eval_spots)
    assert len(result.step_returns) == cfg.num_steps
    assert result.before is not None and result.after is not None
