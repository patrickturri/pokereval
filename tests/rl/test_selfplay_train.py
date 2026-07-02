import pytest
from pokereval.rl.config import get_preset
from pokereval.rl.backend import FakeBackend
from pokereval.rl.sampler import MockSamplingClient


def _first_legal(prompt: str) -> str:
    line = next(l for l in prompt.splitlines() if l.startswith("Legal actions:"))
    return line.split(":", 1)[1].split(",")[0].strip()


@pytest.mark.openspiel
def test_selfplay_train_runs_and_calls_train_step_per_step():
    from pokereval.rl.selfplay.train import selfplay_train
    cfg = get_preset("smoke")  # num_steps=2
    be = FakeBackend(MockSamplingClient(action_for=_first_legal))
    result = selfplay_train(cfg, all_spots=[], eval_spots=[], backend=be)
    assert len(be.calls) == cfg.num_steps          # one train_step per step
    assert len(result.step_returns) == cfg.num_steps


import dataclasses


@pytest.mark.openspiel
def test_opponent_refresh_cadence():
    from pokereval.rl.selfplay.train import selfplay_train
    cfg = dataclasses.replace(get_preset("smoke"), num_steps=3, opponent_refresh_every=1)
    be = FakeBackend(MockSamplingClient(action_for=_first_legal))
    selfplay_train(cfg, all_spots=[], eval_spots=[], backend=be)
    # 1 initial snapshot + refreshes at steps 1 and 2 = 3
    assert be.snapshot_calls == 3


@pytest.mark.openspiel
def test_eval_curve_records_at_eval_every_cadence():
    from pokereval.rl.selfplay.train import selfplay_train
    # eval_every=2 over 4 steps → eval after the steps whose 1-indexed count is a
    # multiple of eval_every, i.e. completed steps 1 and 3 (0-indexed).
    cfg = dataclasses.replace(get_preset("smoke"), num_steps=4, eval_every=2)
    be = FakeBackend(MockSamplingClient(action_for=_first_legal))
    result = selfplay_train(cfg, all_spots=[], eval_spots=[], backend=be)
    assert [p.step for p in result.eval_curve] == [1, 3]


@pytest.mark.openspiel
def test_eval_curve_points_carry_wellformed_reports():
    from pokereval.rl.selfplay.train import selfplay_train
    from pokereval.rl.evaluate import EvalReport
    cfg = dataclasses.replace(get_preset("smoke"), num_steps=2, eval_every=1)
    be = FakeBackend(MockSamplingClient(action_for=_first_legal))
    result = selfplay_train(cfg, all_spots=[], eval_spots=[], backend=be)
    # eval_every=1 → a curve point after every step; the final point mirrors `after`.
    assert [p.step for p in result.eval_curve] == [0, 1]
    assert all(isinstance(p.report, EvalReport) for p in result.eval_curve)
    assert result.eval_curve[-1].report.mixed_exploitability == result.after.mixed_exploitability


@pytest.mark.openspiel
def test_eval_curve_disabled_when_eval_every_nonpositive():
    from pokereval.rl.selfplay.train import selfplay_train
    cfg = dataclasses.replace(get_preset("smoke"), num_steps=2, eval_every=0)
    be = FakeBackend(MockSamplingClient(action_for=_first_legal))
    result = selfplay_train(cfg, all_spots=[], eval_spots=[], backend=be)
    assert result.eval_curve == []
