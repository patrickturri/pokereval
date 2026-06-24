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
