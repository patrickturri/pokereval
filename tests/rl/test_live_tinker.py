import os
import pytest


@pytest.mark.tinker
def test_live_smoke_one_step():
    if not os.environ.get("TINKER_API_KEY"):
        pytest.skip("TINKER_API_KEY not set")
    from pokereval.interface.types import GameVariant
    from pokereval.synth.generator import build_labeled_spots
    from pokereval.rl.config import get_preset
    from pokereval.rl.dataset import split_spots
    from pokereval.rl.backend import TinkerBackend
    from pokereval.rl.train import train

    cfg = get_preset("smoke")
    spots = build_labeled_spots(GameVariant.LEDUC, cfg.iterations)
    tr, ev = split_spots(spots, cfg.eval_split, cfg.seed)
    backend = TinkerBackend(cfg)
    result = train(cfg, tr, ev, backend)
    assert len(result.step_rewards) == cfg.num_steps
