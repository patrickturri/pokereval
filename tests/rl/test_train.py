from pokereval.interface.types import State, GameVariant, Action
from pokereval.synth.types import LabeledSpot
from pokereval.rl.config import get_preset
from pokereval.rl.backend import FakeBackend
from pokereval.rl.sampler import MockSamplingClient
from pokereval.rl.train import train


def _spots(n):
    out = []
    for i in range(n):
        s = State(variant=GameVariant.LEDUC, hero_cards=["Js"], pot=1.0,
                  legal_actions=[Action.CHECK, Action.BET], info_state_key=f"k{i}",
                  meta={"os_legal": {"check": 0, "bet": 1}})
        out.append(LabeledSpot(state=s, nash_action_probs={"check": 0.7, "bet": 0.3}, tags=[]))
    return out


def test_loop_runs_and_calls_train_step():
    cfg = get_preset("smoke")          # num_steps=2, batch_spots=2, group_size=2
    be = FakeBackend(MockSamplingClient(fixed_action="check"))
    logs = []
    res = train(cfg, _spots(4), _spots(2), be, logger=logs.append)
    # one train_step per step
    assert len(be.calls) == cfg.num_steps
    # each step built batch_spots * group_size datums
    assert be.calls[0][0] == cfg.batch_spots * cfg.group_size
    assert len(res.step_rewards) == cfg.num_steps
    # 'check' has nash prob 0.7 -> mean reward should be ~0.7
    assert abs(res.step_rewards[0] - 0.7) < 1e-9
    assert len(logs) == cfg.num_steps
