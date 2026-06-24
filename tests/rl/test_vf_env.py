import pytest
from pokereval.interface.types import State, GameVariant, Action
from pokereval.synth.types import LabeledSpot

vf = pytest.importorskip("verifiers")  # skip whole module if verifiers absent


def _spots():
    s = State(variant=GameVariant.LEDUC, hero_cards=["Js"], pot=1.0,
              legal_actions=[Action.CHECK, Action.BET], info_state_key="k0",
              meta={"os_legal": {"check": 0, "bet": 1}})
    return [LabeledSpot(state=s, nash_action_probs={"check": 0.6, "bet": 0.4}, tags=[])]


def test_env_builds_with_dataset():
    from pokereval.rl.vf_env import build_vf_env
    env = build_vf_env(_spots())
    assert len(env.dataset) == 1
