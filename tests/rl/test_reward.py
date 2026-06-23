from pokereval.interface.types import State, GameVariant, Action
from pokereval.synth.types import LabeledSpot
from pokereval.rl.reward import spot_reward


def _spot(probs):
    state = State(
        variant=GameVariant.LEDUC,
        hero_cards=["Js"],
        pot=2.0,
        legal_actions=[Action.CHECK, Action.BET],
        info_state_key="k1",
        meta={"os_legal": {"check": 0, "bet": 1}},
    )
    return LabeledSpot(state=state, nash_action_probs=probs, tags=["mixed"])


def test_reward_is_nash_prob_of_chosen_action():
    spot = _spot({"check": 0.7, "bet": 0.3})
    r, dec = spot_reward(spot, "reasoning...\nACTION: check")
    assert r == 0.7
    assert dec.action == Action.CHECK


def test_illegal_action_scores_zero():
    spot = _spot({"check": 0.7, "bet": 0.3})
    r, dec = spot_reward(spot, "ACTION: fold")  # fold not legal here
    assert r == 0.0
    assert dec is None


def test_unparseable_scores_zero():
    spot = _spot({"check": 0.7, "bet": 0.3})
    r, dec = spot_reward(spot, "I have no idea")
    assert r == 0.0
    assert dec is None
