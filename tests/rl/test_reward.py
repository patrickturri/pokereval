import math
from pokereval.interface.types import State, GameVariant, Action
from pokereval.synth.types import LabeledSpot
from pokereval.rl.reward import spot_reward, kl_to_nash_reward, ILLEGAL_PENALTY


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


def test_kl_reward_is_log_nash_minus_logpi():
    spot = _spot({"check": 0.7, "bet": 0.3})
    # action_logprob = log pi(a); reward = log nash(a) - log pi(a)
    r = kl_to_nash_reward(spot, "ACTION: check", action_logprob=-2.0)
    assert r == math.log(0.7) - (-2.0)


def test_kl_reward_rewards_rarer_action_more_at_equal_nash():
    # Equal Nash prob, but action sampled with LOWER pi gets HIGHER reward
    # (the entropy term that prevents mode collapse).
    spot = _spot({"check": 0.5, "bet": 0.5})
    confident = kl_to_nash_reward(spot, "ACTION: check", action_logprob=-0.1)
    rare = kl_to_nash_reward(spot, "ACTION: check", action_logprob=-3.0)
    assert rare > confident


def test_kl_reward_penalizes_illegal():
    spot = _spot({"check": 0.7, "bet": 0.3})
    assert kl_to_nash_reward(spot, "ACTION: fold", action_logprob=-1.0) == ILLEGAL_PENALTY
