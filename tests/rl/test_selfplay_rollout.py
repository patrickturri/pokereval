import random
import pytest
from pokereval.interface.types import GameVariant
from pokereval.engine.kuhn_leduc import load_game
from pokereval.rl.sampler import MockSamplingClient
from pokereval.rl.selfplay.game import sample_deal
from pokereval.rl.selfplay.rollout import play_hand, play_group


def _first_legal(prompt: str) -> str:
    line = next(l for l in prompt.splitlines() if l.startswith("Legal actions:"))
    return line.split(":", 1)[1].split(",")[0].strip()


@pytest.mark.openspiel
def test_play_hand_returns_terminal_return_and_records_decisions():
    game = load_game(GameVariant.LEDUC)
    deal = sample_deal(game, random.Random(0))
    sampler = MockSamplingClient(action_for=_first_legal)
    encode = lambda s: [ord(c) for c in s]
    hr = play_hand(game, GameVariant.LEDUC, deal, sampler, sampler,
                   learner_seat=0, encode=encode, temperature=1.0)
    assert isinstance(hr.learner_return, float)
    assert len(hr.decisions) >= 1
    assert all(len(d.prompt_tokens) > 0 for d in hr.decisions)


@pytest.mark.openspiel
def test_learner_return_is_seat_specific_and_zero_sum():
    game = load_game(GameVariant.LEDUC)
    deal = sample_deal(game, random.Random(3))
    sampler = MockSamplingClient(action_for=_first_legal)
    encode = lambda s: [ord(c) for c in s]
    # Both seats use the same deterministic policy, so the SAME hand is played
    # regardless of which seat is "learner". learner_return is therefore the
    # per-seat showdown chips: seat 0 and seat 1 must sum to zero (zero-sum),
    # which verifies learner_return is read from the correct seat of returns().
    seat0 = play_hand(game, GameVariant.LEDUC, deal, sampler, sampler, 0, encode, 1.0)
    seat1 = play_hand(game, GameVariant.LEDUC, deal, sampler, sampler, 1, encode, 1.0)
    assert abs(seat0.learner_return + seat1.learner_return) < 1e-9


@pytest.mark.openspiel
def test_illegal_completion_does_not_crash():
    game = load_game(GameVariant.LEDUC)
    deal = sample_deal(game, random.Random(4))
    # "bet" is NOT a legal Leduc action label → forces the fold/first-legal fallback
    illegal = MockSamplingClient(fixed_action="bet")
    encode = lambda s: [ord(c) for c in s]
    hr = play_hand(game, GameVariant.LEDUC, deal, illegal, illegal, 0, encode, 1.0)
    assert isinstance(hr.learner_return, float)
    assert len(hr.decisions) >= 1  # the illegal completion is still recorded


@pytest.mark.openspiel
def test_play_group_returns_one_result_per_hand():
    game = load_game(GameVariant.LEDUC)
    deal = sample_deal(game, random.Random(5))
    sampler = MockSamplingClient(action_for=_first_legal)
    encode = lambda s: [ord(c) for c in s]
    results = play_group(game, GameVariant.LEDUC, deal, sampler, sampler, 0, encode, 1.0, hands=4)
    assert len(results) == 4
