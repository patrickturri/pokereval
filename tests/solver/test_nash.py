import pytest

pyspiel = pytest.importorskip("pyspiel")
pytestmark = pytest.mark.openspiel

from pokereval.engine.kuhn_leduc import load_game  # noqa: E402
from pokereval.interface.types import GameVariant  # noqa: E402
from pokereval.solver.nash import (  # noqa: E402
    exploitability_of_choices,
    nash_exploitability,
    nash_probs_by_key,
    solve_nash,
)


def test_nash_policy_is_near_optimal():
    game = load_game(GameVariant.KUHN)
    avg = solve_nash(game, iterations=2000)
    assert nash_exploitability(game, avg) < 0.05


def test_nash_probs_cover_all_decision_keys_and_sum_to_one():
    probs = nash_probs_by_key(GameVariant.KUHN, iterations=500)
    assert probs
    for _key, dist in probs.items():
        assert abs(sum(dist.values()) - 1.0) < 1e-6


def test_choices_policy_returns_nonnegative_float():
    probs = nash_probs_by_key(GameVariant.KUHN, iterations=500)
    choices = {k: int(max(dist, key=dist.get)) for k, dist in probs.items()}
    expl = exploitability_of_choices(GameVariant.KUHN, choices)
    assert isinstance(expl, float) and expl >= 0.0
