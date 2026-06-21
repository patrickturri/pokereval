import pytest

pyspiel = pytest.importorskip("pyspiel")
pytestmark = pytest.mark.openspiel

from pokereval.interface.types import GameVariant
from pokereval.solver.nash import (
    solve_nash,
    nash_probs_by_key,
    nash_exploitability,
    exploitability_of_choices,
)
from pokereval.engine.kuhn_leduc import load_game


def test_nash_policy_is_near_optimal():
    game = load_game(GameVariant.KUHN)
    avg = solve_nash(game, iterations=2000)
    assert nash_exploitability(game, avg) < 0.05


def test_nash_probs_by_key_structure():
    probs = nash_probs_by_key(GameVariant.KUHN, iterations=2000)
    # Kuhn poker has 12 decision nodes
    assert len(probs) == 12
    for key, dist in probs.items():
        assert isinstance(key, str)
        assert isinstance(dist, dict)
        # All action keys are string ints
        for a_str, p in dist.items():
            assert isinstance(a_str, str)
            int(a_str)  # should not raise
            assert isinstance(p, float)
        # Probabilities sum to ~1
        assert abs(sum(dist.values()) - 1.0) < 1e-6


def test_exploitability_of_choices_deterministic():
    # A deterministic strategy (all-fold) should have positive exploitability
    from pokereval.engine.kuhn_leduc import iter_nodes

    # Build a pure fold/pass strategy: always pick first action (action id 0)
    choices = {}
    for node in iter_nodes(GameVariant.KUHN):
        choices[node.info_key] = 0  # always pick first legal action

    exp = exploitability_of_choices(GameVariant.KUHN, choices)
    assert isinstance(exp, float)
    assert exp > 0.0
