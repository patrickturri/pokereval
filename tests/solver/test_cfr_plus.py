"""CFR+ solver: faster convergence than vanilla CFR, same Nash target.

CFR+ (Tammelin, 2014) uses regret-matching⁺ and linear averaging and reaches a
given exploitability in far fewer iterations than vanilla CFR. These tests pin
that empirical claim at fixed, small iteration counts (deterministic — the
solvers carry no randomness), so the result is reproducible offline with no
account.
"""

import pytest

pyspiel = pytest.importorskip("pyspiel")
pytestmark = pytest.mark.openspiel

from pokereval.interface.types import GameVariant
from pokereval.solver.nash import (
    solve_nash,
    nash_exploitability,
    nash_probs_by_key,
)
from pokereval.engine.kuhn_leduc import load_game


def test_unknown_method_raises():
    game = load_game(GameVariant.KUHN)
    with pytest.raises(ValueError, match="Unknown solver method"):
        solve_nash(game, iterations=1, method="not-a-solver")


def test_cfr_plus_converges_faster_on_leduc():
    # Leduc is the harder game (bigger tree, more mixing), so the convergence-rate
    # gap between the two algorithms is clear at a small, cheap iteration count.
    game = load_game(GameVariant.LEDUC)
    iters = 50
    expl_cfr = nash_exploitability(game, solve_nash(game, iters, method="cfr"))
    expl_plus = nash_exploitability(game, solve_nash(game, iters, method="cfr+"))
    # CFR+ is empirically ~5x lower here (~0.034 vs ~0.189); assert a robust 2x.
    assert expl_plus < expl_cfr / 2.0
    # And it is already near-Nash at 50 iterations, where vanilla CFR is not.
    assert expl_plus < 0.1


def test_cfr_plus_reaches_nash_on_kuhn():
    game = load_game(GameVariant.KUHN)
    avg = solve_nash(game, iterations=2000, method="cfr+")
    assert nash_exploitability(game, avg) < 0.01


def test_cfr_plus_probs_by_key_matches_cfr_structure():
    # Same info-state coverage and valid distributions regardless of method.
    probs = nash_probs_by_key(GameVariant.KUHN, iterations=500, method="cfr+")
    assert len(probs) == 12  # Kuhn has 12 decision nodes
    for dist in probs.values():
        assert abs(sum(dist.values()) - 1.0) < 1e-6
