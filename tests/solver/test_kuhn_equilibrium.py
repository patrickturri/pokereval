"""Validate the CFR solver against Kuhn poker's closed-form Nash equilibrium.

The analytic-math tests (closed-form strategy shape) run without OpenSpiel.
The solver-comparison test is ``@pytest.mark.openspiel`` and skips when
``pyspiel`` is unavailable.
"""

import pytest

from pokereval.solver.kuhn_equilibrium import (
    ALPHA_MAX,
    analytic_kuhn_strategy,
    alpha_of_strategy,
)

# CFR iterations + tolerance for the solver-vs-analytic comparison. At 8000
# iterations every action probability is within ~0.007 of the closed form;
# 0.02 leaves comfortable deterministic margin.
ITERATIONS = 8000
TOL = 0.02

THIRD = 1.0 / 3.0


# --------------------------------------------------------------------------- #
# Pure analytic-strategy properties (no OpenSpiel required).
# --------------------------------------------------------------------------- #

def test_analytic_strategy_is_a_valid_distribution():
    for alpha in (0.0, 0.1, 0.2, ALPHA_MAX):
        strat = analytic_kuhn_strategy(alpha)
        assert len(strat) == 12  # Kuhn has 12 decision info states
        for key, dist in strat.items():
            assert abs(sum(dist.values()) - 1.0) < 1e-9, key
            for p in dist.values():
                assert -1e-9 <= p <= 1.0 + 1e-9, key


def test_analytic_strategy_encodes_the_known_invariants():
    alpha = 0.2
    s = analytic_kuhn_strategy(alpha)
    # Player 0 opening: King value-bets at 3x the Jack bluff rate; Queen never bets.
    assert s["2"]["bet"] == pytest.approx(3.0 * alpha)
    assert s["0"]["bet"] == pytest.approx(alpha)
    assert s["1"]["bet"] == pytest.approx(0.0)
    # Player 1 facing a bet: fold J, call K, call Q exactly 1/3.
    assert s["0b"]["call"] == pytest.approx(0.0)
    assert s["2b"]["call"] == pytest.approx(1.0)
    assert s["1b"]["call"] == pytest.approx(THIRD)
    # Player 1 after a check: bet K, check Q, bluff J exactly 1/3.
    assert s["2p"]["bet"] == pytest.approx(1.0)
    assert s["1p"]["bet"] == pytest.approx(0.0)
    assert s["0p"]["bet"] == pytest.approx(THIRD)
    # Player 0 after check-then-bet: fold J, call K, call Q at alpha + 1/3.
    assert s["0pb"]["call"] == pytest.approx(0.0)
    assert s["2pb"]["call"] == pytest.approx(1.0)
    assert s["1pb"]["call"] == pytest.approx(alpha + THIRD)


def test_alpha_recovery_round_trips():
    for alpha in (0.0, 0.07, 0.2, ALPHA_MAX):
        assert alpha_of_strategy(analytic_kuhn_strategy(alpha)) == pytest.approx(alpha)


def test_analytic_strategy_rejects_out_of_range_alpha():
    with pytest.raises(ValueError):
        analytic_kuhn_strategy(0.5)  # > 1/3
    with pytest.raises(ValueError):
        analytic_kuhn_strategy(-0.1)


# --------------------------------------------------------------------------- #
# Solver-vs-analytic comparison (requires OpenSpiel).
# --------------------------------------------------------------------------- #

@pytest.mark.openspiel
def test_cfr_solver_matches_closed_form_equilibrium():
    pytest.importorskip("pyspiel")
    from pokereval.solver.kuhn_equilibrium import solver_kuhn_strategy

    solved = solver_kuhn_strategy(ITERATIONS)

    # CFR's average policy lands on one member of the equilibrium family; recover
    # which one from P(bet | Jack), then check it is in the valid range.
    alpha = alpha_of_strategy(solved)
    assert 0.0 <= alpha <= ALPHA_MAX + TOL

    # Headline alpha-independent invariants, read straight off the solver:
    #   - King value-bets at 3x the Jack bluff rate (the classic bluff ratio),
    #   - Queen never opens,
    #   - the defender calls a bet with the Queen exactly one-third of the time.
    assert solved["2"]["bet"] == pytest.approx(3.0 * solved["0"]["bet"], abs=TOL)
    assert solved["1"]["bet"] == pytest.approx(0.0, abs=TOL)
    assert solved["1b"]["call"] == pytest.approx(THIRD, abs=TOL)
    assert solved["0p"]["bet"] == pytest.approx(THIRD, abs=TOL)

    # Full closed-form match: every info state, every action, vs analytic(alpha).
    analytic = analytic_kuhn_strategy(min(alpha, ALPHA_MAX))
    for key, dist in analytic.items():
        for action, prob in dist.items():
            assert solved[key][action] == pytest.approx(prob, abs=TOL), (
                f"{key}/{action}: solver={solved[key][action]:.4f} "
                f"analytic={prob:.4f} (alpha={alpha:.4f})"
            )
