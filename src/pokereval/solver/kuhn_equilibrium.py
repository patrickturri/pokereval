"""Closed-form Nash equilibrium of Kuhn poker — a ground-truth check for the CFR solver.

Kuhn poker (Kuhn, 1950) is the smallest non-trivial imperfect-information poker
game, and one of the very few with a fully *analytic* Nash equilibrium. That
makes it the ideal correctness anchor for this repo: the CFR solver in
``solver/nash.py`` defines "ground truth" for every downstream result
(eval exploitability, synth/RLVR Nash labels, the web demo), so it is worth
proving — not just asserting — that it reproduces the known equilibrium.

The equilibrium is a one-parameter family indexed by ``alpha = P(bet | Jack)``
for the first player, with ``alpha in [0, 1/3]``. Every other action
probability is pinned by ``alpha`` through indifference conditions. The
relationships below are textbook (see references); this module encodes them so a
test can check the solver against the closed form rather than a hand-transcribed
table of numbers.

Information-state keys match OpenSpiel's ``kuhn_poker`` exactly (the same keys
``nash_probs_by_key`` and ``iter_nodes`` produce), so the analytic strategy is
directly comparable to the solver output:

    card index 0 = Jack (J), 1 = Queen (Q), 2 = King (K)

    "<card>"      first action, player 0          actions: check / bet
    "<card>b"     player 1, facing player 0's bet  actions: fold / call
    "<card>p"     player 1, after player 0 checks  actions: check / bet
    "<card>pb"    player 0, after check-then-bet    actions: fold / call

References
----------
- H. W. Kuhn, "A Simplified Two-Person Poker," *Contributions to the Theory of
  Games*, vol. 1, pp. 97-103, 1950 — the original game and equilibrium analysis.
- Wikipedia, "Kuhn poker" — the standard statement of the one-parameter
  equilibrium family used here.
- M. Zinkevich et al., "Regret Minimization in Games with Incomplete
  Information," NeurIPS 2007 — the CFR algorithm whose average policy converges
  to one member of this family.
"""

from __future__ import annotations

CARD_NAMES: dict[str, str] = {"0": "Jack", "1": "Queen", "2": "King"}

# The free parameter is bounded: the first player bluffs the Jack at rate alpha
# and value-bets the King at rate 3*alpha, so 3*alpha <= 1 forces alpha <= 1/3.
ALPHA_MAX = 1.0 / 3.0


def analytic_kuhn_strategy(alpha: float) -> dict[str, dict[str, float]]:
    """Return the closed-form Kuhn Nash equilibrium for a given ``alpha``.

    Args:
        alpha: ``P(bet | Jack)`` for the first player, in ``[0, 1/3]``.

    Returns:
        ``{info_key: {action_value: probability}}`` keyed exactly like
        ``nash_probs_by_key(GameVariant.KUHN, ...)`` after mapping OpenSpiel
        action ids back to ``Action`` values. Every info state's probabilities
        sum to 1.

    Raises:
        ValueError: if ``alpha`` is outside ``[0, 1/3]``.
    """
    if not (0.0 - 1e-9 <= alpha <= ALPHA_MAX + 1e-9):
        raise ValueError(f"alpha must be in [0, 1/3]; got {alpha}")

    third = 1.0 / 3.0
    return {
        # ---- Player 0, first action (check / bet) ----
        # Jack: bluff at rate alpha. Queen: never bet. King: value-bet at 3*alpha.
        "0": {"check": 1.0 - alpha, "bet": alpha},
        "1": {"check": 1.0, "bet": 0.0},
        "2": {"check": 1.0 - 3.0 * alpha, "bet": 3.0 * alpha},
        # ---- Player 1, facing a bet (fold / call) ----
        # Jack: always fold. Queen: call exactly 1/3. King: always call.
        "0b": {"fold": 1.0, "call": 0.0},
        "1b": {"fold": 1.0 - third, "call": third},
        "2b": {"fold": 0.0, "call": 1.0},
        # ---- Player 1, after player 0 checks (check / bet) ----
        # Jack: bluff exactly 1/3. Queen: always check. King: always bet.
        "0p": {"check": 1.0 - third, "bet": third},
        "1p": {"check": 1.0, "bet": 0.0},
        "2p": {"check": 0.0, "bet": 1.0},
        # ---- Player 0, after check-then-bet (fold / call) ----
        # Jack: always fold. Queen: call at alpha + 1/3. King: always call.
        "0pb": {"fold": 1.0, "call": 0.0},
        "1pb": {"fold": 1.0 - (alpha + third), "call": alpha + third},
        "2pb": {"fold": 0.0, "call": 1.0},
    }


def alpha_of_strategy(strategy: dict[str, dict[str, float]]) -> float:
    """Recover ``alpha`` from a Kuhn strategy: it is simply ``P(bet | Jack)``."""
    return float(strategy["0"]["bet"])


def solver_kuhn_strategy(iterations: int = 8000) -> dict[str, dict[str, float]]:
    """Solve Kuhn with CFR and return the average policy keyed by ``Action`` value.

    Thin adapter over ``nash_probs_by_key`` that maps OpenSpiel action ids back
    to ``Action`` values (``check``/``bet``/``fold``/``call``), so the result is
    directly comparable to :func:`analytic_kuhn_strategy`.

    OpenSpiel is imported lazily so this module stays importable without
    ``pyspiel`` (only this function requires it).

    Args:
        iterations: CFR iterations; more = closer to a Nash equilibrium.

    Returns:
        ``{info_key: {action_value: probability}}``.
    """
    from pokereval.interface.types import GameVariant
    from pokereval.solver.nash import nash_probs_by_key
    from pokereval.engine.kuhn_leduc import iter_nodes

    probs = nash_probs_by_key(GameVariant.KUHN, iterations)
    id_to_value = {
        n.info_key: {os_id: value for value, os_id in n.os_legal.items()}
        for n in iter_nodes(GameVariant.KUHN)
    }
    out: dict[str, dict[str, float]] = {}
    for key, dist in probs.items():
        mapping = id_to_value[key]
        out[key] = {mapping[int(a)]: float(p) for a, p in dist.items()}
    return out
