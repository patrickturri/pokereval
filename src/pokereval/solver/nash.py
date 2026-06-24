"""CFR solver, Nash action probabilities, and exploitability helpers.

CFR plays double duty in this project: it produces the exact(-ish) Nash policy
the verifiable grader scores against, and ``exploitability_of_choices`` turns a
model's *whole policy* into a single number — how many chips/round a best
responder could win against it — which is the headline metric for the small
games.
"""

from open_spiel.python import policy as os_policy
from open_spiel.python.algorithms import cfr, exploitability

from ..engine.kuhn_leduc import iter_nodes, load_game
from ..interface.types import GameVariant


def solve_nash(game, iterations: int = 2000):
    """Run vanilla CFR for ``iterations`` and return the average policy."""
    solver = cfr.CFRSolver(game)
    for _ in range(iterations):
        solver.evaluate_and_update_policy()
    return solver.average_policy()


def nash_exploitability(game, avg_policy) -> float:
    """Exploitability (chips/round vs. a best responder) of ``avg_policy``."""
    return float(exploitability.exploitability(game, avg_policy))


def nash_probs_by_key(
    variant: GameVariant, iterations: int = 2000
) -> dict[str, dict[str, float]]:
    """Solve the game and return ``{info_key: {os_action_id_str: prob}}``."""
    game = load_game(variant)
    avg = solve_nash(game, iterations)
    out: dict[str, dict[str, float]] = {}
    for node in iter_nodes(variant):
        dist = avg.action_probabilities(node.os_state)
        out[node.info_key] = {str(int(a)): float(p) for a, p in dist.items()}
    return out


def exploitability_of_choices(variant: GameVariant, choices: dict[str, int]) -> float:
    """Exploitability of a deterministic policy ``{info_key: os_action_id}``.

    Information states absent from ``choices`` keep ``TabularPolicy``'s default
    (uniform) distribution, so a partial policy is still gradeable.
    """
    game = load_game(variant)
    tab = os_policy.TabularPolicy(game)
    for node in iter_nodes(variant):
        if node.info_key not in choices:
            continue
        chosen = choices[node.info_key]
        legal = list(node.os_state.legal_actions())
        row = tab.policy_for_key(node.info_key)
        for i, a in enumerate(legal):
            row[i] = 1.0 if a == chosen else 0.0
    return float(exploitability.exploitability(game, tab))
