"""CFR-based Nash solver for Kuhn/Leduc poker variants.

Public API
----------
solve_nash(game, iterations) -> TabularPolicy
    Run CFR for `iterations` steps and return the average policy.

nash_exploitability(game, avg_policy) -> float
    Compute the exploitability (NashConv / 2) of the given average policy.

nash_probs_by_key(variant, iterations) -> dict[str, dict[str, float]]
    Return action probabilities keyed by info-state string.

exploitability_of_choices(variant, choices) -> float
    Build a deterministic TabularPolicy from `choices` and compute its exploitability.
"""

from __future__ import annotations

import pyspiel
from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import exploitability as _exploitability_module
from open_spiel.python import policy as os_policy

from pokereval.interface.types import GameVariant
from pokereval.engine.kuhn_leduc import load_game, iter_nodes


def solve_nash(game: pyspiel.Game, iterations: int = 2000) -> os_policy.TabularPolicy:
    """Run CFR for `iterations` steps and return the average policy.

    Parameters
    ----------
    game:
        A pyspiel.Game object (from load_game).
    iterations:
        Number of CFR iterations to run (more = closer to Nash equilibrium).

    Returns
    -------
    TabularPolicy
        The average policy across all iterations — this converges to a Nash
        equilibrium as iterations -> infinity.
    """
    solver = cfr.CFRSolver(game)
    for _ in range(iterations):
        solver.evaluate_and_update_policy()
    return solver.average_policy()


def nash_exploitability(
    game: pyspiel.Game,
    avg_policy: os_policy.TabularPolicy,
) -> float:
    """Compute the exploitability of `avg_policy` under `game`.

    Exploitability is defined as the average of the best-response values
    for each player (NashConv / num_players). A Nash equilibrium has
    exploitability = 0.

    Parameters
    ----------
    game:
        The pyspiel.Game the policy was solved for.
    avg_policy:
        A TabularPolicy (e.g. from solve_nash).

    Returns
    -------
    float
        The exploitability value. Near-zero means near-Nash.
    """
    return float(_exploitability_module.exploitability(game, avg_policy))


def nash_probs_by_key(
    variant: GameVariant,
    iterations: int = 2000,
) -> dict[str, dict[str, float]]:
    """Solve Nash and return action probabilities keyed by info-state string.

    Parameters
    ----------
    variant:
        The GameVariant to solve (KUHN or LEDUC).
    iterations:
        Number of CFR iterations.

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping of info_key -> {str(os_action_id): probability}.
        One entry per unique decision node (deduplicated by info_key).
    """
    game = load_game(variant)
    avg = solve_nash(game, iterations)
    out: dict[str, dict[str, float]] = {}
    for node in iter_nodes(variant):
        # action_probabilities returns {action_id (int): probability (float)}
        dist = avg.action_probabilities(node.os_state)
        out[node.info_key] = {str(int(a)): float(p) for a, p in dist.items()}
    return out


def exploitability_of_choices(
    variant: GameVariant,
    choices: dict[str, int],
) -> float:
    """Build a deterministic TabularPolicy and return its exploitability.

    For info keys present in `choices`, the corresponding action is given
    probability 1.0. Missing keys retain the uniform default from TabularPolicy.

    Parameters
    ----------
    variant:
        The GameVariant to evaluate.
    choices:
        Mapping of info_key -> os_action_id (an integer from legal_actions()).

    Returns
    -------
    float
        Exploitability of the deterministic strategy defined by `choices`.
    """
    game = load_game(variant)
    tab = os_policy.TabularPolicy(game)
    for node in iter_nodes(variant):
        if node.info_key not in choices:
            continue
        chosen = choices[node.info_key]
        legal = list(node.os_state.legal_actions())
        # policy_for_key returns a mutable numpy array over legal-action order
        row = tab.policy_for_key(node.info_key)
        for i, a in enumerate(legal):
            row[i] = 1.0 if a == chosen else 0.0
    return float(_exploitability_module.exploitability(game, tab))
