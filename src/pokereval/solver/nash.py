"""CFR-based Nash solver for Kuhn/Leduc poker variants.

Two regret-minimization algorithms are exposed via the ``method`` argument:

- ``"cfr"`` — vanilla Counterfactual Regret Minimization (Zinkevich et al., 2007).
- ``"cfr+"`` — CFR+ (Tammelin, 2014): regret-matching⁺ (negative regrets are
  floored at zero) plus linear averaging of the iterates. CFR+ converges to a
  Nash equilibrium roughly an order of magnitude faster than vanilla CFR and is
  the algorithm behind *solving* heads-up limit Hold'em (Bowling, Burch,
  Johanson & Tammelin, "Heads-up limit hold'em poker is solved," Science, 2015).
  On Leduc it reaches a given exploitability in ~5× fewer iterations here.

Both return OpenSpiel ``TabularPolicy`` average policies and are interchangeable
everywhere a Nash policy is consumed (eval, synth labels, the web demo).

Public API
----------
solve_nash(game, iterations, method="cfr") -> TabularPolicy
    Run CFR / CFR+ for `iterations` steps and return the average policy.

nash_exploitability(game, avg_policy) -> float
    Compute the exploitability (NashConv / 2) of the given average policy.

nash_probs_by_key(variant, iterations, method="cfr") -> dict[str, dict[str, float]]
    Return action probabilities keyed by info-state string.

exploitability_of_choices(variant, choices) -> float
    Build a deterministic TabularPolicy from `choices` and compute its exploitability.
"""

from __future__ import annotations

import contextlib
import os
import sys

import pyspiel


@contextlib.contextmanager
def _suppress_native_stdout():
    """Silence C/C++-level writes to stdout (file descriptor 1).

    OpenSpiel's ``open_spiel.python.algorithms`` package eagerly imports its
    Python game registry, which prints lines like
    ``Optional module pokerkit_wrapper was not importable: ...`` directly to
    stdout from native code. A plain ``contextlib.redirect_stdout`` cannot catch
    that because it only rebinds ``sys.stdout`` in Python; we must redirect the
    underlying file descriptor.
    """
    saved_fd = os.dup(1)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stdout.flush()
        os.dup2(devnull_fd, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved_fd, 1)
        os.close(devnull_fd)
        os.close(saved_fd)


with _suppress_native_stdout():
    from open_spiel.python.algorithms import cfr
    from open_spiel.python.algorithms import exploitability as _exploitability_module
    from open_spiel.python import policy as os_policy

from pokereval.interface.types import GameVariant
from pokereval.engine.kuhn_leduc import load_game, iter_nodes

# Regret-minimization solvers, keyed by the ``method`` argument. CFR+ uses
# regret-matching⁺ and linear averaging and converges much faster (see module
# docstring); vanilla CFR is kept as the default for backward compatibility.
_SOLVERS = {
    "cfr": cfr.CFRSolver,
    "cfr+": cfr.CFRPlusSolver,
}


def _make_solver(game: pyspiel.Game, method: str):
    try:
        return _SOLVERS[method](game)
    except KeyError:
        raise ValueError(
            f"Unknown solver method {method!r}; choose one of {sorted(_SOLVERS)}"
        ) from None


def solve_nash(
    game: pyspiel.Game,
    iterations: int = 2000,
    method: str = "cfr",
) -> os_policy.TabularPolicy:
    """Run CFR / CFR+ for `iterations` steps and return the average policy.

    Parameters
    ----------
    game:
        A pyspiel.Game object (from load_game).
    iterations:
        Number of solver iterations to run (more = closer to Nash equilibrium).
    method:
        ``"cfr"`` (vanilla, default) or ``"cfr+"`` (faster-converging; see the
        module docstring). Both return an averaged TabularPolicy.

    Returns
    -------
    TabularPolicy
        The average policy across all iterations — this converges to a Nash
        equilibrium as iterations -> infinity.
    """
    solver = _make_solver(game, method)
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
    method: str = "cfr",
) -> dict[str, dict[str, float]]:
    """Solve Nash and return action probabilities keyed by info-state string.

    Parameters
    ----------
    variant:
        The GameVariant to solve (KUHN or LEDUC).
    iterations:
        Number of solver iterations.
    method:
        ``"cfr"`` (default) or ``"cfr+"`` — see :func:`solve_nash`.

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping of info_key -> {str(os_action_id): probability}.
        One entry per unique decision node (deduplicated by info_key).
    """
    game = load_game(variant)
    avg = solve_nash(game, iterations, method=method)
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


def exploitability_of_distribution(
    variant: GameVariant,
    dist_by_key: dict[str, dict[int, float]],
) -> float:
    """Build a MIXED TabularPolicy from per-info-state action distributions.

    Unlike ``exploitability_of_choices`` (deterministic), this assigns each legal
    action its given probability mass, so a policy that *mixes* (as Leduc Nash
    requires) is evaluated faithfully rather than collapsed to an argmax.

    Parameters
    ----------
    variant:
        The GameVariant to evaluate.
    dist_by_key:
        Mapping info_key -> {os_action_id: probability}. Probabilities are
        renormalized over legal actions; info keys absent (or with zero mass)
        retain the uniform default.

    Returns
    -------
    float
        Exploitability of the mixed strategy.
    """
    game = load_game(variant)
    tab = os_policy.TabularPolicy(game)
    for node in iter_nodes(variant):
        dist = dist_by_key.get(node.info_key)
        if not dist:
            continue
        legal = list(node.os_state.legal_actions())
        masses = [float(dist.get(a, 0.0)) for a in legal]
        total = sum(masses)
        if total <= 0.0:
            continue
        row = tab.policy_for_key(node.info_key)
        for i, m in enumerate(masses):
            row[i] = m / total
    return float(_exploitability_module.exploitability(game, tab))
