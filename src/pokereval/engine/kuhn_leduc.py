"""OpenSpiel wrapper for the exactly-solvable games (Kuhn / Leduc).

This module knows nothing about LLMs. It loads the OpenSpiel game, maps the
engine's per-game action strings onto our generic :class:`Action` enum, and
enumerates every *player* decision node of the full game tree (chance nodes are
expanded, terminal nodes skipped) so the solver and spot builder can iterate
over a stable, deduplicated set of information states.
"""

from dataclasses import dataclass

import pyspiel

from ..interface.types import Action, GameVariant

_GAME_NAME = {GameVariant.KUHN: "kuhn_poker", GameVariant.LEDUC: "leduc_poker"}


def load_game(variant: GameVariant):
    """Load the OpenSpiel game for ``KUHN``/``LEDUC``; reject ``HOLDEM``."""
    if variant not in _GAME_NAME:
        raise ValueError(f"No OpenSpiel game for {variant}")
    return pyspiel.load_game(_GAME_NAME[variant])


def normalize_os_action(action_string: str, facing_bet: bool) -> Action:
    """Map an OpenSpiel action label to our :class:`Action`.

    Kuhn uses "Pass"/"Bet"; a Pass is a :attr:`Action.CHECK` when no bet is
    pending and a :attr:`Action.FOLD` when facing one. Leduc uses the explicit
    "Fold"/"Call"/"Raise"/"Check" labels.
    """
    s = action_string.strip().lower()
    if s == "fold":
        return Action.FOLD
    if s == "call":
        return Action.CALL
    if s == "raise":
        return Action.RAISE
    if s == "check":
        return Action.CHECK
    if s == "bet":
        return Action.BET
    if s == "pass":  # Kuhn: pass = fold if facing a bet, else check
        return Action.FOLD if facing_bet else Action.CHECK
    raise ValueError(f"Unrecognized OpenSpiel action string: {action_string!r}")


@dataclass
class NodeInfo:
    """A single information state of the game tree.

    ``os_legal`` maps our action *value* (e.g. ``"raise"``) to the OpenSpiel
    action id, and ``os_state`` is the live ``pyspiel.State`` at this node.
    """

    info_key: str
    player: int
    os_legal: dict[str, int]
    facing_bet: bool
    os_state: "pyspiel.State"


def iter_nodes(variant: GameVariant):
    """Yield one :class:`NodeInfo` per unique decision information state."""
    game = load_game(variant)

    def _walk(state):
        if state.is_terminal():
            return
        if state.is_chance_node():
            for action, _prob in state.chance_outcomes():
                yield from _walk(state.child(action))
            return
        player = state.current_player()
        info_key = state.information_state_string(player)
        labels = {a: state.action_to_string(player, a) for a in state.legal_actions()}
        # A pending "Call"/"Fold" option implies a bet we could be folding to.
        facing_bet = any(
            lbl.strip().lower() in ("call", "fold") for lbl in labels.values()
        )
        os_legal: dict[str, int] = {}
        for a, label in labels.items():
            act = normalize_os_action(label, facing_bet)
            os_legal[act.value] = a
        yield NodeInfo(info_key, player, os_legal, facing_bet, state)
        for a in state.legal_actions():
            yield from _walk(state.child(a))

    seen: set[str] = set()
    for node in _walk(game.new_initial_state()):
        if node.info_key in seen:
            continue
        seen.add(node.info_key)
        yield node
