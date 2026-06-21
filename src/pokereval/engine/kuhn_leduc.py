from dataclasses import dataclass
import pyspiel
from ..interface.types import Action, GameVariant

_GAME_NAME = {GameVariant.KUHN: "kuhn_poker", GameVariant.LEDUC: "leduc_poker"}


def load_game(variant: GameVariant):
    if variant not in _GAME_NAME:
        raise ValueError(f"No OpenSpiel game for {variant}")
    return pyspiel.load_game(_GAME_NAME[variant])


def normalize_os_action(action_string: str, facing_bet: bool) -> Action:
    """Map an OpenSpiel action string to our Action enum.

    Observed action strings per game:
    - Kuhn poker: 'Pass', 'Bet'
    - Leduc poker: 'Fold', 'Call', 'Raise'

    Kuhn semantics:
      - 'Pass' when not facing a bet = CHECK (check behind)
      - 'Pass' when facing a bet     = FOLD  (give up)
      - 'Bet'  when not facing a bet = BET   (open the betting)
      - 'Bet'  when facing a bet     = CALL  (put in chips to match)

    Leduc semantics:
      - 'Fold'  = FOLD (always)
      - 'Call'  when not facing a bet = CHECK (no chips to call, free action)
      - 'Call'  when facing a bet     = CALL  (match the bet)
      - 'Raise' = RAISE (always)
    """
    s = action_string.strip().lower()

    if s == "fold":
        return Action.FOLD
    if s == "raise":
        return Action.RAISE
    if s == "pass":
        # Kuhn: pass = fold if facing a bet, else check
        return Action.FOLD if facing_bet else Action.CHECK
    if s == "bet":
        # Kuhn: bet = call if facing a bet, else bet
        return Action.CALL if facing_bet else Action.BET
    if s == "call":
        # Leduc: call = call if facing a bet, else check (free action)
        return Action.CALL if facing_bet else Action.CHECK
    if s == "check":
        return Action.CHECK

    raise ValueError(f"Unrecognized OpenSpiel action string: {action_string!r}")


@dataclass
class NodeInfo:
    info_key: str
    player: int
    os_legal: dict  # action value (str) -> OpenSpiel action id (int)
    facing_bet: bool
    os_state: "pyspiel.State"


def iter_nodes(variant: GameVariant):
    """Enumerate every unique player decision node in the full game tree.

    Chance nodes are expanded automatically. Terminal nodes are skipped.
    Yields one NodeInfo per unique information state (deduplicated by info_key).
    """
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
        # facing_bet: a 'Fold' option (Leduc) or the Pass+Bet combo after a bet (Kuhn)
        # means there is a pending bet.  We detect this by checking if 'fold' or
        # 'pass' appears alongside a 'bet'/'call'/'raise' — simplest proxy: Fold present.
        raw_labels = {s.strip().lower() for s in labels.values()}
        facing_bet = "fold" in raw_labels
        os_legal: dict = {}
        for a, label in labels.items():
            act = normalize_os_action(label, facing_bet)
            os_legal[act.value] = a
        yield NodeInfo(info_key, player, os_legal, facing_bet, state)
        for a in state.legal_actions():
            yield from _walk(state.child(a))

    seen: set = set()
    for node in _walk(game.new_initial_state()):
        if node.info_key in seen:
            continue
        seen.add(node.info_key)
        yield node
