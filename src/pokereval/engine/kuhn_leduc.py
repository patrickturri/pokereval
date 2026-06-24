from dataclasses import dataclass
import pyspiel
from ..interface.types import Action, GameVariant, State

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
    os_legal: dict[str, int]  # action value (str) -> OpenSpiel action id (int)
    facing_bet: bool
    os_state: "pyspiel.State"


def iter_nodes(variant: GameVariant):
    """Enumerate every unique player decision node in the full game tree.

    Chance nodes are expanded automatically. Terminal nodes are skipped.
    Yields one NodeInfo per unique information state (deduplicated by info_key).
    """
    game = load_game(variant)

    def _walk(state, pending_bet: bool = False):
        if state.is_terminal():
            return
        if state.is_chance_node():
            # A new betting round starts after a chance node; no pending bet.
            for action, _prob in state.chance_outcomes():
                yield from _walk(state.child(action), pending_bet=False)
            return
        player = state.current_player()
        info_key = state.information_state_string(player)
        labels: dict[str, int] = {
            a: state.action_to_string(player, a) for a in state.legal_actions()
        }
        # facing_bet is threaded through the recursion so it is correct for all
        # variants.  In Kuhn, OpenSpiel only exposes 'Pass'/'Bet' at every node,
        # so label-based detection never fires; the recursive parameter is the
        # only reliable source.  In Leduc, 'Fold' appears iff there is a pending
        # bet, which agrees with the recursive parameter.
        facing_bet = pending_bet
        os_legal: dict[str, int] = {}
        for a, label in labels.items():
            act = normalize_os_action(label, facing_bet)
            os_legal[act.value] = a
        yield NodeInfo(info_key, player, os_legal, facing_bet, state)
        for a, label in labels.items():
            raw = label.strip().lower()
            # A bet-opening action ('bet' or 'raise') creates a pending bet for
            # the next player; all other actions (pass/check/call/fold) either
            # answer or ignore the bet.
            child_pending = raw in ("bet", "raise")
            yield from _walk(state.child(a), pending_bet=child_pending)

    seen: set = set()
    for node in _walk(game.new_initial_state()):
        if node.info_key in seen:
            continue
        seen.add(node.info_key)
        yield node


def build_state(
    variant: GameVariant,
    info_key: str,
    os_legal: dict[str, int],
) -> State:
    """Build a rendered ``State`` from a pre-computed os_legal map.

    Args:
        variant: The game variant (KUHN or LEDUC).
        info_key: The OpenSpiel information-state string for the acting player.
        os_legal: Mapping of ``Action.value`` strings to OpenSpiel action ids.

    Returns:
        A fully-populated ``State`` with ``legal_actions`` derived from the keys
        of ``os_legal``, ``info_state_key`` and ``hero_cards`` set to ``info_key``,
        and the original ``os_legal`` map stored in ``meta``.
    """
    legal = [Action(v) for v in os_legal]
    return State(
        variant=variant,
        hero_cards=[info_key],
        pot=0.0,
        history=info_key,
        legal_actions=legal,
        info_state_key=info_key,
        meta={"os_legal": os_legal},
    )


def node_to_state(
    variant: GameVariant,
    os_state,
) -> tuple[State, dict[str, int]]:
    """Convert a live OpenSpiel decision node to a rendered ``State``.

    Args:
        variant: The game variant (KUHN or LEDUC).
        os_state: A live OpenSpiel state that is not a chance or terminal node.

    Returns:
        A ``(State, os_legal)`` pair where ``os_legal`` maps ``Action.value``
        strings to OpenSpiel action ids, mirroring the layout used by
        ``iter_nodes`` / ``NodeInfo``.
    """
    player = os_state.current_player()
    info_key = os_state.information_state_string(player)
    labels: dict[int, str] = {
        a: os_state.action_to_string(player, a) for a in os_state.legal_actions()
    }
    # In Leduc, 'Fold' only appears when there is a pending bet.
    facing_bet = any(lbl.strip().lower() == "fold" for lbl in labels.values())
    os_legal: dict[str, int] = {}
    for a, lbl in labels.items():
        os_legal[normalize_os_action(lbl, facing_bet).value] = a
    return build_state(variant, info_key, os_legal), os_legal
