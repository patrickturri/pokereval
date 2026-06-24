"""Leduc hand stepper: deal sampling and fixed-deal replay."""

import random
from dataclasses import dataclass
from typing import Callable


@dataclass
class Deal:
    """Recorded chance outcomes for a single Leduc hand.

    ``chance_actions`` holds action ids in the order they were drawn from
    OpenSpiel chance nodes: [p0_private, p1_private, public_board].
    """

    chance_actions: list[int]


def _passive_action(state) -> int:
    """Return the call/check/pass action id for *state* (a player node).

    Keeps the hand alive so the passive walk reaches the public-card chance
    node that only appears after the first betting round.
    """
    player = state.current_player()
    for a in state.legal_actions():
        label = state.action_to_string(player, a).strip().lower()
        if label in {"call", "check", "pass"}:
            return a
    # Fallback: take the first legal action (should never be needed).
    return state.legal_actions()[0]


def sample_deal(game, rng: random.Random) -> Deal:
    """Sample a Leduc deal by walking a passive (never-fold) line to terminal.

    The passive walk ensures we pass through all three chance nodes — two
    private cards and the public board card — regardless of which hands are
    dealt.  Any later replay of the same deal can use any betting line.
    """
    st = game.new_initial_state()
    recorded: list[int] = []
    while not st.is_terminal():
        if st.is_chance_node():
            outcomes = [a for a, _prob in st.chance_outcomes()]
            a = rng.choice(outcomes)
            recorded.append(a)
            st = st.child(a)
        else:
            st = st.child(_passive_action(st))
    return Deal(chance_actions=recorded)


def play_from_deal(game, deal: Deal, choose: Callable[[object], int]):
    """Replay *deal*, forcing recorded chance outcomes and delegating player
    nodes to *choose(os_state) -> action_id*.  Returns the terminal state."""
    st = game.new_initial_state()
    ci = 0
    while not st.is_terminal():
        if st.is_chance_node():
            st = st.child(deal.chance_actions[ci])
            ci += 1
        else:
            st = st.child(choose(st))
    return st
