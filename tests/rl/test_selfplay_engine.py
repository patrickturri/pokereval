"""Tests for shared state-construction helpers in engine/kuhn_leduc.py."""

import pytest
from pokereval.interface.types import Action, GameVariant
from pokereval.engine.kuhn_leduc import build_state


def test_build_state_from_os_legal():
    os_legal = {"check": 1, "raise": 2}
    state = build_state(GameVariant.LEDUC, "Kp", os_legal)
    assert state.variant == GameVariant.LEDUC
    assert state.info_state_key == "Kp"
    assert state.hero_cards == ["Kp"]
    assert state.meta["os_legal"] == os_legal
    assert set(state.legal_actions) == {Action.CHECK, Action.RAISE}


@pytest.mark.openspiel
def test_node_to_state_on_live_leduc_root():
    from pokereval.engine.kuhn_leduc import load_game, node_to_state

    game = load_game(GameVariant.LEDUC)
    st = game.new_initial_state()
    # apply private-card chance deals to reach player 0's first decision
    while st.is_chance_node():
        st = st.child(st.chance_outcomes()[0][0])
    state, os_legal = node_to_state(GameVariant.LEDUC, st)
    # no bet open: check + raise are legal, fold is not
    assert Action.CHECK in state.legal_actions
    assert Action.FOLD not in state.legal_actions
    # os_legal maps rendered action value back to real OpenSpiel action id
    assert set(os_legal.values()) == set(st.legal_actions())
