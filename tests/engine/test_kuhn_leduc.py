import pytest
pyspiel = pytest.importorskip("pyspiel")
pytestmark = pytest.mark.openspiel

from pokereval.interface.types import GameVariant, Action
from pokereval.engine.kuhn_leduc import load_game, iter_nodes, normalize_os_action

def test_load_holdem_raises():
    with pytest.raises(ValueError):
        load_game(GameVariant.HOLDEM)

def test_kuhn_nodes_are_nonempty_with_unique_keys():
    nodes = list(iter_nodes(GameVariant.KUHN))
    keys = [n.info_key for n in nodes]
    assert len(keys) > 0
    assert len(set(keys)) == len(keys)              # info keys are unique
    for n in nodes:
        assert n.os_legal                            # every node has >=1 legal action
        assert all(v in {a.value for a in Action} for v in n.os_legal)

def test_leduc_has_more_decision_nodes_than_kuhn():
    assert len(list(iter_nodes(GameVariant.LEDUC))) > len(list(iter_nodes(GameVariant.KUHN)))

def test_normalize_pass_depends_on_facing_bet():
    assert normalize_os_action("Pass", facing_bet=False) is Action.CHECK
    assert normalize_os_action("Pass", facing_bet=True) is Action.FOLD


def test_kuhn_facing_bet_determines_os_legal_keys():
    """Regression: Kuhn nodes facing a bet must expose fold/call, not check/bet.

    In Kuhn poker OpenSpiel always surfaces 'Pass' and 'Bet' regardless of
    context, so facing_bet must be derived from the betting history (not from
    label text).  This test pins the expected action-key sets per facing_bet
    value for every Kuhn decision node.
    """
    nodes = list(iter_nodes(GameVariant.KUHN))
    assert nodes, "iter_nodes(KUHN) returned no nodes"

    for node in nodes:
        keys = set(node.os_legal.keys())
        if node.facing_bet:
            assert keys == {"fold", "call"}, (
                f"Node {node.info_key!r} has facing_bet=True but os_legal keys={keys!r}; "
                "expected {'fold', 'call'}"
            )
        else:
            assert keys == {"check", "bet"}, (
                f"Node {node.info_key!r} has facing_bet=False but os_legal keys={keys!r}; "
                "expected {'check', 'bet'}"
            )
