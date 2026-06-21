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
