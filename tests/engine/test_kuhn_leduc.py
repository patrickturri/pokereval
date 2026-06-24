import pytest

pyspiel = pytest.importorskip("pyspiel")
pytestmark = pytest.mark.openspiel

from pokereval.engine.kuhn_leduc import (  # noqa: E402
    iter_nodes,
    load_game,
    normalize_os_action,
)
from pokereval.interface.types import Action, GameVariant  # noqa: E402


def test_load_holdem_raises():
    with pytest.raises(ValueError):
        load_game(GameVariant.HOLDEM)


def test_kuhn_nodes_are_nonempty_with_unique_keys():
    nodes = list(iter_nodes(GameVariant.KUHN))
    keys = [n.info_key for n in nodes]
    assert len(keys) > 0
    assert len(set(keys)) == len(keys)
    for n in nodes:
        assert n.os_legal
        assert all(v in {a.value for a in Action} for v in n.os_legal)


def test_leduc_has_more_decision_nodes_than_kuhn():
    assert len(list(iter_nodes(GameVariant.LEDUC))) > len(
        list(iter_nodes(GameVariant.KUHN))
    )


def test_normalize_pass_depends_on_facing_bet():
    assert normalize_os_action("Pass", facing_bet=False) is Action.CHECK
    assert normalize_os_action("Pass", facing_bet=True) is Action.FOLD
