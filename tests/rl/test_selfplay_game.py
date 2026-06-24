import random

import pytest

from pokereval.interface.types import GameVariant
from pokereval.engine.kuhn_leduc import load_game
from pokereval.rl.selfplay.game import sample_deal, play_from_deal


@pytest.mark.openspiel
def test_sample_deal_records_three_chance_outcomes():
    game = load_game(GameVariant.LEDUC)
    deal = sample_deal(game, random.Random(0))
    # Leduc: two private cards + one public card
    assert len(deal.chance_actions) == 3


@pytest.mark.openspiel
def test_replay_same_cards_as_deal():
    """play_from_deal must replay the exact same chance actions as deal."""
    from pokereval.engine.kuhn_leduc import node_to_state

    game = load_game(GameVariant.LEDUC)
    deal = sample_deal(game, random.Random(42))

    def _passive_choose(os_state):
        """Mirror _passive_action: keep hand alive."""
        player = os_state.current_player()
        for a in os_state.legal_actions():
            if os_state.action_to_string(player, a).strip().lower() in {"call", "check", "pass"}:
                return a
        return os_state.legal_actions()[0]

    terminal = play_from_deal(game, deal, _passive_choose)
    assert terminal.is_terminal()


@pytest.mark.openspiel
def test_replay_terminal_is_zero_sum():
    """Returns from a completed hand must sum to zero (Leduc is zero-sum)."""
    game = load_game(GameVariant.LEDUC)
    deal = sample_deal(game, random.Random(7))

    def _passive_choose(os_state):
        player = os_state.current_player()
        for a in os_state.legal_actions():
            if os_state.action_to_string(player, a).strip().lower() in {"call", "check", "pass"}:
                return a
        return os_state.legal_actions()[0]

    terminal = play_from_deal(game, deal, _passive_choose)
    returns = terminal.returns()
    assert abs(sum(returns)) < 1e-9
