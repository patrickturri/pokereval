import random

import pytest

from pokereval.interface.types import GameVariant
from pokereval.engine.kuhn_leduc import load_game
from pokereval.rl.selfplay.game import sample_deal, play_from_deal


def _passive_choose(os_state):
    """Return call/check/pass action to keep the hand alive; else first legal."""
    player = os_state.current_player()
    for a in os_state.legal_actions():
        if os_state.action_to_string(player, a).strip().lower() in {"call", "check", "pass"}:
            return a
    return os_state.legal_actions()[0]


@pytest.mark.openspiel
def test_sample_deal_records_three_chance_outcomes():
    game = load_game(GameVariant.LEDUC)
    deal = sample_deal(game, random.Random(0))
    # Leduc: two private cards + one public card
    assert len(deal.chance_actions) == 3


@pytest.mark.openspiel
def test_replay_fixes_private_cards_across_replays():
    """play_from_deal must replay the identical private cards on every replay.

    Same Deal replayed twice must present player 0 with the same information
    state string at their first decision node.  If chance outcomes were
    re-sampled, the cards could differ, breaking the guarantee.
    """
    game = load_game(GameVariant.LEDUC)
    deal = sample_deal(game, random.Random(1))
    seen: list[str] = []

    def capture(os_state):
        if os_state.current_player() == 0 and not seen:
            seen.append(os_state.information_state_string(0))
        return _passive_choose(os_state)

    play_from_deal(game, deal, capture)
    first = seen[0]
    seen.clear()
    play_from_deal(game, deal, capture)
    assert seen[0] == first  # identical private card for player 0 both replays


@pytest.mark.openspiel
def test_replay_terminal_is_zero_sum():
    """Returns from a completed hand must sum to zero (Leduc is zero-sum)."""
    game = load_game(GameVariant.LEDUC)
    deal = sample_deal(game, random.Random(7))

    terminal = play_from_deal(game, deal, _passive_choose)
    returns = terminal.returns()
    assert abs(sum(returns)) < 1e-9
