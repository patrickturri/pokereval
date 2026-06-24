from pokereval.rl.config import get_preset


def test_selfplay_preset_fields():
    smoke = get_preset("smoke")
    assert smoke.deals_per_step == 2
    assert smoke.hands_per_deal == 2
    assert smoke.opponent_refresh_every == 1
    real = get_preset("real")
    assert real.deals_per_step == 8
    assert real.hands_per_deal == 8
    assert real.opponent_refresh_every == 10
