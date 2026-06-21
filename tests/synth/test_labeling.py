"""Pure-Python failure-mode classification — no OpenSpiel needed."""
from pokereval.synth.labeling import mixedness, classify_distribution, MIXED_THRESHOLD


def test_mixedness_pure_strategy_is_zero():
    assert mixedness({"fold": 1.0, "call": 0.0}) == 0.0


def test_mixedness_uniform_two_way_is_half():
    assert mixedness({"check": 0.5, "bet": 0.5}) == 0.5


def test_mixedness_empty_is_zero():
    assert mixedness({}) == 0.0


def test_classify_pure_strategy_is_near_deterministic():
    tags = classify_distribution({"fold": 1.0, "call": 0.0}, facing_bet=False)
    assert "near_deterministic" in tags
    assert "mixed" not in tags


def test_classify_mixed_strategy_tagged_mixed():
    # A genuinely mixed equilibrium decision (the Phase-1 hypothesized weakness).
    tags = classify_distribution({"check": 0.7, "bet": 0.3}, facing_bet=False)
    assert "mixed" in tags


def test_classify_threshold_boundary():
    # Clearly under the threshold -> not mixed; clearly over -> mixed.
    below = {"a": 1 - (MIXED_THRESHOLD / 2), "b": MIXED_THRESHOLD / 2}
    assert "mixed" not in classify_distribution(below, facing_bet=False)
    above = {"a": 1 - (MIXED_THRESHOLD * 2), "b": MIXED_THRESHOLD * 2}
    assert "mixed" in classify_distribution(above, facing_bet=False)


def test_classify_facing_bet_tag():
    tags = classify_distribution({"fold": 0.5, "call": 0.5}, facing_bet=True)
    assert "facing_bet" in tags
    assert "facing_bet" not in classify_distribution({"check": 1.0}, facing_bet=False)
