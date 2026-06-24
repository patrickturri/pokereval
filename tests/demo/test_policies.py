"""Tests for the demo's finding-analysis module.

The keystone test reproduces the Phase-2 headline *live*: the deterministic
policy the RLVR model collapsed onto (byte-identical to first-legal-action) has
Leduc greedy exploitability 1.75 — exactly the recorded `after` value. The demo
is therefore a live reproduction of the finding, not a mock.
"""
import pytest

from pokereval.interface.types import GameVariant


pytestmark = pytest.mark.openspiel


@pytest.fixture(scope="module")
def analysis():
    from pokereval.demo.policies import build_analysis

    # Small CFR budget keeps the fixture fast; the collapsed-policy
    # exploitability is exact regardless of solver convergence (it is a
    # deterministic policy), so the keystone assertion still holds.
    return build_analysis(GameVariant.LEDUC, iterations=200, max_spots=40)


def test_collapsed_reproduces_recorded_exploitability(analysis):
    # Recorded RLVR `after` greedy exploitability = 1.750 (results/rl-leduc-kl-lr1e4.json).
    assert analysis.collapsed_exploitability == pytest.approx(1.75, abs=0.02)


def test_nash_optimum_is_zero(analysis):
    # A Nash equilibrium is by definition unexploitable.
    assert analysis.nash_exploitability == 0.0


def test_enumerates_all_leduc_info_states(analysis):
    assert analysis.n_info_states == 936


def test_collapsed_action_is_first_legal(analysis):
    spot = analysis.spots[0]
    assert spot.collapsed_action == spot.legal_actions[0]


def test_display_spots_are_deduped_by_shape(analysis):
    # Each distinct (legal set, Nash-shape-to-5%) appears at most once.
    sigs = [
        (tuple(s.legal_actions), tuple(sorted((a, round(p * 20) / 20) for a, p in s.nash_probs.items())))
        for s in analysis.spots
    ]
    assert len(sigs) == len(set(sigs))


def test_parses_leduc_card_and_board():
    from pokereval.demo.policies import _parse_leduc

    parsed = _parse_leduc("[Observer: 1][Private: 2][Round 2][Pot: 22][Public: 4][Round1: 1][Round2: 1]")
    assert parsed["hero"] == "Q"  # card index 2 → rank Q
    assert parsed["board"] == "K"  # card index 4 → rank K
    assert parsed["pot"] == 22
    assert parsed["round_num"] == 2


def test_displayed_spots_are_mixed_nodes(analysis):
    # The teaching spots are exactly where Nash mixes — so a deterministic
    # policy is provably wrong there (collapsed action gets < 100% Nash mass).
    assert analysis.spots, "expected at least one mixed spot to display"
    for spot in analysis.spots:
        assert spot.nash_prob_of_collapsed < 1.0
        assert max(spot.nash_probs.values()) < 1.0


def test_nash_mass_missed_is_complement_of_collapsed_prob(analysis):
    for spot in analysis.spots:
        assert spot.nash_mass_missed == pytest.approx(1.0 - spot.nash_prob_of_collapsed, abs=1e-9)
        assert spot.nash_mass_missed > 0.0  # mixed node ⇒ collapse always discards some mass


def test_spots_sorted_by_mixedness_descending(analysis):
    mixedness = [spot.mixedness for spot in analysis.spots]
    assert mixedness == sorted(mixedness, reverse=True)


def test_roundtrips_through_dict(analysis):
    from pokereval.demo.policies import FindingAnalysis

    restored = FindingAnalysis.from_dict(analysis.to_dict())
    assert restored.collapsed_exploitability == analysis.collapsed_exploitability
    assert restored.variant == analysis.variant
    assert len(restored.spots) == len(analysis.spots)
    assert restored.spots[0] == analysis.spots[0]
