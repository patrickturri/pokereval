import json

import pytest

pyspiel = pytest.importorskip("pyspiel")
pytestmark = pytest.mark.openspiel

from pokereval.interface.types import GameVariant
from pokereval.synth.generator import (
    build_labeled_spots,
    select_by_tag,
    to_jsonl_records,
)


def test_build_labeled_spots_have_normalized_nash_labels():
    spots = build_labeled_spots(GameVariant.KUHN, iterations=500)
    assert spots
    for s in spots:
        assert s.state.info_state_key is not None
        # Nash action probabilities are keyed by Action value and sum to ~1.
        assert abs(sum(s.nash_action_probs.values()) - 1.0) < 1e-6
        # Keys are legal action values for this state.
        legal = {a.value for a in s.state.legal_actions}
        assert set(s.nash_action_probs) <= legal
        assert s.tags  # every spot is classified


def test_kuhn_contains_mixed_strategy_spots():
    # Kuhn equilibrium genuinely mixes (bluff/call frequencies), so the
    # targeted-failure-mode generator must surface some 'mixed' spots.
    spots = build_labeled_spots(GameVariant.KUHN, iterations=2000)
    mixed = select_by_tag(spots, "mixed")
    assert len(mixed) > 0


def test_to_jsonl_records_roundtrip():
    spots = build_labeled_spots(GameVariant.KUHN, iterations=300)
    records = to_jsonl_records(spots)
    assert len(records) == len(spots)
    # Each record serializes to a single JSON line and carries the label + tags.
    line = json.dumps(records[0])
    back = json.loads(line)
    assert "nash_action_probs" in back
    assert "tags" in back
    assert "state" in back
