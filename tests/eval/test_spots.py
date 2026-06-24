import json
from pathlib import Path
import pytest
from pokereval.interface.types import GameVariant, Action
from pokereval.eval.spots import SpotSet, load_holdem_spotset

def test_load_holdem_spotset(tmp_path):
    rows = [{
        "variant": "holdem", "hero_cards": ["As", "Ks"], "board": ["Ah", "7d", "2c"],
        "pot": 10.0, "to_call": 0.0,
        "legal_actions": ["check", "bet"], "meta": {"reference_action": "bet"},
    }]
    p = tmp_path / "h.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    ss = load_holdem_spotset(p)
    assert len(ss.spots) == 1
    assert ss.spots[0].legal_actions == [Action.CHECK, Action.BET]
    assert ss.spots[0].meta["reference_action"] == "bet"

def test_shipped_holdem_file_is_valid():
    ss = load_holdem_spotset(Path("data/holdem_spots.jsonl"))
    assert len(ss.spots) >= 5
    for s in ss.spots:
        assert s.variant is GameVariant.HOLDEM
        assert "reference_action" in s.meta

@pytest.mark.openspiel
def test_build_kuhn_spotset_has_keys_and_nash():
    pytest.importorskip("pyspiel")
    ss = build = __import__(
        "pokereval.eval.spots", fromlist=["build_kuhn_leduc_spotset"]
    ).build_kuhn_leduc_spotset(GameVariant.KUHN, iterations=300)
    assert len(ss.spots) > 0
    for s in ss.spots:
        assert s.info_state_key in ss.nash_probs
        assert "os_legal" in s.meta
