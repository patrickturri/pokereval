import pytest
from pokereval.interface.types import State, GameVariant, Action
from pokereval.interface.prompts import render_state, parse_decision, ParseError

def _state():
    return State(
        variant=GameVariant.LEDUC, hero_cards=["Js"], pot=4.0, to_call=2.0,
        legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
    )

def test_render_mentions_cards_pot_and_legal_actions():
    text = render_state(_state())
    assert "Js" in text
    assert "4.0" in text or "4" in text
    assert "fold" in text and "call" in text and "raise" in text
    assert "ACTION:" in text

def test_parse_picks_last_action_line_and_reasoning():
    text = "I think about it.\nACTION: raise 4"
    d = parse_decision(text, [Action.FOLD, Action.CALL, Action.RAISE])
    assert d.action is Action.RAISE
    assert d.size == 4.0
    assert d.reasoning == "I think about it."

def test_parse_uses_final_action_when_multiple():
    text = "ACTION: call\n...changed my mind...\nACTION: fold"
    d = parse_decision(text, [Action.FOLD, Action.CALL])
    assert d.action is Action.FOLD

def test_parse_rejects_illegal_action():
    with pytest.raises(ParseError):
        parse_decision("ACTION: raise", [Action.FOLD, Action.CALL])

def test_parse_rejects_missing_action():
    with pytest.raises(ParseError):
        parse_decision("no action here", [Action.FOLD])
