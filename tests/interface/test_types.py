from pokereval.interface.types import Action, Decision, GameVariant, Score, State


def test_state_roundtrips_through_json():
    s = State(
        variant=GameVariant.LEDUC,
        hero_cards=["Js"],
        pot=4.0,
        to_call=2.0,
        legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
        info_state_key="J:cr",
        meta={"os_legal": {"fold": 0, "call": 1, "raise": 2}},
    )
    again = State.model_validate(s.model_dump())
    assert again == s
    assert again.legal_actions[0] is Action.FOLD


def test_decision_defaults():
    d = Decision(action=Action.BET, size=2.0)
    assert d.reasoning == ""
    assert d.action is Action.BET


def test_score_holds_detail():
    sc = Score(value=0.75, detail={"nash_prob": 0.75})
    assert sc.value == 0.75
    assert sc.detail["nash_prob"] == 0.75
