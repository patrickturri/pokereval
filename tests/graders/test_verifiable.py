from pokereval.graders.verifiable import VerifiableGrader
from pokereval.interface.types import Action, Decision, GameVariant, State


def _leduc_state():
    return State(
        variant=GameVariant.LEDUC,
        hero_cards=["Js"],
        pot=4.0,
        to_call=2.0,
        legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
        info_state_key="K1",
        meta={"os_legal": {"fold": 0, "call": 1, "raise": 2}},
    )


def test_nash_grade_returns_probability_mass_and_exact_match():
    nash = {"K1": {"0": 0.1, "1": 0.2, "2": 0.7}}
    g = VerifiableGrader(nash)
    sc = g.grade(_leduc_state(), Decision(action=Action.RAISE))
    assert sc.value == 0.7
    assert sc.detail["exact_match"] is True
    sc2 = g.grade(_leduc_state(), Decision(action=Action.FOLD))
    assert sc2.value == 0.1
    assert sc2.detail["exact_match"] is False


def test_reference_grade_for_holdem():
    g = VerifiableGrader({})
    st = State(
        variant=GameVariant.HOLDEM,
        hero_cards=["As", "Ks"],
        board=["Ah", "7d", "2c"],
        pot=10.0,
        legal_actions=[Action.CHECK, Action.BET],
        meta={"reference_action": "bet"},
    )
    assert g.grade(st, Decision(action=Action.BET)).value == 1.0
    assert g.grade(st, Decision(action=Action.CHECK)).value == 0.0


def test_ungradeable_when_no_data():
    g = VerifiableGrader({})
    st = State(
        variant=GameVariant.HOLDEM,
        hero_cards=["As"],
        pot=1.0,
        legal_actions=[Action.CHECK],
        meta={},
    )
    assert g.grade(st, Decision(action=Action.CHECK)).detail["kind"] == "ungradeable"
