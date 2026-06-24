from pokereval.interface.types import State, Decision, Action, GameVariant
from pokereval.models.client import FakeClient
from pokereval.graders.judge import JudgeGrader, RUBRIC


def _state():
    return State(variant=GameVariant.HOLDEM, hero_cards=["As", "Ks"],
                 board=["Ah", "7d", "2c"], pot=10.0,
                 legal_actions=[Action.CHECK, Action.BET])


def test_judge_averages_rubric_scores():
    payload = "{" + ", ".join(f'"{c}": 1' for c in RUBRIC) + "}"
    g = JudgeGrader(FakeClient("judge", payload))
    sc = g.grade(_state(), Decision(action=Action.BET, reasoning="Top pair, bet for value."))
    assert sc.value == 1.0
    assert sc.detail["kind"] == "judge"


def test_judge_partial_credit():
    scores = {c: (1 if i == 0 else 0) for i, c in enumerate(RUBRIC)}
    import json
    g = JudgeGrader(FakeClient("judge", json.dumps(scores)))
    sc = g.grade(_state(), Decision(action=Action.BET))
    assert 0.0 < sc.value < 1.0


def test_judge_handles_unparseable_output():
    g = JudgeGrader(FakeClient("judge", "I cannot produce JSON"))
    sc = g.grade(_state(), Decision(action=Action.BET))
    assert sc.detail["kind"] == "judge_error"
    assert sc.value == 0.0
