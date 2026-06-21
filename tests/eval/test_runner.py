from pokereval.interface.types import State, Action, GameVariant
from pokereval.models.client import FakeClient
from pokereval.graders.verifiable import VerifiableGrader
from pokereval.eval.spots import SpotSet
from pokereval.eval.runner import run_eval, SpotResult

def _spotset():
    s = State(variant=GameVariant.LEDUC, hero_cards=["K"], pot=4.0, to_call=2.0,
              legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
              info_state_key="K1", meta={"os_legal": {"fold": 0, "call": 1, "raise": 2}})
    return SpotSet(spots=[s], nash_probs={"K1": {"0": 0.1, "1": 0.2, "2": 0.7}})

def test_run_eval_grades_a_valid_decision():
    ss = _spotset()
    results = run_eval(FakeClient("m", "ACTION: raise"), ss, VerifiableGrader(ss.nash_probs))
    assert len(results) == 1
    assert results[0].action == "raise"
    assert results[0].verifiable.value == 0.7
    assert results[0].error is None

def test_run_eval_records_parse_errors():
    ss = _spotset()
    results = run_eval(FakeClient("m", "no action"), ss, VerifiableGrader(ss.nash_probs))
    assert results[0].error is not None
    assert results[0].verifiable is None
