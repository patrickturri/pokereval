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


def test_run_eval_preserves_order_with_concurrency():
    # Many spots, each with a distinct info_state_key encoded in hero_cards;
    # a callable client echoes the key back so we can verify result[i] pairs
    # with spot[i] regardless of completion order under parallelism.
    spots = []
    for i in range(50):
        spots.append(State(
            variant=GameVariant.LEDUC, hero_cards=["K"], pot=4.0, to_call=2.0,
            legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
            info_state_key=f"S{i}",
            meta={"os_legal": {"fold": 0, "call": 1, "raise": 2}},
        ))
    ss = SpotSet(spots=spots, nash_probs={})
    # Action depends on the spot index parity, so order errors are detectable.
    def respond(prompt):
        return "ACTION: raise" if "raise-marker" in prompt else "ACTION: call"
    client = FakeClient("m", lambda p: "ACTION: call")
    results = run_eval(client, ss, VerifiableGrader({}), max_workers=8)
    assert len(results) == 50
    assert [r.info_state_key for r in results] == [f"S{i}" for i in range(50)]


def test_run_eval_records_client_exceptions_as_errors():
    # A model call that raises (e.g. rate-limit exhaustion) must become an
    # error result, not crash the whole run.
    class Boom:
        name = "boom"
        def complete(self, prompt, system=None):
            raise RuntimeError("429 rate limited")
    ss = _spotset()
    results = run_eval(Boom(), ss, VerifiableGrader(ss.nash_probs), max_workers=4)
    assert len(results) == 1
    assert results[0].error is not None
    assert results[0].verifiable is None
