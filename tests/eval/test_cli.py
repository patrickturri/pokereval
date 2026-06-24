from pokereval.cli import main, run_for_client
from pokereval.eval.spots import SpotSet
from pokereval.graders.verifiable import VerifiableGrader
from pokereval.interface.types import Action, GameVariant, State
from pokereval.models.client import FakeClient


def test_run_for_client_returns_summary():
    s = State(
        variant=GameVariant.LEDUC,
        hero_cards=["K"],
        pot=4.0,
        to_call=2.0,
        legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
        info_state_key="K1",
        meta={"os_legal": {"fold": 0, "call": 1, "raise": 2}},
    )
    ss = SpotSet(spots=[s], nash_probs={"K1": {"0": 0.1, "1": 0.2, "2": 0.7}})
    summary = run_for_client(
        FakeClient("m", "ACTION: raise"), ss, VerifiableGrader(ss.nash_probs), None
    )
    assert summary.model == "m"
    assert summary.mean_verifiable == 0.7


def test_main_runs_holdem_endtoend(capsys):
    code = main(
        ["run", "--variant", "holdem", "--holdem-path", "data/holdem_spots.jsonl"]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "| Model |" in out
