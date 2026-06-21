import subprocess
import sys

import pytest

from pokereval.interface.types import GameVariant, Action, State
from pokereval.models.client import FakeClient
from pokereval.graders.verifiable import VerifiableGrader
from pokereval.eval.spots import SpotSet
from pokereval.cli import run_for_client, main


def test_run_for_client_returns_summary():
    s = State(variant=GameVariant.LEDUC, hero_cards=["K"], pot=4.0, to_call=2.0,
              legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
              info_state_key="K1", meta={"os_legal": {"fold": 0, "call": 1, "raise": 2}})
    ss = SpotSet(spots=[s], nash_probs={"K1": {"0": 0.1, "1": 0.2, "2": 0.7}})
    summary = run_for_client(FakeClient("m", "ACTION: raise"), ss,
                             VerifiableGrader(ss.nash_probs), None)
    assert summary.model == "m"
    assert summary.mean_verifiable == 0.7


def test_main_runs_holdem_endtoend(capsys):
    code = main(["run", "--variant", "holdem", "--holdem-path", "data/holdem_spots.jsonl"])
    assert code == 0
    out = capsys.readouterr().out
    assert "| Model |" in out


@pytest.mark.openspiel
def test_cli_stdout_has_no_openspiel_import_noise():
    """OpenSpiel's C++ game registry prints an 'Optional module ... not
    importable' line to stdout at import time. The leaderboard must be clean
    so the output is parseable and copy-pasteable. Run in a subprocess because
    the noisy import is cached after the first import within a process."""
    pytest.importorskip("pyspiel")
    proc = subprocess.run(
        [sys.executable, "-m", "pokereval.cli", "run", "--variant", "leduc",
         "--iterations", "50"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "not importable" not in proc.stdout
    assert "pokerkit_wrapper" not in proc.stdout
    assert "| Model |" in proc.stdout
