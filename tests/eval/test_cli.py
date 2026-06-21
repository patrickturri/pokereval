import subprocess
import sys

import pytest

from pokereval.interface.types import GameVariant, Action, State
from pokereval.models.client import FakeClient
from pokereval.graders.verifiable import VerifiableGrader
from pokereval.eval.spots import SpotSet
from pokereval.cli import run_for_client, main, build_client


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


def test_build_client_fake_is_credential_free():
    c = build_client("fake", model="ignored")
    assert isinstance(c, FakeClient)
    # Deterministic, parseable default response — no network, no credentials.
    assert "ACTION:" in c.complete("any prompt")


def test_build_client_fake_custom_name():
    c = build_client("fake", model="x", name="baseline")
    assert c.name == "baseline"


def test_build_client_unknown_provider_raises():
    with pytest.raises(ValueError):
        build_client("not-a-provider", model="x")


def test_build_client_anthropic_dispatches_without_calling_api(monkeypatch):
    # build_client must construct the right client type for "anthropic" without
    # us hitting the network. We patch the client class so no SDK/key is needed.
    import pokereval.cli as cli_mod

    captured = {}

    class _Stub:
        def __init__(self, name, model, **kwargs):
            captured["name"] = name
            captured["model"] = model
            self.name = name

    monkeypatch.setattr(cli_mod, "AnthropicClient", _Stub)
    c = build_client("anthropic", model="claude-opus-4-8")
    assert captured["model"] == "claude-opus-4-8"
    # Name defaults to the model id when not given.
    assert c.name == "claude-opus-4-8"


def test_main_accepts_provider_and_model_flags(capsys):
    code = main([
        "run", "--variant", "holdem",
        "--holdem-path", "data/holdem_spots.jsonl",
        "--provider", "fake",
    ])
    assert code == 0
    assert "| Model |" in capsys.readouterr().out
