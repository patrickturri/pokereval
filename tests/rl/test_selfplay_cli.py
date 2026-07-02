# tests/rl/test_selfplay_cli.py
import pytest
from pokereval.cli import main


@pytest.mark.openspiel
def test_rl_selfplay_offline_smoke_returns_zero(capsys):
    rc = main(["rl-selfplay", "--variant", "leduc", "--preset", "smoke",
               "--iterations", "50"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "step_returns" in out


@pytest.mark.openspiel
def test_rl_selfplay_tabular_offline_returns_zero(capsys):
    rc = main(["rl-selfplay", "--variant", "leduc", "--preset", "smoke",
               "--tabular", "--iterations", "50"])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"mode": "tabular"' in out
    assert "step_returns" in out
