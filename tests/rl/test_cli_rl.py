import pytest
from pokereval.cli import main

pytest.importorskip("open_spiel")  # rl-train builds CFR-labeled spots


@pytest.mark.openspiel
def test_rl_train_smoke_offline(capsys):
    rc = main(["rl-train", "--variant", "leduc", "--preset", "smoke"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "step_rewards" in out or "mean_reward" in out
