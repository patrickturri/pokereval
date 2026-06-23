import pytest
from pokereval.interface.types import GameVariant
from pokereval.rl.sampler import MockSamplingClient
from pokereval.rl.evaluate import evaluate_policy, greedy_choices

openspiel = pytest.importorskip("open_spiel")  # exploitability needs OpenSpiel


@pytest.mark.openspiel
def test_evaluate_runs_on_leduc():
    from pokereval.synth.generator import build_labeled_spots
    spots = build_labeled_spots(GameVariant.LEDUC, iterations=200)
    # Policy that always picks the first legal action listed in the prompt
    # (Leduc actions are check/raise/fold/call — there is no "bet").
    def action_for(prompt: str) -> str:
        line = next(l for l in prompt.splitlines() if l.startswith("Legal actions:"))
        return line.split(":", 1)[1].split(",")[0].strip()
    sc = MockSamplingClient(action_for=action_for)
    choices = greedy_choices(GameVariant.LEDUC, spots, sc)
    assert len(choices) > 0
    rep = evaluate_policy(GameVariant.LEDUC, spots, spots[:20], sc)
    assert rep.exploitability >= 0.0
    assert 0.0 <= rep.mean_nash_prob <= 1.0
    assert 0.0 <= rep.exact_match <= 1.0
    assert rep.n == 20
