"""Metric-divergence panel: action-match vs. exploitability.

The pure-math tests (sharpened / argmax / action_match_rate) carry no OpenSpiel
dependency and always run. The panel/CLI tests need pyspiel and are gated, like
tests/solver/test_nash.py.
"""
import pytest


# --- pure math (no OpenSpiel) -------------------------------------------------

from pokereval.eval.metric_divergence import (
    sharpened,
    argmax_by_key,
    action_match_rate,
)


def test_sharpened_tau_one_is_identity():
    nash = {"n": {0: 0.1, 1: 0.2, 2: 0.7}}
    out = sharpened(nash, 1.0)["n"]
    assert abs(out[0] - 0.1) < 1e-9
    assert abs(out[1] - 0.2) < 1e-9
    assert abs(out[2] - 0.7) < 1e-9


def test_sharpened_small_tau_concentrates_on_argmax():
    nash = {"n": {0: 0.1, 1: 0.2, 2: 0.7}}
    out = sharpened(nash, 0.05)["n"]
    assert out[2] > 0.99  # nearly all mass on the modal action
    assert out[0] < 1e-3 and out[1] < 1e-3


def test_sharpened_large_tau_flattens_toward_uniform():
    nash = {"n": {0: 0.1, 1: 0.2, 2: 0.7}}
    out = sharpened(nash, 50.0)["n"]
    # Closer to uniform-over-support (1/3 each) than the original was.
    spread_orig = max(nash["n"].values()) - min(nash["n"].values())
    spread_new = max(out.values()) - min(out.values())
    assert spread_new < spread_orig
    assert abs(sum(out.values()) - 1.0) < 1e-9


def test_sharpened_preserves_zero_support():
    nash = {"n": {0: 0.0, 1: 0.4, 2: 0.6}}
    out = sharpened(nash, 0.3)["n"]
    assert out.get(0, 0.0) == 0.0  # an action with zero Nash mass stays zero


def test_argmax_by_key_picks_modal_action():
    assert argmax_by_key({"a": {0: 0.1, 1: 0.2, 2: 0.7}}) == {"a": 2}


def test_action_match_rate_is_mass_on_nash_argmax():
    # action-match = expected fraction of sampled decisions equal to the Nash
    # modal action = mean policy mass on that action.
    nash_argmax = {"a": 2, "b": 0}
    policy = {"a": {0: 0.1, 1: 0.2, 2: 0.7}, "b": {0: 1.0}}
    assert abs(action_match_rate(policy, nash_argmax) - (0.7 + 1.0) / 2) < 1e-9


def test_action_match_rate_point_mass_extremes():
    nash_argmax = {"a": 2}
    assert action_match_rate({"a": {2: 1.0}}, nash_argmax) == 1.0
    assert action_match_rate({"a": {0: 1.0}}, nash_argmax) == 0.0


# --- panel (OpenSpiel-gated) --------------------------------------------------

pyspiel = pytest.importorskip("pyspiel")
pytestmark = pytest.mark.openspiel

from pokereval.eval.metric_divergence import evaluate_panel, render_panel_markdown
from pokereval.cli import main


def _rows_by_name(rows):
    return {r.name: r for r in rows}


def test_panel_greedy_beats_mixed_on_action_match_but_is_more_exploitable():
    rows = evaluate_panel("kuhn", iterations=2000)
    by = _rows_by_name(rows)
    mixed, greedy = by["nash_mixed"], by["nash_greedy"]

    # nash_mixed is near-Nash → ~0 exploitability, and (because Kuhn mixes) it
    # does NOT put full mass on the modal action.
    assert mixed.mixed_exploitability < 0.1
    assert mixed.action_match < 0.999

    # Collapsing onto the modal action gives a *perfect* action-match score …
    assert greedy.action_match == pytest.approx(1.0, abs=1e-6)
    # … while being strictly more exploitable. The better-scoring policy is worse.
    assert greedy.mixed_exploitability > mixed.mixed_exploitability
    assert greedy.action_match > mixed.action_match


def test_panel_tau_sweep_metrics_are_anti_aligned():
    rows = evaluate_panel("kuhn", iterations=2000, taus=(1.0, 0.1))
    sweep = {r.tau: r for r in rows if r.tau is not None}
    hi, lo = sweep[1.0], sweep[0.1]
    # Sharpening (tau 1.0 -> 0.1) raises action-match …
    assert lo.action_match >= hi.action_match - 1e-9
    # … and raises exploitability too: the two move together, the wrong way.
    assert lo.mixed_exploitability >= hi.mixed_exploitability - 1e-6


def test_render_panel_markdown_has_header():
    rows = evaluate_panel("kuhn", iterations=500)
    md = render_panel_markdown(rows)
    assert "| Policy |" in md
    assert "Action-Match" in md


def test_cli_metric_divergence_runs(capsys):
    code = main(["metric-divergence", "--variant", "kuhn", "--iterations", "500"])
    assert code == 0
    out = capsys.readouterr().out
    assert "| Policy |" in out


def test_cli_metric_divergence_json(capsys):
    import json

    code = main(["metric-divergence", "--variant", "kuhn",
                 "--iterations", "500", "--format", "json"])
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert isinstance(data, list)
    assert {"name", "action_match", "mixed_exploitability"} <= set(data[0].keys())
