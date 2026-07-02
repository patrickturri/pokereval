"""Tests for exploitability leak attribution.

Pure-Python helpers (``repair`` / ``deviating_keys`` / ranking) are tested
without OpenSpiel. The end-to-end attribution invariants run on **Kuhn** (the
smallest game) so the best-response calls stay cheap, and assert structural
facts that hold at any iteration count rather than brittle magic numbers.
"""
from __future__ import annotations

import pytest

from pokereval.eval.leak_attribution import (
    LeakReport,
    NodeLeak,
    attribute_leaks,
    deviating_keys,
    rank_leaks,
    render_markdown,
    repair,
)

# --- pure-Python helpers (no OpenSpiel) --------------------------------------


def test_repair_replaces_only_named_keys_without_mutating_input():
    policy = {"a": {0: 1.0, 1: 0.0}, "b": {0: 0.3, 1: 0.7}}
    nash = {"a": {0: 0.5, 1: 0.5}, "b": {0: 0.9, 1: 0.1}}

    out = repair(policy, nash, ["a"])

    assert out["a"] == {0: 0.5, 1: 0.5}  # repaired to Nash
    assert out["b"] == {0: 0.3, 1: 0.7}  # untouched
    assert policy["a"] == {0: 1.0, 1: 0.0}  # input not mutated
    assert out is not policy


def test_repair_missing_nash_key_is_left_as_is():
    policy = {"a": {0: 1.0}}
    out = repair(policy, {}, ["a"])
    assert out["a"] == {0: 1.0}


def test_deviating_keys_finds_only_genuinely_different_nodes():
    policy = {
        "pure": {0: 1.0, 1: 0.0},      # identical to Nash → not deviating
        "collapsed": {0: 1.0, 1: 0.0},  # Nash mixes here → deviating
        "noise": {0: 0.5000000001, 1: 0.4999999999},  # within atol → not deviating
    }
    nash = {
        "pure": {0: 1.0, 1: 0.0},
        "collapsed": {0: 0.6, 1: 0.4},
        "noise": {0: 0.5, 1: 0.5},
    }
    assert deviating_keys(policy, nash) == ["collapsed"]


def test_deviating_keys_handles_string_action_ids_and_missing_actions():
    # Solver hands back stringified action ids; comparison must coerce.
    policy = {"k": {"0": 1.0}}              # missing action 1 ⇒ implicit 0.0
    nash = {"k": {0: 0.5, 1: 0.5}}
    assert deviating_keys(policy, nash) == ["k"]


def test_rank_leaks_sorts_descending_by_marginal_leak():
    leaks = [
        NodeLeak(info_key="small", marginal_leak=0.1, n_legal=2),
        NodeLeak(info_key="big", marginal_leak=0.9, n_legal=2),
        NodeLeak(info_key="mid", marginal_leak=0.5, n_legal=3),
    ]
    ranked = rank_leaks(leaks)
    assert [n.info_key for n in ranked] == ["big", "mid", "small"]


def test_render_markdown_smoke():
    report = LeakReport(
        variant="kuhn",
        baseline_exploitability=1.0,
        nash_exploitability=0.0,
        leaks=[NodeLeak(info_key="K[Private:2]", marginal_leak=0.4, n_legal=2)],
        cumulative=[(1, 0.6)],
    )
    md = render_markdown(report)
    assert "Exploitability" in md or "exploitability" in md
    assert "0.4" in md


# --- end-to-end attribution (OpenSpiel, on small Kuhn) -----------------------


@pytest.mark.openspiel
def test_nash_policy_leaks_nothing():
    """The Nash mix itself has no per-node leak: every marginal leak ≈ 0 and the
    baseline equals the Nash exploitability."""
    from pokereval.eval.metric_divergence import _normalize_int_keys
    from pokereval.solver.nash import nash_probs_by_key

    nash = _normalize_int_keys(nash_probs_by_key("kuhn", 300))
    report = attribute_leaks("kuhn", nash, nash=nash, iterations=300)

    assert report.baseline_exploitability == pytest.approx(
        report.nash_exploitability, abs=1e-6
    )
    assert report.leaks == []  # nothing deviates from Nash


@pytest.mark.openspiel
def test_collapsed_policy_attribution_and_full_repair_recovers_nash():
    """A greedy-collapsed policy is more exploitable than Nash, the leak is
    attributed to ≥1 node, and repairing *all* deviating nodes recovers exactly
    the Nash exploitability (full repair == the Nash policy)."""
    from pokereval.eval.metric_divergence import (
        _normalize_int_keys,
        argmax_by_key,
        _point_mass,
    )
    from pokereval.solver.nash import nash_probs_by_key

    nash = _normalize_int_keys(nash_probs_by_key("kuhn", 300))
    collapsed = _point_mass(argmax_by_key(nash))

    report = attribute_leaks("kuhn", collapsed, nash=nash, iterations=300)

    # The collapse is strictly more exploitable than the mix it came from.
    assert report.baseline_exploitability > report.nash_exploitability + 1e-3
    # At least one node carries a positive leak, and leaks are ranked.
    assert report.leaks
    assert report.leaks[0].marginal_leak > 0
    assert all(
        report.leaks[i].marginal_leak >= report.leaks[i + 1].marginal_leak
        for i in range(len(report.leaks) - 1)
    )
    # Repairing every deviating node reconstitutes the Nash policy exactly.
    assert report.cumulative[-1][1] == pytest.approx(
        report.nash_exploitability, abs=1e-6
    )


@pytest.mark.openspiel
def test_attribute_leaks_defaults_nash_from_solver():
    """When ``nash`` is omitted it is solved internally; a collapsed policy still
    attributes a positive leak."""
    from pokereval.eval.metric_divergence import (
        _normalize_int_keys,
        argmax_by_key,
        _point_mass,
    )
    from pokereval.solver.nash import nash_probs_by_key

    collapsed = _point_mass(argmax_by_key(_normalize_int_keys(nash_probs_by_key("kuhn", 300))))
    report = attribute_leaks("kuhn", collapsed, iterations=300)
    assert report.leaks
    assert report.baseline_exploitability > report.nash_exploitability


@pytest.mark.openspiel
def test_cli_leak_attribution_markdown(capsys):
    from pokereval.cli import main

    code = main(["leak-attribution", "--variant", "kuhn", "--iterations", "300"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Policy exploitability" in out
    assert "Marginal leak" in out


@pytest.mark.openspiel
def test_cli_leak_attribution_json(capsys):
    import json

    from pokereval.cli import main

    code = main(["leak-attribution", "--variant", "kuhn", "--iterations", "300",
                 "--format", "json"])
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert {"baseline_exploitability", "nash_exploitability", "leaks"} <= set(data.keys())


@pytest.mark.openspiel
def test_cli_leak_attribution_nash_policy_is_clean(capsys):
    """`--policy nash` is the sanity check: a Nash policy has no leaks."""
    from pokereval.cli import main

    code = main(["leak-attribution", "--variant", "kuhn", "--iterations", "300",
                 "--policy", "nash"])
    assert code == 0
    assert "0 mixed decision nodes" in capsys.readouterr().out
