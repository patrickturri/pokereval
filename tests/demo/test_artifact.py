"""Tests for the offline before/after artifact (scripts/demo.py's engine).

`render_before_after` works on plain dicts, so most tests need no OpenSpiel. The
full-build test that runs CFR is `openspiel`-marked.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from pokereval.demo.artifact import (
    _fmt_dist,
    render_before_after,
    build_offline_artifact,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

# A minimal, hand-built stand-in for FindingAnalysis.to_dict() / RecordedRun.to_dict().
_ANALYSIS = {
    "variant": "leduc",
    "n_info_states": 288,
    "collapsed_exploitability": 1.75,
    "nash_exploitability": 0.0,
    "spots": [
        {
            "info_key": "[Private: 4][Round 2][Pot: 22][Public: 0]",
            "hero": "K",
            "board": "J",
            "pot": 22,
            "round_num": 2,
            "legal_actions": ["call", "raise", "fold"],
            "nash_probs": {"raise": 0.6, "call": 0.4},
            "collapsed_action": "call",
            "nash_prob_of_collapsed": 0.4,
            "nash_mass_missed": 0.6,
            "mixedness": 0.4,
            "tags": ["mixed"],
        },
        {
            "info_key": "[Private: 2][Round 1][Pot: 2]",
            "hero": "Q",
            "board": "",
            "pot": 2,
            "round_num": 1,
            "legal_actions": ["call", "raise"],
            "nash_probs": {"call": 0.67, "raise": 0.33},
            "collapsed_action": "call",
            "nash_prob_of_collapsed": 0.67,
            "nash_mass_missed": 0.33,
            "mixedness": 0.33,
            "tags": ["mixed"],
        },
    ],
}
_RUN = {
    "model": "Qwen/Qwen3.5-4B",
    "reward_mode": "kl_nash",
    "run_file": "rl-leduc-kl-lr1e4.json",
    "base": {
        "exploitability": 1.0458,
        "mixed_exploitability": 1.4041,
        "mean_nash_prob": 0.3819,
        "exact_match": 0.3904,
        "n": 187,
    },
    "rlvr": {
        "exploitability": 1.75,
        "mixed_exploitability": 1.5569,
        "mean_nash_prob": 0.5745,
        "exact_match": 0.6096,
        "n": 187,
    },
}


def test_fmt_dist_orders_by_probability_descending():
    assert _fmt_dist({"call": 0.4, "raise": 0.6}) == "raise=60%, call=40%"


def test_render_reports_the_worse_exploitability():
    md = render_before_after(_ANALYSIS, _RUN)
    # The headline: greedy exploitability rose from base to RLVR.
    assert "1.046" in md and "1.750" in md
    assert "⬇ worse" in md  # exploitability is flagged as a regression
    assert "⬆" in md  # the proxy metrics improved
    # Live-recomputed collapsed exploitability appears.
    assert "1.750" in md
    assert "288 Leduc info states" in md


def test_render_breaks_down_the_collapsed_decisions():
    md = render_before_after(_ANALYSIS, _RUN, n_spots=6)
    # Both spots and their discarded equilibrium mass are shown.
    assert "board J" in md and "pre-flop" in md
    assert "raise=60%, call=40%" in md
    assert "RLVR commits (after): `call`" in md
    assert "60%" in md  # nash_mass_missed for spot 1


def test_render_respects_n_spots_cap():
    one = render_before_after(_ANALYSIS, _RUN, n_spots=1)
    assert "### Spot 1" in one
    assert "### Spot 2" not in one


def test_render_tolerates_a_run_missing_mixed_exploitability():
    run = {**_RUN, "base": dict(_RUN["base"]), "rlvr": dict(_RUN["rlvr"])}
    del run["base"]["mixed_exploitability"]
    del run["rlvr"]["mixed_exploitability"]
    md = render_before_after(_ANALYSIS, run)
    assert "Mixed exploitability" not in md
    assert "Greedy exploitability" in md  # the present rows still render


@pytest.mark.openspiel
def test_build_offline_artifact_reproduces_the_collapse():
    md = build_offline_artifact(variant="leduc", iterations=300, n_spots=4)
    assert "# PokerEval" in md
    # The collapsed policy's greedy exploitability reproduces the recorded ~1.75.
    assert "collapsed policy's **greedy exploitability**" in md
    assert "### Spot 1" in md


@pytest.mark.openspiel
def test_script_writes_artifact_offline(tmp_path):
    """End-to-end: the script runs offline (no Tinker) and saves Markdown."""
    out = tmp_path / "demo.md"
    proc = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "scripts" / "demo.py"),
         "--iterations", "300", "--n-spots", "3", "--out", str(out)],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    text = out.read_text()
    assert "before vs after" in text
    assert "### Spot 1" in text
