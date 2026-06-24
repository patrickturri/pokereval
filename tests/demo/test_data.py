"""Tests for the recorded-run loader (pure JSON, no solver)."""
import json

from pokereval.demo.data import load_recorded_run


def test_loads_real_recorded_run():
    run = load_recorded_run()
    # The decisive RLVR run: greedy exploitability rose 1.046 -> 1.750.
    assert run.base["exploitability"] == 1.0458333333333332
    assert run.rlvr["exploitability"] == 1.7500000000000002
    assert run.rlvr["exact_match"] > run.base["exact_match"]  # proxy improved
    assert run.model == "Qwen/Qwen3.5-4B"


def test_falls_back_when_file_missing(tmp_path):
    run = load_recorded_run(tmp_path / "nope.json")
    assert run.rlvr["exploitability"] == 1.7500000000000002
    assert "fallback" in run.run_file


def test_reads_a_custom_run_file(tmp_path):
    custom = tmp_path / "run.json"
    custom.write_text(
        json.dumps(
            {
                "before": {"exploitability": 0.5, "mixed_exploitability": 0.6,
                           "mean_nash_prob": 0.3, "exact_match": 0.3, "n": 10},
                "after": {"exploitability": 0.9, "mixed_exploitability": 0.8,
                          "mean_nash_prob": 0.5, "exact_match": 0.5, "n": 10},
            }
        )
    )
    run = load_recorded_run(custom)
    assert run.base["exploitability"] == 0.5
    assert run.rlvr["exploitability"] == 0.9
    assert run.run_file == "run.json"
