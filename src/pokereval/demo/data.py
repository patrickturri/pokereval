"""Recorded Phase-2 run metrics — the real before/after numbers the demo cites.

These come straight from the committed result of the decisive live Tinker run
(`results/rl-leduc-kl-lr1e4.json`): a LoRA fine-tune of Qwen3.5-4B under the
entropy-regularized `kl_nash` reward. The demo shows them next to the live
collapsed-policy reproduction so the headline is grounded in the actual run, not
asserted. If the file is absent the documented constants are used as a fallback.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_RUN = _REPO_ROOT / "results" / "rl-leduc-kl-lr1e4.json"

# Fallback = the published numbers (docs/phase2-findings.md, Run 3).
_FALLBACK = {
    "before": {
        "exploitability": 1.0458333333333332,
        "mixed_exploitability": 1.4041085059590994,
        "mean_nash_prob": 0.38190271314960644,
        "exact_match": 0.39037433155080214,
        "n": 187,
    },
    "after": {
        "exploitability": 1.7500000000000002,
        "mixed_exploitability": 1.5569280185598546,
        "mean_nash_prob": 0.5744857019656323,
        "exact_match": 0.6096256684491979,
        "n": 187,
    },
}


@dataclass
class RecordedRun:
    model: str
    reward_mode: str
    run_file: str
    base: dict  # EvalReport before RLVR
    rlvr: dict  # EvalReport after RLVR

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "reward_mode": self.reward_mode,
            "run_file": self.run_file,
            "base": self.base,
            "rlvr": self.rlvr,
        }


def load_recorded_run(path: Path | str | None = None) -> RecordedRun:
    """Load the recorded base-vs-RLVR EvalReports, falling back to constants."""
    p = Path(path) if path is not None else _DEFAULT_RUN
    try:
        payload = json.loads(p.read_text())
        before, after = payload["before"], payload["after"]
        run_file = p.name
    except (OSError, KeyError, json.JSONDecodeError):
        before, after = _FALLBACK["before"], _FALLBACK["after"]
        run_file = "rl-leduc-kl-lr1e4.json (fallback constants)"
    return RecordedRun(
        model="Qwen/Qwen3.5-4B",
        reward_mode="kl_nash",
        run_file=run_file,
        base=before,
        rlvr=after,
    )
