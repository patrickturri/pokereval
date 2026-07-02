"""Credential-free before/after artifact for the Phase-2 finding.

`scripts/demo.py` uses this to emit a shareable Markdown write-up — the headline
result reconstructed from the CFR solver and the recorded Tinker run, with no API
keys and no Tinker account. It reuses the demo package's offline reconstruction:

- `demo.policies.build_analysis` — rebuilds the collapsed RLVR policy directly
  from the solver and recomputes its greedy exploitability *live* (reproducing
  the recorded 1.75), plus the per-node Nash mix the collapse discards.
- `demo.data.load_recorded_run` — the real base-vs-RLVR EvalReports from the
  decisive live run (with documented fallback constants).

The rendering half (`render_before_after`) operates on plain dicts, so it is
importable and unit-testable without OpenSpiel; only `build_offline_artifact`
touches the solver.
"""
from __future__ import annotations


def _fmt_dist(dist: dict[str, float]) -> str:
    """Render an action distribution as ``raise=60%, call=40%`` (most-likely first)."""
    return ", ".join(
        f"{a}={p:.0%}" for a, p in sorted(dist.items(), key=lambda kv: -kv[1])
    )


def _spot_headline(spot: dict) -> str:
    """One-line description of a decision node: ``hero K, board J, round 2, pot 22``."""
    parts = [f"hero {spot.get('hero') or '?'}"]
    board = spot.get("board")
    parts.append(f"board {board}" if board else "pre-flop")
    parts.append(f"round {spot.get('round_num', 1)}")
    parts.append(f"pot {spot.get('pot', 0)}")
    return ", ".join(parts)


def render_before_after(analysis: dict, run: dict, n_spots: int = 6) -> str:
    """Render the before/after finding as Markdown.

    ``analysis`` is a ``FindingAnalysis.to_dict()`` and ``run`` a
    ``RecordedRun.to_dict()``; both are plain dicts so this function needs no
    solver/OpenSpiel imports. ``n_spots`` caps the per-decision breakdown.
    """
    base, rlvr = run["base"], run["rlvr"]

    def _row(label: str, key: str, fmt: str, arrow: str) -> str:
        b, a = base.get(key), rlvr.get(key)
        if b is None or a is None:
            return ""
        return f"| {label} | {b:{fmt}} | **{a:{fmt}}** {arrow} |\n"

    lines: list[str] = []
    lines.append("# PokerEval — the Phase-2 finding, before vs after")
    lines.append("")
    lines.append(
        "Reconstructed offline from the CFR solver and the recorded Tinker run — "
        "**no API keys, no Tinker account**. The collapsed policy is rebuilt from "
        "the solver and its exploitability recomputed live below."
    )
    lines.append("")
    lines.append("## Aggregate — RLVR improved the proxy but *worsened* the true objective")
    lines.append("")
    lines.append("| Metric (held-out Leduc) | Base | After RLVR |")
    lines.append("|---|---|---|")
    table = (
        _row("Exact-match with Nash action", "exact_match", ".1%", "⬆")
        + _row("Mean Nash-probability mass", "mean_nash_prob", ".3f", "⬆")
        + _row("Greedy exploitability (vs solver)", "exploitability", ".3f", "⬇ worse")
        + _row("Mixed exploitability (vs solver)", "mixed_exploitability", ".3f", "⬇ worse")
    )
    lines.append(table.rstrip("\n"))
    lines.append("")
    lines.append(
        f"Model: `{run.get('model', '?')}` · reward: `{run.get('reward_mode', '?')}` · "
        f"source: `{run.get('run_file', '?')}`"
    )
    lines.append("")
    lines.append(
        f"Live re-computation of the collapsed policy's **greedy exploitability**: "
        f"**{analysis['collapsed_exploitability']:.3f}** "
        f"(Nash equilibrium: {analysis['nash_exploitability']:.1f}), "
        f"across {analysis['n_info_states']} Leduc info states."
    )
    lines.append("")
    lines.append(
        "The proxy (exact-match, Nash-prob) went **up** while exploitability got "
        "**worse**: single-step RLVR collapsed the policy onto a deterministic mode, "
        "discarding the mixing Leduc's equilibrium requires."
    )
    lines.append("")
    lines.append("## Where it collapsed — decision by decision")
    lines.append("")
    lines.append(
        "Each spot below is a genuinely *mixed* Leduc node (the solver randomizes). "
        "**Before** is the Nash mix the equilibrium plays; **after** is the single "
        "action the collapsed RLVR policy commits to — and the equilibrium mass that "
        "commitment throws away."
    )
    lines.append("")

    spots = analysis.get("spots", [])[:n_spots]
    for i, spot in enumerate(spots, 1):
        lines.append(f"### Spot {i} — {_spot_headline(spot)}")
        lines.append(f"- legal: `{', '.join(spot['legal_actions'])}`")
        lines.append(f"- Nash mix (before): `{_fmt_dist(spot['nash_probs'])}`")
        lines.append(
            f"- RLVR commits (after): `{spot['collapsed_action']}` "
            f"— discards **{spot['nash_mass_missed']:.0%}** of the equilibrium mass"
        )
        lines.append("")

    return "\n".join(lines).rstrip("\n") + "\n"


def build_offline_artifact(
    variant: str = "leduc",
    iterations: int = 2000,
    max_spots: int = 48,
    n_spots: int = 6,
    run_path=None,
) -> str:
    """Build the full before/after Markdown artifact offline (needs OpenSpiel).

    Runs CFR to reconstruct the collapsed policy and its live exploitability, loads
    the recorded run metrics, and renders the two together. Credential-free.
    """
    from ..interface.types import GameVariant
    from .policies import build_analysis
    from .data import load_recorded_run

    analysis = build_analysis(
        GameVariant(variant), iterations=iterations, max_spots=max_spots
    )
    run = load_recorded_run(run_path)
    return render_before_after(analysis.to_dict(), run.to_dict(), n_spots=n_spots)
