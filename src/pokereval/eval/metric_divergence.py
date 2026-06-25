"""Action-match vs. exploitability: a credential-free benchmark-methodology panel.

PokerEval's headline finding — that RLVR *raised* per-decision agreement with the
Nash solver (action-match) while making the whole policy *more* exploitable — is a
single base-vs-RLVR data point that needs a live training run to reproduce. This
module reproduces the same phenomenon from first principles, with no model and no
API keys, by scoring a panel of synthetic policies (all derived from the CFR
solver) on *both* metrics at once.

Two metrics, defined on a policy `π` over a variant's decision nodes:

- **action-match** — the policy-level analogue of the leaderboard's per-decision
  ``exact_match``. For a stochastic policy sampled once per node, the expected
  exact-match rate is the mean mass the policy puts on the Nash-modal action:
  ``mean_node π(node)[argmax_a nash(node, a)]``. This is exactly what PokerBench
  (AAAI 2025) rewards: did you pick the solver's best action?
- **exploitability** — chips/hand a best-responder extracts (0 = Nash), via the
  solver's ``exploitability_of_distribution`` / ``exploitability_of_choices``.

The knob that exposes the disagreement is **temperature-sharpening** the Nash
policy: ``p_i ∝ nash_i ** (1/τ)``. At τ=1 the policy *is* the Nash mix
(exploitability ≈ 0, action-match < 1 wherever Nash mixes); as τ→0 it collapses
onto the argmax, so action-match rises toward 1.0 **and exploitability rises too**.
The two metrics move together in the wrong direction — action-match is blind to the
mixing a poker equilibrium requires.
"""
from __future__ import annotations

import math

from pydantic import BaseModel

from ..interface.types import GameVariant

# Default temperature sweep: τ=2 (flatter than Nash) down through the Nash mix
# (τ=1) to a near-collapsed policy (τ=0.1).
DEFAULT_TAUS: tuple[float, ...] = (2.0, 1.0, 0.5, 0.25, 0.1)

Dist = dict[int, float]
DistByKey = dict[str, Dist]


# --- pure-math core (no OpenSpiel) -------------------------------------------

def _normalize_int_keys(dist_by_key: dict[str, dict]) -> DistByKey:
    """Coerce action keys to ints (the solver hands back stringified ids)."""
    return {k: {int(a): float(p) for a, p in d.items()} for k, d in dist_by_key.items()}


def sharpened(nash_by_key: dict[str, dict], tau: float) -> DistByKey:
    """Per-node temperature-sharpen a Nash distribution: ``p_i ∝ nash_i**(1/τ)``.

    τ=1 returns the input mix unchanged; τ→0 concentrates all mass on the modal
    action (argmax collapse); large τ flattens toward uniform over the support.
    Actions with zero Nash mass stay at zero. Computed in log-space and shifted by
    the max so τ→0 stays numerically stable instead of underflowing to all-zeros.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    out: DistByKey = {}
    inv = 1.0 / tau
    for key, dist in nash_by_key.items():
        logs = {int(a): inv * math.log(p) for a, p in dist.items() if p > 0.0}
        if not logs:
            out[key] = {}
            continue
        m = max(logs.values())
        weights = {a: math.exp(lg - m) for a, lg in logs.items()}
        z = sum(weights.values())
        out[key] = {a: w / z for a, w in weights.items()}
    return out


def argmax_by_key(dist_by_key: dict[str, dict]) -> dict[str, int]:
    """Modal action id per node."""
    return {k: int(max(d, key=d.get)) for k, d in dist_by_key.items() if d}


def action_match_rate(dist_by_key: dict[str, dict], nash_argmax: dict[str, int]) -> float:
    """Mean policy mass on the Nash-modal action (expected exact-match rate)."""
    keys = [k for k in nash_argmax if k in dist_by_key]
    if not keys:
        return 0.0
    total = 0.0
    for k in keys:
        d = dist_by_key[k]
        total += float(d.get(nash_argmax[k], d.get(str(nash_argmax[k]), 0.0)))
    return total / len(keys)


# --- panel -------------------------------------------------------------------

class PanelRow(BaseModel):
    name: str
    tau: float | None = None
    action_match: float
    greedy_exploitability: float
    mixed_exploitability: float


def _point_mass(choices: dict[str, int]) -> DistByKey:
    return {k: {a: 1.0} for k, a in choices.items()}


def _uniform_over_support(nash_int: DistByKey) -> DistByKey:
    out: DistByKey = {}
    for k, d in nash_int.items():
        legal = list(d.keys())
        if legal:
            out[k] = {a: 1.0 / len(legal) for a in legal}
    return out


def _anti_nash_choices(nash_int: DistByKey) -> dict[str, int]:
    """Always pick the *least* likely Nash action — a deliberately bad policy."""
    return {k: int(min(d, key=d.get)) for k, d in nash_int.items() if d}


def evaluate_panel(
    variant: str | GameVariant,
    iterations: int = 2000,
    taus: tuple[float, ...] = DEFAULT_TAUS,
) -> list[PanelRow]:
    """Score a panel of solver-derived policies on action-match AND exploitability.

    Credential-free: everything is computed from the CFR Nash solution; no model
    is queried. Imports the OpenSpiel-backed solver lazily so the pure-math helpers
    above remain importable without pyspiel.
    """
    from ..solver.nash import (
        nash_probs_by_key,
        exploitability_of_choices,
        exploitability_of_distribution,
    )

    var = GameVariant(variant) if not isinstance(variant, GameVariant) else variant
    nash_int = _normalize_int_keys(nash_probs_by_key(var, iterations))
    nash_argmax = argmax_by_key(nash_int)

    def row(name: str, dist: DistByKey, tau: float | None) -> PanelRow:
        choices = argmax_by_key(dist)
        return PanelRow(
            name=name,
            tau=tau,
            action_match=action_match_rate(dist, nash_argmax),
            greedy_exploitability=exploitability_of_choices(var, choices),
            mixed_exploitability=exploitability_of_distribution(var, dist),
        )

    rows: list[PanelRow] = [
        row("nash_mixed", nash_int, None),
        row("nash_greedy", _point_mass(nash_argmax), None),
        row("uniform", _uniform_over_support(nash_int), None),
        row("anti_nash", _point_mass(_anti_nash_choices(nash_int)), None),
    ]
    # τ-sweep, descending so the table reads Nash-mix → collapse.
    for tau in sorted(taus, reverse=True):
        rows.append(row(f"sharpened τ={tau:g}", sharpened(nash_int, tau), tau))
    return rows


# --- rendering ----------------------------------------------------------------

def render_panel_markdown(rows: list[PanelRow]) -> str:
    header = "| Policy | τ | Action-Match ↑ | Greedy Expl. | Mixed Expl. |"
    sep = "|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        tau = f"{r.tau:g}" if r.tau is not None else "—"
        lines.append(
            f"| {r.name} | {tau} | {r.action_match:.3f} | "
            f"{r.greedy_exploitability:.4f} | {r.mixed_exploitability:.4f} |"
        )
    return "\n".join(lines)


def verdict(rows: list[PanelRow]) -> str:
    by = {r.name: r for r in rows}
    mixed, greedy = by.get("nash_mixed"), by.get("nash_greedy")
    sweep = sorted((r for r in rows if r.tau is not None), key=lambda r: r.tau, reverse=True)
    parts: list[str] = []
    if mixed and greedy:
        parts.append(
            f"`nash_greedy` scores a perfect action-match ({greedy.action_match:.2f}) "
            f"yet is the *more* exploitable policy ({greedy.mixed_exploitability:.3f} vs "
            f"`nash_mixed`'s {mixed.mixed_exploitability:.3f}) — the better-scoring policy "
            f"is the worse one."
        )
    if len(sweep) >= 2:
        hi, lo = sweep[0], sweep[-1]
        parts.append(
            f"Across the τ-sweep ({hi.tau:g}→{lo.tau:g}), action-match rises "
            f"{hi.action_match:.2f}→{lo.action_match:.2f} while mixed exploitability rises "
            f"{hi.mixed_exploitability:.3f}→{lo.mixed_exploitability:.3f}: the two metrics "
            f"move together in the wrong direction."
        )
    parts.append("Action-match is blind to the mixing a poker equilibrium requires.")
    return " ".join(parts)
