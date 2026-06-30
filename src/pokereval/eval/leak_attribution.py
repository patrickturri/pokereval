"""Exploitability leak attribution — *where* does a policy bleed chips?

A single exploitability number ("this policy leaks 1.75 chips/hand to a
best-responder") tells you a policy is exploitable but not *which decisions* are
responsible. This module attributes that number across a policy's information
states by a counterfactual **repair**: replace one info state's action
distribution with the Nash distribution, recompute exploitability, and read off
the drop. That drop is the value a best-responder was extracting *at that
decision* — the chips that decision is leaking.

Why this is the right lens for the PokerEval story
--------------------------------------------------
The headline finding is that single-step RLVR *collapses* a mixed policy onto its
modal action and so becomes **more** exploitable (greedy 1.05 → 1.75). "More
exploitable" is an aggregate; leak attribution localizes it. Run it on the
collapsed policy and the leak concentrates on exactly the **mixed** decision
nodes — the spots where committing to one action discards equilibrium mass that a
best-responder then punishes. Repairing the handful of worst nodes back to the
Nash mix recovers most of the gap, and repairing *all* deviating nodes recovers
it exactly (full repair == the Nash policy). It turns the abstract metric into a
ranked, legible list of culpable decisions.

Method & caveats
----------------
The marginal leak of a node is ``E(π) − E(π with that node repaired to Nash)``,
where ``E`` is OpenSpiel's exact best-response exploitability (no sampling, no
oracle beyond the solver — the same one used for every other number in the repo).
Marginal leaks are a *one-at-a-time* decomposition, so they need not sum to the
total: best-response is nonlinear, and fixing one node can change how much a
best-responder gains elsewhere. The **cumulative** curve (repair nodes in ranked
order and remeasure) is the honest accounting — it is exact at every prefix and
lands precisely on the Nash exploitability once every deviating node is repaired.
This is the same family of idea as a *local best response* probe (Lisý &
Bowling, "Equilibrium Approximation Quality of Current No-Limit Poker Bots",
2017): localize where a strategy is exploitable rather than only scoring the
whole policy. Everything here is offline and credential-free.
"""
from __future__ import annotations

from pydantic import BaseModel

from ..interface.types import GameVariant
from .metric_divergence import _normalize_int_keys

Dist = dict[int, float]
DistByKey = dict[str, Dist]


# --- pure-Python core (no OpenSpiel) -----------------------------------------


def repair(policy: dict[str, dict], nash: dict[str, dict], keys) -> DistByKey:
    """Return a copy of ``policy`` with each info state in ``keys`` replaced by
    its Nash distribution.

    The input ``policy`` is never mutated. A key absent from ``nash`` is left
    unchanged (nothing to repair it to).
    """
    keyset = set(keys)
    out: DistByKey = {k: dict(v) for k, v in policy.items()}
    for k in keyset:
        if k in nash:
            out[k] = dict(nash[k])
    return out


def deviating_keys(policy: dict[str, dict], nash: dict[str, dict], atol: float = 1e-6) -> list[str]:
    """Info states where ``policy`` differs from ``nash`` by more than ``atol``.

    These are the only nodes that can carry a leak: repairing a node already at
    Nash leaves exploitability unchanged. Action ids are coerced to ints and
    compared over the union of both supports (a missing action counts as 0.0),
    so stringified solver keys and dropped zero-mass actions compare correctly.
    """
    pol = _normalize_int_keys(policy)
    nsh = _normalize_int_keys(nash)
    out: list[str] = []
    for k, pd in pol.items():
        nd = nsh.get(k, {})
        actions = set(pd) | set(nd)
        if any(abs(pd.get(a, 0.0) - nd.get(a, 0.0)) > atol for a in actions):
            out.append(k)
    return out


class NodeLeak(BaseModel):
    """One decision node's contribution to a policy's exploitability."""

    info_key: str
    marginal_leak: float  # E(π) − E(π with this node repaired to Nash); chips/hand
    n_legal: int
    policy_dist: dict[int, float] | None = None
    nash_dist: dict[int, float] | None = None


def rank_leaks(leaks: list[NodeLeak]) -> list[NodeLeak]:
    """Leaks, worst (largest marginal leak) first."""
    return sorted(leaks, key=lambda n: n.marginal_leak, reverse=True)


# --- report ------------------------------------------------------------------


class LeakReport(BaseModel):
    variant: str
    baseline_exploitability: float  # E(π) — the policy under attribution
    nash_exploitability: float      # E(Nash) — the floor full repair lands on
    leaks: list[NodeLeak]           # deviating nodes, worst-first
    # (n_nodes_repaired, exploitability_after) repairing in ranked order; the
    # last point repairs every deviating node and equals nash_exploitability.
    cumulative: list[tuple[int, float]]


def attribute_leaks(
    variant: str | GameVariant,
    policy: dict[str, dict],
    nash: dict[str, dict] | None = None,
    iterations: int = 2000,
    atol: float = 1e-6,
    cumulative: bool = True,
    cumulative_top: int | None = None,
) -> LeakReport:
    """Attribute ``policy``'s exploitability across its decision nodes.

    Parameters
    ----------
    variant:
        ``GameVariant`` (or its string value) to evaluate.
    policy:
        ``info_key -> {action_id: prob}``. Action ids may be ints or strings.
    nash:
        The reference Nash policy in the same shape. If ``None`` it is solved
        with CFR for ``iterations`` steps. Pass it in to reuse a cached solution.
    iterations:
        CFR iterations used only when ``nash`` is solved here.
    atol:
        Absolute tolerance for "this node deviates from Nash".
    cumulative:
        If True (default) compute the ranked cumulative-repair curve. Costs one
        extra best-response solve per recorded point.
    cumulative_top:
        Cap the cumulative curve to the top ``cumulative_top`` prefixes plus the
        full-repair endpoint (which always equals the Nash exploitability). None
        records every prefix. Bounds best-response solves on larger games.

    The solver is imported lazily so the pure helpers above stay importable
    without OpenSpiel.
    """
    from ..solver.nash import exploitability_of_distribution, nash_probs_by_key

    var = GameVariant(variant) if not isinstance(variant, GameVariant) else variant
    pol = _normalize_int_keys(policy)
    nsh = _normalize_int_keys(nash) if nash is not None else _normalize_int_keys(
        nash_probs_by_key(var, iterations)
    )

    def expl(dist: DistByKey) -> float:
        return float(exploitability_of_distribution(var, dist))

    baseline = expl(pol)
    nash_expl = expl(nsh)

    dev = deviating_keys(pol, nsh, atol)
    leaks: list[NodeLeak] = []
    for k in dev:
        marginal = baseline - expl(repair(pol, nsh, [k]))
        # Legal-action count is the support union: a collapsed policy stores a
        # point mass (one entry), so the true arity comes from the Nash support.
        n_legal = len(set(pol[k]) | set(nsh.get(k, {})))
        leaks.append(
            NodeLeak(
                info_key=k,
                marginal_leak=marginal,
                n_legal=n_legal,
                policy_dist=pol[k],
                nash_dist=nsh.get(k),
            )
        )
    leaks = rank_leaks(leaks)

    cumulative_pts: list[tuple[int, float]] = []
    if cumulative and leaks:
        n = len(leaks)
        # Record every prefix, or — when capped — the top ``cumulative_top``
        # prefixes plus the full-repair endpoint (which always lands on the Nash
        # floor). Capping bounds the best-response solves on large games while
        # keeping the legible "top-k closes X%" curve and the exact endpoint.
        if cumulative_top is None:
            record = set(range(1, n + 1))
        else:
            record = set(range(1, min(cumulative_top, n) + 1)) | {n}
        repaired: list[str] = []
        for i, leak in enumerate(leaks, start=1):
            repaired.append(leak.info_key)
            if i in record:
                cumulative_pts.append((i, expl(repair(pol, nsh, repaired))))

    return LeakReport(
        variant=var.value,
        baseline_exploitability=baseline,
        nash_exploitability=nash_expl,
        leaks=leaks,
        cumulative=cumulative_pts,
    )


# --- rendering ----------------------------------------------------------------


def render_markdown(report: LeakReport, top: int | None = 8) -> str:
    """Render a leak report as a Markdown table + a cumulative-repair summary."""
    gap = report.baseline_exploitability - report.nash_exploitability
    lines = [
        f"**Policy exploitability:** {report.baseline_exploitability:.4f} "
        f"(Nash floor {report.nash_exploitability:.4f}; gap {gap:+.4f} chips/hand "
        f"across {len(report.leaks)} deviating nodes)",
        "",
        "| Rank | Info state | Legal | Marginal leak ↓ | Policy → Nash |",
        "|---:|---|---:|---:|---|",
    ]
    shown = report.leaks if top is None else report.leaks[:top]
    for i, leak in enumerate(shown, start=1):
        pol_s = _fmt_dist(leak.policy_dist)
        nash_s = _fmt_dist(leak.nash_dist)
        key = leak.info_key if len(leak.info_key) <= 48 else leak.info_key[:45] + "…"
        lines.append(
            f"| {i} | `{key}` | {leak.n_legal} | {leak.marginal_leak:+.4f} | "
            f"{pol_s} → {nash_s} |"
        )
    if report.cumulative:
        n_top, share = _gap_share(report)
        lines += [
            "",
            f"Repairing the top {n_top} node(s) to Nash "
            f"closes {share:.0%} of the exploitability gap; repairing all "
            f"{report.cumulative[-1][0]} deviating nodes recovers the Nash policy "
            f"exactly ({report.cumulative[-1][1]:.4f}).",
        ]
    return "\n".join(lines)


_ELBOW = 3  # how many leading nodes the summary line credits


def _gap_share(report: LeakReport) -> tuple[int, float]:
    """Return ``(n_nodes, fraction)``: the largest recorded prefix of at most
    ``_ELBOW`` nodes, and the fraction of the baseline→Nash gap it closes.

    Selecting by node count (not list position) keeps the summary correct even
    when the cumulative curve is capped to the top few prefixes plus the endpoint.
    """
    gap = report.baseline_exploitability - report.nash_exploitability
    eligible = [c for c in report.cumulative if c[0] <= _ELBOW]
    if gap <= 0 or not eligible:
        return (eligible[-1][0] if eligible else 0), 0.0
    n_top, expl_after = max(eligible, key=lambda c: c[0])
    closed = report.baseline_exploitability - expl_after
    return n_top, max(0.0, min(1.0, closed / gap))


def _fmt_dist(dist: dict[int, float] | None) -> str:
    if not dist:
        return "—"
    parts = [f"{a}:{p:.2f}" for a, p in sorted(dist.items())]
    return "{" + ", ".join(parts) + "}"
