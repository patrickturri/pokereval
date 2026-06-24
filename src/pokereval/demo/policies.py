"""Finding-analysis for the Phase-3 demo — the Phase-2 result, made computable.

Phase 2 established that single-step RLVR against a Nash oracle collapsed the
model onto a *deterministic* policy that was byte-identical (on every metric) to
a trivial "first-legal-action" heuristic: Leduc greedy exploitability 1.75,
exact-match 0.61. This module reconstructs that collapsed policy directly from
the CFR solver and the engine — no Tinker, no API keys — so the demo can compute
the headline number *live* and show, per decision node, where a deterministic
policy provably departs from the mixed Nash equilibrium that Leduc requires.

Everything here is offline and credential-free. A live model sampler can later
override the per-spot policy (see `server.create_app(live_sampler=...)`); the
offline default faithfully reproduces the collapsed policy the run produced.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from ..interface.types import GameVariant
from ..solver.nash import exploitability_of_choices
from ..synth.generator import build_labeled_spots

_RANKS = "JQK"  # Leduc deck: 3 ranks × 2 suits, card index // 2 → rank


def _parse_leduc(info_key: str) -> dict:
    """Pull human-readable poker state out of an OpenSpiel Leduc info string.

    Example key:
      [Observer: 1][Private: 2][Round 2][Pot: 22]...[Public: 1][Round1: ...]
    → {hero: "Q", board: "J", pot: 22, round_num: 2}
    """

    def _grab(pat: str):
        m = re.search(pat, info_key)
        return m.group(1) if m else None

    priv = _grab(r"\[Private(?:Card)?: (\d+)\]")
    pub = _grab(r"\[Public(?:Card)?: (\d+)\]")
    pot = _grab(r"\[Pot: (\d+)\]")
    rnd = _grab(r"\[Round:? (\d+)\]")
    return {
        "hero": _RANKS[int(priv) // 2] if priv is not None else "",
        "board": _RANKS[int(pub) // 2] if pub is not None else "",
        "pot": int(pot) if pot is not None else 0,
        "round_num": int(rnd) if rnd is not None else 1,
    }


@dataclass
class SpotView:
    """One Leduc decision node, framed as solver-vs-collapsed-policy."""

    info_key: str
    hero: str  # hero's card rank (J/Q/K)
    board: str  # public card rank, or "" pre-flop
    pot: int
    round_num: int
    legal_actions: list[str]  # Action values, in OpenSpiel legal order
    nash_probs: dict[str, float]  # Action value -> Nash probability (the solver mixes)
    collapsed_action: str  # what the collapsed deterministic policy plays here
    nash_prob_of_collapsed: float  # Nash mass on that single action (< 1 ⇒ wrong to commit)
    nash_mass_missed: float  # equilibrium mass the collapse discards here (1 − above)
    mixedness: float  # 1 − max Nash prob; how much the solver randomizes at this node
    tags: list[str] = field(default_factory=list)


@dataclass
class FindingAnalysis:
    variant: GameVariant
    n_info_states: int
    collapsed_exploitability: float  # live, reproduces the recorded 1.75
    nash_exploitability: float  # 0.0 — a Nash equilibrium is unexploitable by definition
    spots: list[SpotView]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["variant"] = self.variant.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FindingAnalysis":
        return cls(
            variant=GameVariant(d["variant"]),
            n_info_states=d["n_info_states"],
            collapsed_exploitability=d["collapsed_exploitability"],
            nash_exploitability=d["nash_exploitability"],
            spots=[SpotView(**s) for s in d["spots"]],
        )


def _first_legal_osid(spot) -> int:
    """OpenSpiel action id of the action the collapsed policy commits to."""
    first = spot.state.legal_actions[0].value
    return int(spot.state.meta["os_legal"][first])


def first_legal_choices(spots) -> dict[str, int]:
    """The collapsed deterministic policy: info_key -> first-legal OpenSpiel id."""
    return {s.state.info_state_key: _first_legal_osid(s) for s in spots}


def collapsed_exploitability(variant: GameVariant, spots) -> float:
    """Live greedy exploitability of the collapsed policy (reproduces recorded 1.75)."""
    return exploitability_of_choices(variant, first_legal_choices(spots))


def build_analysis(
    variant: GameVariant = GameVariant.LEDUC,
    iterations: int = 2000,
    max_spots: int = 48,
) -> FindingAnalysis:
    """Build the full finding-analysis for the demo.

    Display spots are the genuinely *mixed* nodes (Nash randomizes ≥ 10%),
    deduplicated to one representative per distinct equilibrium shape so the
    explorer shows the variety of decisions the collapse flattens — not 16 copies
    of the same 50/50. The single global best-response (collapsed exploitability)
    is the only expensive call; per-node figures are exact and free.
    """
    spots = build_labeled_spots(variant, iterations)

    # Genuinely mixed nodes only (the "mixed" tag ⇒ max Nash prob ≤ 0.9): these
    # are where a deterministic policy provably leaves equilibrium value behind.
    mixed = [s for s in spots if "mixed" in s.tags and len(s.state.legal_actions) > 1]

    seen: set = set()
    views: list[SpotView] = []
    for s in mixed:
        legal = [a.value for a in s.state.legal_actions]
        collapsed_av = legal[0]
        nash = s.nash_action_probs
        # Dedupe by equilibrium shape (legal set + Nash rounded to 5%) so each
        # distinct decision type appears once, not once per hidden card.
        sig = (tuple(legal), tuple(sorted((a, round(p * 20) / 20) for a, p in nash.items())))
        if sig in seen:
            continue
        seen.add(sig)
        parsed = _parse_leduc(s.state.info_state_key or "")
        nash_on_collapsed = nash.get(collapsed_av, 0.0)
        views.append(
            SpotView(
                info_key=s.state.info_state_key,
                hero=parsed["hero"],
                board=parsed["board"],
                pot=parsed["pot"],
                round_num=parsed["round_num"],
                legal_actions=legal,
                nash_probs={a: round(p, 4) for a, p in nash.items()},
                collapsed_action=collapsed_av,
                nash_prob_of_collapsed=round(nash_on_collapsed, 4),
                nash_mass_missed=round(1.0 - nash_on_collapsed, 4),
                mixedness=round(1.0 - max(nash.values(), default=1.0), 4),
                tags=list(s.tags),
            )
        )

    views.sort(key=lambda v: v.mixedness, reverse=True)

    return FindingAnalysis(
        variant=variant,
        n_info_states=len(spots),
        collapsed_exploitability=round(collapsed_exploitability(variant, spots), 4),
        nash_exploitability=0.0,
        spots=views[:max_spots],
    )
