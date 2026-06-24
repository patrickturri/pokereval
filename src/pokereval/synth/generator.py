"""Generate CFR/Nash-labeled synthetic spots for Kuhn/Leduc.

Wraps the engine (decision-node enumeration) and solver (Nash policy) to emit
``LabeledSpot``s carrying the exact equilibrium action distribution plus
failure-mode tags. This is the labeling half of the Phase-2 flywheel; the spots
double as (a) verifiable-reward training data for RLVR and (b) a targeted eval
set focused on the failure modes surfaced in Phase 1.
"""

from __future__ import annotations

from ..interface.types import GameVariant
from ..engine.kuhn_leduc import build_state, iter_nodes
from ..solver.nash import nash_probs_by_key
from .labeling import classify_distribution
from .types import LabeledSpot


def build_labeled_spots(
    variant: GameVariant,
    iterations: int = 2000,
) -> list[LabeledSpot]:
    """Enumerate every decision node and attach its Nash label + tags."""
    nash = nash_probs_by_key(variant, iterations)
    spots: list[LabeledSpot] = []
    for node in iter_nodes(variant):
        # solver keys the distribution by OpenSpiel action id (as str); the
        # engine's os_legal maps Action value -> os id. Invert to get an
        # Action-value-keyed distribution.
        osid_to_action = {str(osid): av for av, osid in node.os_legal.items()}
        dist_by_osid = nash.get(node.info_key, {})
        action_probs: dict[str, float] = {}
        for osid_str, prob in dist_by_osid.items():
            action_value = osid_to_action.get(osid_str)
            if action_value is not None:
                action_probs[action_value] = action_probs.get(action_value, 0.0) + prob

        state = build_state(variant, node.info_key, node.os_legal)
        tags = classify_distribution(action_probs, node.facing_bet)
        spots.append(LabeledSpot(state=state, nash_action_probs=action_probs, tags=tags))
    return spots


def select_by_tag(spots: list[LabeledSpot], tag: str) -> list[LabeledSpot]:
    """Filter to spots carrying a given failure-mode tag (e.g. 'mixed')."""
    return [s for s in spots if tag in s.tags]


def to_jsonl_records(spots: list[LabeledSpot]) -> list[dict]:
    """Serialize spots to JSON-able dicts (one per line for a .jsonl dataset)."""
    return [s.model_dump() for s in spots]
