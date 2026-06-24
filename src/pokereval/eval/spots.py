"""Spot sets — the evaluation data.

A :class:`SpotSet` is just a list of :class:`State` decision points plus the
Nash probabilities needed to grade them. Kuhn/Leduc sets are enumerated and
solved on the fly; the Hold'em set is hand-curated JSONL with reference actions.
"""

import json
from pathlib import Path

from pydantic import BaseModel, Field

from ..engine.kuhn_leduc import iter_nodes
from ..interface.types import Action, GameVariant, State
from ..solver.nash import nash_probs_by_key


class SpotSet(BaseModel):
    spots: list[State] = Field(default_factory=list)
    nash_probs: dict[str, dict[str, float]] = Field(default_factory=dict)


def build_kuhn_leduc_spotset(
    variant: GameVariant, iterations: int = 2000
) -> SpotSet:
    """Enumerate every decision node of ``variant`` and solve it with CFR."""
    nash = nash_probs_by_key(variant, iterations)
    spots: list[State] = []
    for node in iter_nodes(variant):
        legal = [Action(v) for v in node.os_legal]
        spots.append(
            State(
                variant=variant,
                hero_cards=[node.info_key],  # info string encodes card + history
                pot=0.0,
                history=node.info_key,
                legal_actions=legal,
                info_state_key=node.info_key,
                meta={"os_legal": node.os_legal},
            )
        )
    return SpotSet(spots=spots, nash_probs=nash)


def load_holdem_spotset(path: str | Path) -> SpotSet:
    """Load curated Hold'em spots from a JSONL file of ``State``-shaped dicts."""
    spots: list[State] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            spots.append(State.model_validate(json.loads(line)))
    return SpotSet(spots=spots, nash_probs={})
