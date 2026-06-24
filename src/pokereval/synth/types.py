from pydantic import BaseModel, Field

from ..interface.types import State


class LabeledSpot(BaseModel):
    """A synthetic poker spot with a CFR/Nash ground-truth label.

    This is the unit of the Phase-2 synthetic-data flywheel: a decision state
    paired with its exact equilibrium action distribution (the verifiable
    reward signal for RLVR) and failure-mode tags (so we can target the spots
    where Phase-1 evals show frontier models systematically deviate).
    """

    state: State
    # Equilibrium action distribution, keyed by Action value (sums to ~1).
    nash_action_probs: dict[str, float] = Field(default_factory=dict)
    # Failure-mode / difficulty tags, e.g. "mixed", "facing_bet".
    tags: list[str] = Field(default_factory=list)
