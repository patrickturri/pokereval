"""Failure-mode classification of equilibrium action distributions.

Pure functions (no OpenSpiel) so the labeling logic is unit-testable on its own.
The tags drive the targeted-failure-mode flywheel: Phase-1 evals suggest frontier
models collapse *mixed* equilibrium strategies to pure ones and misplay
*facing-a-bet* decisions, so those are the spots Phase-2 generates against.
"""

from __future__ import annotations

# Minimum probability mass off the modal action for a decision to count as a
# genuinely mixed (randomized) equilibrium rather than effectively pure.
MIXED_THRESHOLD = 0.1


def mixedness(dist: dict[str, float]) -> float:
    """How far the distribution is from a pure strategy, in [0, 1).

    0.0 means all mass on a single action; higher means more randomization.
    Defined as ``1 - max(prob)`` — interpretable and tie-robust.
    """
    if not dist:
        return 0.0
    return 1.0 - max(dist.values())


def classify_distribution(dist: dict[str, float], facing_bet: bool) -> list[str]:
    """Tag a Nash action distribution by failure mode / difficulty.

    Tags are stable and order-deterministic so downstream selection and tests
    are reproducible.
    """
    tags: list[str] = []
    if mixedness(dist) >= MIXED_THRESHOLD:
        tags.append("mixed")
    else:
        tags.append("near_deterministic")
    if facing_bet:
        tags.append("facing_bet")
    return tags
