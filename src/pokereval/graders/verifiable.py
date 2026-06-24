"""Verifiable grader — the exact, ground-truth reward signal.

For Kuhn/Leduc the score is the probability mass the Nash policy assigns to the
chosen action (so matching a mixed strategy is rewarded proportionally, and the
modal action counts as an "exact match"). For Hold'em there is no exact policy,
so we fall back to matching a curated reference action.
"""

from ..interface.types import Decision, Score, State


class VerifiableGrader:
    def __init__(self, nash_probs: dict[str, dict[str, float]]):
        self._nash = nash_probs

    def grade(self, state: State, decision: Decision) -> Score:
        key = state.info_state_key
        if key is not None and key in self._nash:
            os_legal = state.meta.get("os_legal", {})
            os_id = os_legal.get(decision.action.value)
            dist = self._nash[key]
            if os_id is None or str(os_id) not in dist:
                return Score(value=0.0, detail={"kind": "ungradeable"})
            value = dist[str(os_id)]
            return Score(
                value=value,
                detail={
                    "kind": "nash",
                    "nash_prob": value,
                    "exact_match": value == max(dist.values()),
                },
            )
        ref = state.meta.get("reference_action")
        if ref is not None:
            value = 1.0 if decision.action.value == ref else 0.0
            return Score(value=value, detail={"kind": "reference", "reference": ref})
        return Score(value=0.0, detail={"kind": "ungradeable"})
