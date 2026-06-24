"""Verifiable grader using Nash equilibrium probabilities or reference actions."""
from pokereval.interface.types import State, Decision, Score


class VerifiableGrader:
    """Grades decisions against Nash probabilities or reference actions."""

    def __init__(self, nash_probs: dict[str, dict[str, float]]):
        """Initialize with Nash probability distributions.

        Args:
            nash_probs: Dictionary mapping info_state_key to distributions.
                       Each distribution maps action string IDs to probabilities.
        """
        self._nash = nash_probs

    def grade(self, state: State, decision: Decision) -> Score:
        """Grade a decision against Nash probabilities or reference actions.

        Args:
            state: The game state.
            decision: The decision (action) to grade.

        Returns:
            Score with value and detail dict.
        """
        key = state.info_state_key
        if key is not None and key in self._nash:
            # Nash probability grading (Kuhn/Leduc)
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

        # Reference action grading (Hold'em)
        ref = state.meta.get("reference_action")
        if ref is not None:
            value = 1.0 if decision.action.value == ref else 0.0
            return Score(value=value, detail={"kind": "reference", "reference": ref})

        return Score(value=0.0, detail={"kind": "ungradeable"})
