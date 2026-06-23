from __future__ import annotations
from pokereval.interface.types import Decision
from pokereval.interface.prompts import parse_decision, ParseError
from pokereval.synth.types import LabeledSpot


def spot_reward(spot: LabeledSpot, completion_text: str) -> tuple[float, Decision | None]:
    """Verifiable reward: Nash probability of the chosen action, else 0.0."""
    try:
        decision = parse_decision(completion_text, spot.state.legal_actions)
    except ParseError:
        return 0.0, None
    reward = spot.nash_action_probs.get(decision.action.value, 0.0)
    return float(reward), decision
