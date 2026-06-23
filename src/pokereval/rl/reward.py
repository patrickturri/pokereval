from __future__ import annotations
import math
from pokereval.interface.types import Decision
from pokereval.interface.prompts import parse_decision, ParseError
from pokereval.synth.types import LabeledSpot

# Penalty (in the KL-reward's log space) for an illegal/unparseable completion.
ILLEGAL_PENALTY = -6.0


def spot_reward(spot: LabeledSpot, completion_text: str) -> tuple[float, Decision | None]:
    """Verifiable reward: Nash probability of the chosen action, else 0.0.

    This is the descriptive per-decision metric (used for eval reporting). The
    training reward is ``kl_to_nash_reward`` below.
    """
    try:
        decision = parse_decision(completion_text, spot.state.legal_actions)
    except ParseError:
        return 0.0, None
    reward = spot.nash_action_probs.get(decision.action.value, 0.0)
    return float(reward), decision


def kl_to_nash_reward(
    spot: LabeledSpot,
    completion_text: str,
    action_logprob: float,
    eps: float = 1e-3,
) -> float:
    """Reward whose policy-gradient optimum is the Nash *distribution*, not its mode.

    Maximizing ``E_{a~pi}[log nash(a) - log pi(a)] = -KL(pi || nash)`` drives the
    model's per-state action distribution toward Nash (preserving mixing), unlike
    ``nash_prob(a)`` whose optimum collapses onto the single modal action and
    *raises* exploitability in a mixed-equilibrium game like Leduc.

    ``action_logprob`` is ``log pi(a)`` — the summed sampling logprobs of the
    completion that expressed action ``a``.
    """
    nash_prob, decision = spot_reward(spot, completion_text)
    if decision is None:
        return ILLEGAL_PENALTY
    return math.log(max(nash_prob, eps)) - float(action_logprob)
