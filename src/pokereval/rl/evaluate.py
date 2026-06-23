from __future__ import annotations
from dataclasses import dataclass
from pokereval.interface.types import GameVariant
from pokereval.interface.prompts import render_state, parse_decision, ParseError
from pokereval.solver.nash import exploitability_of_choices
from pokereval.rl.reward import spot_reward
from pokereval.rl.sampler import SamplingClient


@dataclass
class EvalReport:
    exploitability: float
    mean_nash_prob: float
    exact_match: float
    n: int


def greedy_choices(variant: GameVariant, spots, sampler: SamplingClient) -> dict[str, int]:
    choices: dict[str, int] = {}
    for spot in spots:
        prompt = render_state(spot.state)
        comp = sampler.sample(prompt, n=1, temperature=0.0)[0]
        try:
            dec = parse_decision(comp.text, spot.state.legal_actions)
        except ParseError:
            continue
        os_id = spot.state.meta.get("os_legal", {}).get(dec.action.value)
        key = spot.state.info_state_key
        if os_id is not None and key is not None:
            choices[key] = int(os_id)
    return choices


def evaluate_policy(variant: GameVariant, all_spots, eval_spots, sampler: SamplingClient) -> EvalReport:
    choices = greedy_choices(variant, all_spots, sampler)
    expl = exploitability_of_choices(variant, choices)

    nash_sum = 0.0
    exact = 0
    for spot in eval_spots:
        prompt = render_state(spot.state)
        comp = sampler.sample(prompt, n=1, temperature=0.0)[0]
        r, _ = spot_reward(spot, comp.text)
        nash_sum += r
        if spot.nash_action_probs and r == max(spot.nash_action_probs.values()):
            exact += 1
    n = len(eval_spots)
    return EvalReport(
        exploitability=float(expl),
        mean_nash_prob=nash_sum / n if n else 0.0,
        exact_match=exact / n if n else 0.0,
        n=n,
    )
