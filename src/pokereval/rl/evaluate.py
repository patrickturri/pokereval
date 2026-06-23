from __future__ import annotations
from dataclasses import dataclass
from pokereval.interface.types import GameVariant
from pokereval.interface.prompts import render_state, parse_decision, ParseError
from pokereval.solver.nash import exploitability_of_choices, exploitability_of_distribution
from pokereval.rl.reward import spot_reward
from pokereval.rl.sampler import SamplingClient


@dataclass
class EvalReport:
    exploitability: float          # greedy (argmax) deterministic policy
    mixed_exploitability: float    # headline: mixed policy from sampled action freqs
    mean_nash_prob: float
    exact_match: float
    n: int


def greedy_choices(variant: GameVariant, spots, sampler: SamplingClient) -> dict[str, int]:
    prompts = [render_state(s.state) for s in spots]
    batches = sampler.sample_many(prompts, n=1, temperature=0.0)
    choices: dict[str, int] = {}
    for spot, comps in zip(spots, batches):
        if not comps:
            continue
        try:
            dec = parse_decision(comps[0].text, spot.state.legal_actions)
        except ParseError:
            continue
        os_id = spot.state.meta.get("os_legal", {}).get(dec.action.value)
        key = spot.state.info_state_key
        if os_id is not None and key is not None:
            choices[key] = int(os_id)
    return choices


def mixed_distribution(variant: GameVariant, spots, sampler: SamplingClient,
                       samples: int, temperature: float) -> dict[str, dict[int, float]]:
    """Estimate each info state's action distribution from sampled completions."""
    prompts = [render_state(s.state) for s in spots]
    batches = sampler.sample_many(prompts, n=samples, temperature=temperature)
    dist_by_key: dict[str, dict[int, float]] = {}
    for spot, comps in zip(spots, batches):
        os_legal = spot.state.meta.get("os_legal", {})
        counts: dict[int, float] = {}
        for c in comps:
            try:
                dec = parse_decision(c.text, spot.state.legal_actions)
            except ParseError:
                continue
            os_id = os_legal.get(dec.action.value)
            if os_id is not None:
                counts[int(os_id)] = counts.get(int(os_id), 0.0) + 1.0
        key = spot.state.info_state_key
        if counts and key is not None:
            dist_by_key[key] = counts
    return dist_by_key


def mixed_exploitability(variant: GameVariant, all_spots, sampler: SamplingClient,
                         samples: int, temperature: float) -> float:
    dist = mixed_distribution(variant, all_spots, sampler, samples, temperature)
    return float(exploitability_of_distribution(variant, dist))


def evaluate_policy(variant: GameVariant, all_spots, eval_spots, sampler: SamplingClient,
                    mixed_samples: int = 16, mixed_temperature: float = 1.0) -> EvalReport:
    choices = greedy_choices(variant, all_spots, sampler)
    expl = exploitability_of_choices(variant, choices)
    mixed_expl = mixed_exploitability(variant, all_spots, sampler, mixed_samples, mixed_temperature)

    prompts = [render_state(s.state) for s in eval_spots]
    batches = sampler.sample_many(prompts, n=1, temperature=0.0)
    nash_sum = 0.0
    exact = 0
    for spot, comps in zip(eval_spots, batches):
        r, _ = spot_reward(spot, comps[0].text) if comps else (0.0, None)
        nash_sum += r
        if spot.nash_action_probs and r == max(spot.nash_action_probs.values()):
            exact += 1
    n = len(eval_spots)
    return EvalReport(
        exploitability=float(expl),
        mixed_exploitability=float(mixed_expl),
        mean_nash_prob=nash_sum / n if n else 0.0,
        exact_match=exact / n if n else 0.0,
        n=n,
    )
