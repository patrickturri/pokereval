from __future__ import annotations
from dataclasses import dataclass, field
from pokereval.interface.types import GameVariant
from pokereval.interface.prompts import render_state
from pokereval.rl.config import RLConfig
from pokereval.rl.backend import Backend
from pokereval.rl.reward import spot_reward, kl_to_nash_reward
from pokereval.rl.advantage import group_advantages
from pokereval.rl.datum import build_datum
from pokereval.rl.evaluate import evaluate_policy, EvalReport


@dataclass
class TrainResult:
    before: EvalReport | None = None
    after: EvalReport | None = None
    step_rewards: list[float] = field(default_factory=list)


def _maybe_eval(config, all_spots, eval_spots, sampler) -> EvalReport | None:
    try:
        import open_spiel  # noqa: F401
    except Exception:
        return None
    return evaluate_policy(
        GameVariant(config.variant), all_spots, eval_spots, sampler,
        mixed_samples=config.eval_samples, mixed_temperature=config.temperature,
    )


def _training_reward(config, spot, comp) -> float:
    """Per-completion training reward selected by config.reward_mode."""
    if config.reward_mode == "kl_nash":
        return kl_to_nash_reward(spot, comp.text, action_logprob=sum(comp.logprobs))
    return spot_reward(spot, comp.text)[0]  # "nash_prob" (legacy / ablation)


def train(config: RLConfig, train_spots, eval_spots, backend: Backend, logger=None) -> TrainResult:
    all_spots = list(train_spots) + list(eval_spots)
    result = TrainResult()
    sampler = backend.sampler()
    result.before = _maybe_eval(config, all_spots, eval_spots, sampler)

    cursor = 0
    for step in range(config.num_steps):
        if step % config.refresh_every == 0:
            sampler = backend.sampler()
        batch = [train_spots[(cursor + i) % len(train_spots)] for i in range(config.batch_spots)]
        cursor += config.batch_spots

        data = []
        nash_sum = 0.0
        count = 0
        prompts = [render_state(spot.state) for spot in batch]
        batch_comps = sampler.sample_many(prompts, config.group_size, config.temperature)
        for spot, prompt, comps in zip(batch, prompts, batch_comps):
            train_rewards = [_training_reward(config, spot, c) for c in comps]
            advs = group_advantages(train_rewards)
            prompt_tokens = backend.encode(prompt)
            for comp, adv in zip(comps, advs):
                data.append(build_datum(prompt_tokens, comp.tokens, comp.logprobs, adv))
            # Track Nash-prob (interpretable) for logging, separate from train reward.
            nash_sum += sum(spot_reward(spot, c.text)[0] for c in comps)
            count += len(comps)

        backend.train_step(data, config.learning_rate)
        mean_nash = nash_sum / count if count else 0.0
        result.step_rewards.append(mean_nash)
        if logger:
            logger({"step": step, "mean_nash_prob": mean_nash, "n_data": len(data)})

    result.after = _maybe_eval(config, all_spots, eval_spots, backend.sampler())
    return result
