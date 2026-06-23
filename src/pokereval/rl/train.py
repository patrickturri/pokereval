from __future__ import annotations
from dataclasses import dataclass, field
from pokereval.interface.types import GameVariant
from pokereval.interface.prompts import render_state
from pokereval.rl.config import RLConfig
from pokereval.rl.backend import Backend
from pokereval.rl.reward import spot_reward
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
    return evaluate_policy(GameVariant(config.variant), all_spots, eval_spots, sampler)


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
        step_reward_sum = 0.0
        step_reward_count = 0
        for spot in batch:
            prompt = render_state(spot.state)
            comps = sampler.sample(prompt, config.group_size, config.temperature)
            rewards = [spot_reward(spot, c.text)[0] for c in comps]
            advs = group_advantages(rewards)
            prompt_tokens = backend.encode(prompt)
            for comp, adv in zip(comps, advs):
                data.append(build_datum(prompt_tokens, comp.tokens, comp.logprobs, adv))
            step_reward_sum += sum(rewards)
            step_reward_count += len(rewards)

        backend.train_step(data, config.learning_rate)
        mean_reward = step_reward_sum / step_reward_count if step_reward_count else 0.0
        result.step_rewards.append(mean_reward)
        if logger:
            logger({"step": step, "mean_reward": mean_reward, "n_data": len(data)})

    result.after = _maybe_eval(config, all_spots, eval_spots, backend.sampler())
    return result
