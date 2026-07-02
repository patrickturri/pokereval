# src/pokereval/rl/selfplay/train.py
from __future__ import annotations
import random
from dataclasses import dataclass, field
from ...interface.types import GameVariant
from ...engine.kuhn_leduc import load_game
from ..config import RLConfig
from ..advantage import group_advantages
from ..datum import build_datum
from ..evaluate import evaluate_policy, EvalReport
from .game import sample_deal
from .rollout import play_group


@dataclass
class EvalPoint:
    """A single periodic-eval sample: the 0-indexed step just completed and the
    exploitability report of the learner's policy at that point."""
    step: int
    report: EvalReport


@dataclass
class SelfPlayResult:
    before: EvalReport | None = None
    after: EvalReport | None = None
    step_returns: list[float] = field(default_factory=list)
    # Learner exploitability sampled every ``eval_every`` steps — the
    # "exploitability vs CFR across refreshes" trajectory the design spec calls
    # the defensible claim. Empty offline curve is expected when eval is
    # unavailable (no OpenSpiel) or disabled (``eval_every <= 0``).
    eval_curve: list[EvalPoint] = field(default_factory=list)


def _maybe_eval(config: RLConfig, all_spots, eval_spots, sampler) -> EvalReport | None:
    try:
        import open_spiel  # noqa: F401
    except Exception:
        return None
    return evaluate_policy(
        GameVariant(config.variant), all_spots, eval_spots, sampler,
        mixed_samples=config.eval_samples, mixed_temperature=config.temperature,
    )


def selfplay_train(config: RLConfig, all_spots, eval_spots, backend, logger=None) -> SelfPlayResult:
    """Self-play GRPO: learner vs a frozen-snapshot opponent, reward = chips at
    showdown, advantages deal-conditioned (group = hands from one fixed deal)."""
    variant = GameVariant(config.variant)
    game = load_game(variant)
    rng = random.Random(config.seed)
    result = SelfPlayResult()

    result.before = _maybe_eval(config, all_spots, eval_spots, backend.sampler())
    opponent = backend.snapshot()  # initial opponent = base weights

    for step in range(config.num_steps):
        if step > 0 and step % config.opponent_refresh_every == 0:
            opponent = backend.snapshot()
        learner = backend.sampler()

        data: list = []
        return_sum = 0.0
        n_returns = 0
        for d in range(config.deals_per_step):
            deal = sample_deal(game, rng)
            learner_seat = d % 2  # alternate positions
            results = play_group(
                game, variant, deal, learner, opponent, learner_seat,
                backend.encode, config.temperature, config.hands_per_deal,
            )
            returns = [hr.learner_return for hr in results]
            advs = group_advantages(returns)  # deal-conditioned baseline
            for hr, adv in zip(results, advs):
                for dec in hr.decisions:
                    data.append(build_datum(
                        dec.prompt_tokens, dec.completion.tokens,
                        dec.completion.logprobs, adv,
                    ))
            return_sum += sum(returns)
            n_returns += len(returns)

        backend.train_step(data, config.learning_rate)
        mean_return = return_sum / n_returns if n_returns else 0.0
        result.step_returns.append(mean_return)

        # Periodic exploitability eval of the freshly-updated learner (spec step 7).
        report = None
        if config.eval_every > 0 and (step + 1) % config.eval_every == 0:
            report = _maybe_eval(config, all_spots, eval_spots, backend.sampler())
            if report is not None:
                result.eval_curve.append(EvalPoint(step=step, report=report))

        if logger:
            entry = {"step": step, "mean_return": mean_return, "n_data": len(data)}
            if report is not None:
                entry["mixed_exploitability"] = report.mixed_exploitability
                entry["exploitability"] = report.exploitability
            logger(entry)

    result.after = _maybe_eval(config, all_spots, eval_spots, backend.sampler())
    return result
