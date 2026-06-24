# src/pokereval/rl/selfplay/rollout.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
from ...interface.types import GameVariant, Action
from ...interface.prompts import render_state, parse_decision, ParseError
from ...engine.kuhn_leduc import node_to_state
from .game import Deal, play_from_deal


@dataclass
class Decision:
    prompt_tokens: list[int]
    completion: object  # rl.sampler.Completion


@dataclass
class HandResult:
    learner_return: float
    decisions: list = field(default_factory=list)


def _legal_fallback(state, os_legal) -> str:
    """Safe legal action for an illegal/unparseable completion: fold if legal
    (folding usually loses → negative advantage teaches legality), else the
    first legal action."""
    if Action.FOLD.value in os_legal:
        return Action.FOLD.value
    return state.legal_actions[0].value


def play_hand(game, variant: GameVariant, deal: Deal, learner_sampler,
              opponent_sampler, learner_seat: int,
              encode: Callable[[str], list[int]], temperature: float) -> HandResult:
    decisions: list[Decision] = []

    def choose(os_state) -> int:
        state, os_legal = node_to_state(variant, os_state)
        prompt = render_state(state)
        is_learner = os_state.current_player() == learner_seat
        sampler = learner_sampler if is_learner else opponent_sampler
        comp = sampler.sample(prompt, n=1, temperature=temperature)[0]
        try:
            action_value = parse_decision(comp.text, state.legal_actions).action.value
        except ParseError:
            action_value = _legal_fallback(state, os_legal)
        if is_learner:
            # record the learner's actual completion (even when illegal) so its
            # advantage flows back as a gradient signal
            decisions.append(Decision(prompt_tokens=encode(prompt), completion=comp))
        return os_legal[action_value]

    terminal = play_from_deal(game, deal, choose)
    return HandResult(learner_return=float(terminal.returns()[learner_seat]), decisions=decisions)


def play_group(game, variant: GameVariant, deal: Deal, learner_sampler,
               opponent_sampler, learner_seat: int,
               encode: Callable[[str], list[int]], temperature: float,
               hands: int) -> list[HandResult]:
    """Play ``hands`` independent hands from the SAME fixed deal (the GRPO group)."""
    return [
        play_hand(game, variant, deal, learner_sampler, opponent_sampler,
                  learner_seat, encode, temperature)
        for _ in range(hands)
    ]
