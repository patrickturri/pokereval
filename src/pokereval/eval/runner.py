"""Eval runner — render → complete → parse → grade, one row per spot.

A parse failure (no legal ACTION line) is recorded as an ``error`` and the spot
is left ungraded rather than silently scored zero, so the leaderboard can report
format-failure rates separately from poker-skill scores.
"""

from pydantic import BaseModel

from ..graders.judge import JudgeGrader
from ..graders.verifiable import VerifiableGrader
from ..interface.prompts import ParseError, parse_decision, render_state
from ..interface.types import Score
from ..models.client import LLMClient
from .spots import SpotSet


class SpotResult(BaseModel):
    info_state_key: str | None = None
    variant: str
    action: str | None = None
    verifiable: Score | None = None
    judge: Score | None = None
    error: str | None = None


def run_eval(
    client: LLMClient,
    spotset: SpotSet,
    verifiable: VerifiableGrader,
    judge: JudgeGrader | None = None,
) -> list[SpotResult]:
    results: list[SpotResult] = []
    for state in spotset.spots:
        prompt = render_state(state)
        text = client.complete(prompt)
        try:
            decision = parse_decision(text, state.legal_actions)
        except ParseError as exc:
            results.append(
                SpotResult(
                    info_state_key=state.info_state_key,
                    variant=state.variant.value,
                    error=str(exc),
                )
            )
            continue
        results.append(
            SpotResult(
                info_state_key=state.info_state_key,
                variant=state.variant.value,
                action=decision.action.value,
                verifiable=verifiable.grade(state, decision),
                judge=judge.grade(state, decision) if judge else None,
            )
        )
    return results
