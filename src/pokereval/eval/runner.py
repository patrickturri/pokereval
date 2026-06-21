from pydantic import BaseModel
from ..interface.types import Score
from ..interface.prompts import render_state, parse_decision, ParseError
from ..models.client import LLMClient
from ..graders.verifiable import VerifiableGrader
from ..graders.judge import JudgeGrader
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
            results.append(SpotResult(
                info_state_key=state.info_state_key,
                variant=state.variant.value, error=str(exc),
            ))
            continue
        results.append(SpotResult(
            info_state_key=state.info_state_key,
            variant=state.variant.value,
            action=decision.action.value,
            verifiable=verifiable.grade(state, decision),
            judge=judge.grade(state, decision) if judge else None,
        ))
    return results
