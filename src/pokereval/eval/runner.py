from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel
from ..interface.types import State, Score
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


def _eval_one(
    state: State,
    client: LLMClient,
    verifiable: VerifiableGrader,
    judge: JudgeGrader | None,
) -> SpotResult:
    prompt = render_state(state)
    try:
        text = client.complete(prompt)
    except Exception as exc:  # noqa: BLE001 — keep one failure from tanking the run
        return SpotResult(
            info_state_key=state.info_state_key,
            variant=state.variant.value,
            error=f"{type(exc).__name__}: {exc}",
        )
    try:
        decision = parse_decision(text, state.legal_actions)
    except ParseError as exc:
        return SpotResult(
            info_state_key=state.info_state_key,
            variant=state.variant.value,
            error=str(exc),
        )
    return SpotResult(
        info_state_key=state.info_state_key,
        variant=state.variant.value,
        action=decision.action.value,
        verifiable=verifiable.grade(state, decision),
        judge=judge.grade(state, decision) if judge else None,
    )


def run_eval(
    client: LLMClient,
    spotset: SpotSet,
    verifiable: VerifiableGrader,
    judge: JudgeGrader | None = None,
    max_workers: int = 1,
) -> list[SpotResult]:
    """Evaluate every spot: render -> complete -> parse -> grade.

    `max_workers > 1` runs the (I/O-bound) model calls concurrently via a
    thread pool. Result order always matches `spotset.spots` order, so callers
    can pair results with spots positionally.
    """
    def work(state: State) -> SpotResult:
        return _eval_one(state, client, verifiable, judge)

    if max_workers <= 1:
        return [work(state) for state in spotset.spots]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        # executor.map preserves input order.
        return list(pool.map(work, spotset.spots))
