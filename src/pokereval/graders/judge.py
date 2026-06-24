"""LLM-judge grader — the *unverifiable* reward signal.

Scores reasoning quality against a fixed rubric. Studying how this soft signal
agrees (or doesn't) with the exact verifiable grader is itself a deliverable:
the "grading verifiable vs. unverifiable domains" skill.
"""

import json
import re

from ..interface.prompts import render_state
from ..interface.types import Decision, Score, State
from ..models.client import LLMClient

RUBRIC = [
    "identifies_pot_odds_or_equity",
    "reasons_about_opponent_range",
    "sound_bet_sizing_logic",
    "decision_consistent_with_reasoning",
]

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class JudgeGrader:
    def __init__(self, client: LLMClient, rubric: list[str] = RUBRIC):
        self.client = client
        self.rubric = rubric

    def _prompt(self, state: State, decision: Decision) -> str:
        criteria = "\n".join(f"- {c}" for c in self.rubric)
        return (
            "You are grading a poker player's reasoning. The situation:\n"
            f"{render_state(state)}\n\n"
            f"The player chose: {decision.action.value}"
            f"{' ' + str(decision.size) if decision.size else ''}\n"
            f"Their reasoning:\n{decision.reasoning or '(none given)'}\n\n"
            "Score each criterion 1 (met) or 0 (not met). "
            "Respond with ONLY a JSON object mapping each criterion to 0 or 1:\n"
            f"{criteria}"
        )

    def grade(self, state: State, decision: Decision) -> Score:
        text = self.client.complete(self._prompt(state, decision))
        match = _JSON_RE.search(text)
        if not match:
            return Score(value=0.0, detail={"kind": "judge_error", "raw": text})
        try:
            scores = json.loads(match.group(0))
            values = [float(scores[c]) for c in self.rubric]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return Score(value=0.0, detail={"kind": "judge_error", "raw": text})
        value = sum(values) / len(values)
        return Score(
            value=value, detail={"kind": "judge", "scores": scores, "raw": text}
        )
