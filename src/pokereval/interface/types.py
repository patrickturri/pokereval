"""Core typed interface shared by every component.

The whole project hangs off one contract: a poker ``State`` is rendered to a
prompt, the model emits a ``Decision``, and a grader turns ``(State, Decision)``
into a ``Score``. Keeping these tiny and Pydantic-validated lets the engine,
graders, eval runner, and (later) the RL loop all speak the same language.
"""

from enum import Enum

from pydantic import BaseModel, Field


class Action(str, Enum):
    """A poker action. Values are the lowercase names used on the wire."""

    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"


class GameVariant(str, Enum):
    """Supported game variants."""

    KUHN = "kuhn"
    LEDUC = "leduc"
    HOLDEM = "holdem"


class State(BaseModel):
    """A single decision point presented to a player.

    ``info_state_key`` ties a Kuhn/Leduc spot back to its OpenSpiel information
    state so the verifiable grader can look up the exact Nash policy. ``meta``
    carries variant-specific extras (``os_legal`` action mapping for the small
    games, ``reference_action`` for curated Hold'em spots).
    """

    variant: GameVariant
    hero_cards: list[str]
    board: list[str] = Field(default_factory=list)
    pot: float
    to_call: float = 0.0
    position: str = ""
    history: str = ""
    legal_actions: list[Action]
    info_state_key: str | None = None
    meta: dict = Field(default_factory=dict)


class Decision(BaseModel):
    """A parsed model decision: an action, optional sizing, and its reasoning."""

    action: Action
    size: float | None = None
    reasoning: str = ""


class Score(BaseModel):
    """A grader's verdict. ``value`` is the headline number; ``detail`` explains."""

    value: float
    detail: dict = Field(default_factory=dict)
