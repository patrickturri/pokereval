from enum import Enum
from pydantic import BaseModel, Field


class Action(str, Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"


class GameVariant(str, Enum):
    KUHN = "kuhn"
    LEDUC = "leduc"
    HOLDEM = "holdem"


class State(BaseModel):
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
    action: Action
    size: float | None = None
    reasoning: str = ""


class Score(BaseModel):
    value: float
    detail: dict = Field(default_factory=dict)
