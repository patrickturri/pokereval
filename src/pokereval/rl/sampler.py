from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable


@dataclass
class Completion:
    text: str
    tokens: list[int]
    logprobs: list[float]


@runtime_checkable
class SamplingClient(Protocol):
    def sample(self, prompt: str, n: int, temperature: float) -> list[Completion]:
        ...


class MockSamplingClient:
    """Offline sampler. Emits 'ACTION: <action>' with deterministic dummy tokens."""

    def __init__(
        self,
        action_for: Callable[[str], str] | None = None,
        fixed_action: str = "check",
    ):
        self._action_for = action_for
        self._fixed = fixed_action

    def sample(self, prompt: str, n: int, temperature: float) -> list[Completion]:
        action = self._action_for(prompt) if self._action_for else self._fixed
        text = f"ACTION: {action}"
        # deterministic dummy token ids/logprobs (1 per character, content irrelevant offline)
        tokens = [ord(ch) % 256 for ch in text]
        logprobs = [-1.0] * len(tokens)
        return [Completion(text=text, tokens=list(tokens), logprobs=list(logprobs)) for _ in range(n)]
