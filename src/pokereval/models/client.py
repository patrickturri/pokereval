"""LLM client abstraction.

Everything downstream depends only on the tiny ``LLMClient`` protocol, so the
whole pipeline runs offline against ``FakeClient``. The provider wrappers lazily
import their SDKs and are never exercised in the test suite (no network, no keys).
"""

from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Anything with a ``name`` and a ``complete(prompt, system=None) -> str``."""

    name: str

    def complete(self, prompt: str, system: str | None = None) -> str: ...


class FakeClient:
    """Deterministic client for tests and credential-free dry runs."""

    def __init__(
        self,
        name: str,
        response: str | Callable[[str], str] = "ACTION: fold",
    ):
        self.name = name
        self._response = response

    def complete(self, prompt: str, system: str | None = None) -> str:
        if callable(self._response):
            return self._response(prompt)
        return self._response


class AnthropicClient:
    """Anthropic API wrapper.

    ⚠ Confirm current model IDs and the SDK call shape via the ``claude-api``
    skill before relying on this in a live run. Model IDs are constructor args,
    never hardcoded elsewhere.
    """

    def __init__(self, name: str, model: str, max_tokens: int = 1024):
        from anthropic import Anthropic  # lazy import; optional dep

        self.name = name
        self.model = model
        self.max_tokens = max_tokens
        self._client = Anthropic()

    def complete(self, prompt: str, system: str | None = None) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")


class OpenAIClient:
    """OpenAI (or any OpenAI-compatible endpoint, e.g. vLLM via ``base_url``).

    ⚠ Confirm current model IDs and the SDK call shape against live docs before
    relying on this in a live run.
    """

    def __init__(self, name: str, model: str, base_url: str | None = None):
        from openai import OpenAI  # lazy import; optional dep

        self.name = name
        self.model = model
        self._client = OpenAI(base_url=base_url) if base_url else OpenAI()

    def complete(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages
        )
        return resp.choices[0].message.content or ""
