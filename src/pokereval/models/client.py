from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    name: str

    def complete(self, prompt: str, system: str | None = None) -> str: ...


class FakeClient:
    """Deterministic client for tests and dry runs."""

    def __init__(self, name: str, response: str | Callable[[str], str] = "ACTION: fold"):
        self.name = name
        self._response = response

    def complete(self, prompt: str, system: str | None = None) -> str:
        if callable(self._response):
            return self._response(prompt)
        return self._response


class AnthropicClient:
    """Anthropic Messages API client (e.g. claude-opus-4-8).

    Uses adaptive thinking — poker decisions are genuine multi-step reasoning,
    and the current Opus/Fable family only supports adaptive thinking (a fixed
    ``budget_tokens`` is rejected). Thinking blocks are returned with empty text
    by default and are filtered out below; the parseable ``ACTION:`` line lives
    in the final ``text`` block. ``max_tokens`` has headroom so thinking does not
    crowd out the visible answer.
    """

    def __init__(self, name: str, model: str, max_tokens: int = 4096):
        from anthropic import Anthropic  # lazy import; optional dep

        self.name = name
        self.model = model
        self.max_tokens = max_tokens
        self._client = Anthropic()

    def complete(self, prompt: str, system: str | None = None) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            thinking={"type": "adaptive"},
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")


class OpenAIClient:
    """⚠ Confirm model IDs and SDK call against live docs before use.
    Also serves OpenAI-compatible endpoints (vLLM) via base_url."""

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
        resp = self._client.chat.completions.create(model=self.model, messages=messages)
        return resp.choices[0].message.content or ""
