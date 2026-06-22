"""GeminiClient uses Google's OpenAI-compatible endpoint (construction only)."""
import pytest

from pokereval.models.client import GeminiClient, OpenAIClient


def test_gemini_client_targets_google_openai_endpoint(monkeypatch):
    monkeypatch.setenv("GOOGLE_VERTEX_API_KEY", "test-key-123")
    c = GeminiClient("gemini-3.1-pro-preview", "gemini-3.1-pro-preview")
    assert c.name == "gemini-3.1-pro-preview"
    assert isinstance(c, OpenAIClient)
    # The underlying OpenAI client points at Gemini's compat base URL.
    assert "generativelanguage.googleapis.com" in str(c._client.base_url)


def test_gemini_client_reads_key_from_env(monkeypatch):
    monkeypatch.delenv("GOOGLE_VERTEX_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "alt-key")
    c = GeminiClient("g", "gemini-3.1-pro-preview")
    assert c.name == "g"
