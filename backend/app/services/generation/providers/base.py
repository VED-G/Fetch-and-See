"""
DocMind — Abstract LLM Provider Protocol
==========================================
All LLM providers implement this interface.
Switching from Groq to Ollama to OpenAI is one env var change.

Interview talking point:
  "I used Python's Protocol (structural subtyping) instead of ABC because
   Protocol doesn't require explicit inheritance — any object with the right
   methods satisfies it.  This means I can mock the provider in tests with
   a plain class, no inheritance boilerplate."
"""

from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Structural interface for all LLM providers."""

    async def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        """Non-streaming completion.  Returns the full response string."""
        ...

    async def stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> AsyncIterator[str]:
        """Streaming completion.  Yields tokens as they arrive."""
        ...


def get_provider() -> LLMProvider:
    """Factory: return the configured LLM provider instance."""
    from app.config import get_settings
    settings = get_settings()

    if settings.LLM_PROVIDER == "groq":
        from app.services.generation.providers.groq import GroqProvider
        return GroqProvider()
    elif settings.LLM_PROVIDER == "ollama":
        from app.services.generation.providers.ollama import OllamaProvider
        return OllamaProvider()
    elif settings.LLM_PROVIDER == "openai":
        from app.services.generation.providers.openai_compat import OpenAIProvider
        return OpenAIProvider()
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {settings.LLM_PROVIDER}")
