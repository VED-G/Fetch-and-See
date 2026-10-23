"""
LLM provider Protocol and factory.

All providers implement complete() and stream(). Switch providers via the
LLM_PROVIDER env var (groq | ollama | openai) without changing pipeline code.
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
