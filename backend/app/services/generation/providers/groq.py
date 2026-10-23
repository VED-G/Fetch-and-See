"""
DocMind — Groq LLM Provider (Primary)
=======================================
Uses the official Groq Python SDK for chat completions.

Why Groq?
  Groq runs LLMs on custom Language Processing Units (LPUs) — dedicated
  silicon optimised for token generation.  Result: 300-500 tokens/sec vs.
  10-30 tokens/sec on a laptop CPU.  Free tier is generous (14k RPM on
  llama-3.1-8b-instant).  Perfect for demos and development.

Streaming:
  We use Groq's native streaming API.  Each chunk arrives as an SSE event
  from Groq, and we forward individual tokens to the client as another SSE
  stream — so the user sees tokens appear in real time.

Interview talking point:
  "Groq's LPU is a dataflow architecture — it avoids the memory bandwidth
   bottleneck that limits GPU throughput for autoregressive inference.
   Every token generation step in an LLM reads the KV-cache from memory;
   Groq's architecture keeps the KV-cache on-chip.  That's why it's 10-20×
   faster than a GPU for generation (not training)."
"""

from __future__ import annotations

from typing import AsyncIterator

import structlog
from groq import AsyncGroq

from app.config import get_settings
from app.core.exceptions import LLMProviderError

log = structlog.get_logger(__name__)
settings = get_settings()


class GroqProvider:
    """Async Groq chat completion provider."""

    def __init__(self) -> None:
        if not settings.GROQ_API_KEY:
            raise LLMProviderError(
                "GROQ_API_KEY is not set. Get a free key at console.groq.com"
            )
        self._client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        self._model = settings.GROQ_MODEL

    async def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        """Non-streaming chat completion."""
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
            )
            content = response.choices[0].message.content or ""
            log.debug(
                "groq.complete",
                tokens_in=response.usage.prompt_tokens if response.usage else 0,
                tokens_out=response.usage.completion_tokens if response.usage else 0,
            )
            return content
        except Exception as exc:
            raise LLMProviderError(f"Groq API error: {exc}") from exc

    async def stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> AsyncIterator[str]:
        """Streaming chat completion — yields token strings."""
        try:
            async with self._client.chat.completions.with_streaming_response.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            ) as response:
                async for chunk in response.iter_lines():
                    if chunk.startswith("data: ") and chunk != "data: [DONE]":
                        import json
                        try:
                            data = json.loads(chunk[6:])
                            token = data["choices"][0]["delta"].get("content", "")
                            if token:
                                yield token
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
        except Exception as exc:
            raise LLMProviderError(f"Groq streaming error: {exc}") from exc
