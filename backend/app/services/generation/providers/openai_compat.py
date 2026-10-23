"""
OpenAI-compatible LLM provider.

Works with OpenAI, Azure, Together.ai, or any OpenAI-spec endpoint.
Configure via OPENAI_API_KEY, OPENAI_BASE_URL, and OPENAI_MODEL in .env.
"""

from __future__ import annotations

from typing import AsyncIterator

import structlog
from openai import AsyncOpenAI

from app.config import get_settings
from app.core.exceptions import LLMProviderError

log = structlog.get_logger(__name__)
settings = get_settings()


class OpenAIProvider:
    def __init__(self) -> None:
        if not settings.OPENAI_API_KEY:
            raise LLMProviderError("OPENAI_API_KEY is not set.")
        self._client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
        )
        self._model = settings.OPENAI_MODEL

    async def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise LLMProviderError(f"OpenAI error: {exc}") from exc

    async def stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> AsyncIterator[str]:
        try:
            async with await self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            ) as stream:
                async for chunk in stream:
                    token = chunk.choices[0].delta.content or ""
                    if token:
                        yield token
        except Exception as exc:
            raise LLMProviderError(f"OpenAI streaming error: {exc}") from exc
