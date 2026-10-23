"""
Ollama LLM provider (local fallback).

Connects to a locally running Ollama server. Set LLM_PROVIDER=ollama in .env.
Requires Ollama installed and running: https://ollama.ai
"""

from __future__ import annotations

from typing import AsyncIterator

import httpx
import structlog

from app.config import get_settings
from app.core.exceptions import LLMProviderError

log = structlog.get_logger(__name__)
settings = get_settings()


class OllamaProvider:
    """Async Ollama provider using the /api/chat endpoint."""

    def __init__(self) -> None:
        self._base_url = settings.OLLAMA_BASE_URL
        self._model = settings.OLLAMA_MODEL

    async def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        """Non-streaming Ollama chat completion."""
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self._base_url}/api/chat",
                    json={
                        "model": self._model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "num_predict": max_tokens,
                            "temperature": temperature,
                        },
                    },
                )
                response.raise_for_status()
                data = response.json()
                return data["message"]["content"]
        except httpx.ConnectError as exc:
            raise LLMProviderError(
                f"Cannot connect to Ollama at {self._base_url}. "
                "Is Ollama running? Run: ollama serve"
            ) from exc
        except Exception as exc:
            raise LLMProviderError(f"Ollama error: {exc}") from exc

    async def stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> AsyncIterator[str]:
        """Streaming Ollama chat — yields token strings."""
        import json
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{self._base_url}/api/chat",
                    json={
                        "model": self._model,
                        "messages": messages,
                        "stream": True,
                        "options": {
                            "num_predict": max_tokens,
                            "temperature": temperature,
                        },
                    },
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                data = json.loads(line)
                                token = data.get("message", {}).get("content", "")
                                if token:
                                    yield token
                                if data.get("done"):
                                    break
                            except json.JSONDecodeError:
                                continue
        except httpx.ConnectError as exc:
            raise LLMProviderError(
                f"Cannot connect to Ollama at {self._base_url}."
            ) from exc
        except Exception as exc:
            raise LLMProviderError(f"Ollama streaming error: {exc}") from exc
