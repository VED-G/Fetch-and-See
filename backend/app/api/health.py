"""
DocMind — /health Endpoint
============================
Returns the liveness and readiness status of every service dependency.
Used by Docker health checks and monitoring systems.

Difference between liveness and readiness:
  - Liveness:  Is the process running?  (always true if this endpoint responds)
  - Readiness: Are all dependencies reachable?  (Qdrant, Redis, LLM provider)
"""

from __future__ import annotations

import time

import httpx
import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.models.schemas import HealthResponse

log = structlog.get_logger(__name__)
router = APIRouter()
settings = get_settings()

APP_VERSION = "0.1.0"


async def _check_qdrant() -> str:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            if settings.QDRANT_MODE == "server":
                r = await client.get(f"{settings.QDRANT_URL}/healthz")
                return "ok" if r.status_code == 200 else "degraded"
            # Local file mode — if we got here, it's working
            return "ok (local)"
    except Exception:
        return "unreachable"


async def _check_redis() -> str:
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(settings.REDIS_URL, socket_timeout=2.0)
        await r.ping()
        await r.aclose()
        return "ok"
    except Exception:
        return "unreachable"


async def _check_llm_provider() -> str:
    try:
        if settings.LLM_PROVIDER == "groq":
            # Groq: just check the API key is set and the endpoint responds
            if not settings.GROQ_API_KEY:
                return "no api key"
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {settings.GROQ_API_KEY}"},
                )
                return "ok" if r.status_code == 200 else f"http_{r.status_code}"
        elif settings.LLM_PROVIDER == "ollama":
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
                return "ok" if r.status_code == 200 else "degraded"
        return "unknown_provider"
    except Exception:
        return "unreachable"


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> JSONResponse:
    """Liveness + readiness probe.

    Returns 200 if all services are healthy, 503 if any are degraded.
    """
    from datetime import datetime, timezone
    start = time.monotonic()

    qdrant_status = await _check_qdrant()
    redis_status = await _check_redis()
    llm_status = await _check_llm_provider()

    services = {
        "qdrant": qdrant_status,
        "redis": redis_status,
        settings.LLM_PROVIDER: llm_status,
        "reranker": "enabled" if settings.ENABLE_RERANKER else "disabled",
    }

    all_ok = all(
        s.startswith("ok") for s in [qdrant_status, redis_status, llm_status]
    )

    response_data = {
        "status": "ok" if all_ok else "degraded",
        "version": APP_VERSION,
        "environment": settings.APP_ENV,
        "services": services,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    log.info(
        "health.checked",
        status=response_data["status"],
        duration_ms=int((time.monotonic() - start) * 1000),
    )

    return JSONResponse(
        content=response_data,
        status_code=200 if all_ok else 503,
    )
