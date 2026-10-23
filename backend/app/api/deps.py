"""
Shared FastAPI dependencies (injectable per-request).

Covers: async DB session, API key auth (Bearer token), admin guard,
and sliding-window rate limiting via Redis sorted sets.
"""

from __future__ import annotations

import hashlib
import time

import structlog
from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.exceptions import AuthenticationError, RateLimitExceeded
from app.models.db import APIKey

log = structlog.get_logger(__name__)
settings = get_settings()


# ─── Database session ─────────────────────────────────────────────────────────

async def get_db(request: Request) -> AsyncSession:  # type: ignore[misc]
    """Yield a new AsyncSession for the current request, then close it."""
    async with request.app.state.db_session_factory() as session:
        yield session


# ─── Auth ─────────────────────────────────────────────────────────────────────

def _hash_key(raw_key: str) -> str:
    """SHA-256 hash of the raw API key (not bcrypt — bcrypt is too slow for per-request use).

    We use SHA-256 for the lookup hash because:
    - bcrypt is intentionally slow (100ms+ per check) → unacceptable per request
    - SHA-256 is fast and sufficient when the key itself is high-entropy (32+ bytes)
    - bcrypt IS used at key creation time to generate key_hash in the DB
    """
    return hashlib.sha256(raw_key.encode()).hexdigest()


async def require_auth(
    authorization: str | None = Header(default=None),
    db: AsyncSession = Depends(get_db),
) -> APIKey:
    """Validate Bearer token and return the APIKey row."""
    if not authorization or not authorization.startswith("Bearer "):
        raise AuthenticationError("Missing or malformed Authorization header.")

    raw_key = authorization.removeprefix("Bearer ").strip()
    key_hash = _hash_key(raw_key)

    result = await db.execute(
        select(APIKey).where(APIKey.key_hash == key_hash, APIKey.is_active == True)  # noqa: E712
    )
    api_key = result.scalar_one_or_none()

    if not api_key:
        raise AuthenticationError("Invalid or revoked API key.")

    # Update last_used_at (fire and forget — don't fail the request if this fails)
    try:
        from datetime import datetime, timezone
        api_key.last_used_at = datetime.now(timezone.utc)
        await db.commit()
    except Exception:
        pass

    log.debug("auth.success", key_prefix=api_key.key_prefix)
    return api_key


async def require_admin(api_key: APIKey = Depends(require_auth)) -> APIKey:
    """Require the API key to have admin privileges."""
    from app.core.exceptions import PermissionDenied
    if not api_key.is_admin:
        raise PermissionDenied("Admin privileges required for this endpoint.")
    return api_key


# ─── Rate Limiting ────────────────────────────────────────────────────────────

async def check_rate_limit(request: Request, api_key: APIKey = Depends(require_auth)) -> None:
    """Token-bucket rate limiting using Redis.

    Each API key gets its own bucket refilled at RATE_LIMIT_REQUESTS per
    RATE_LIMIT_WINDOW_SECONDS.

    Why token-bucket over fixed-window?
      Fixed window allows a 2× burst at window boundaries (last second of
      window N + first second of window N+1).  Token bucket smooths this out.

    Implementation:
      We use a Redis sorted set where each token is a timestamped entry.
      Tokens older than the window are removed before counting.
      This is the 'sliding window log' algorithm — O(log n) per request.
    """
    try:
        import redis.asyncio as aioredis

        r = aioredis.from_url(settings.REDIS_URL)
        key = f"rate_limit:{api_key.id}"
        now = time.time()
        window_start = now - settings.RATE_LIMIT_WINDOW_SECONDS

        async with r.pipeline(transaction=True) as pipe:
            pipe.zremrangebyscore(key, 0, window_start)   # Remove old tokens
            pipe.zadd(key, {str(now): now})                # Add current request
            pipe.zcard(key)                                # Count tokens in window
            pipe.expire(key, settings.RATE_LIMIT_WINDOW_SECONDS + 1)
            results = await pipe.execute()

        request_count = results[2]
        await r.aclose()

        if request_count > settings.RATE_LIMIT_REQUESTS:
            retry_after = int(settings.RATE_LIMIT_WINDOW_SECONDS)
            log.warning("rate_limit.exceeded", key_prefix=api_key.key_prefix, count=request_count)
            raise RateLimitExceeded(
                f"Rate limit exceeded: {settings.RATE_LIMIT_REQUESTS} requests "
                f"per {settings.RATE_LIMIT_WINDOW_SECONDS}s.",
                retry_after=retry_after,
            )
    except RateLimitExceeded:
        raise
    except Exception as e:
        # Redis is down — fail open (allow the request) rather than blocking all traffic
        log.warning("rate_limit.redis_error", error=str(e))
