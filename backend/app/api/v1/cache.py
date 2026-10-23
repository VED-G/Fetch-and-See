"""Fetch and See — cache management endpoints."""
from fastapi import APIRouter, Depends
from app.api.deps import require_admin
from app.models.db import APIKey
from app.models.schemas import CacheStatsResponse

router = APIRouter()


@router.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats(_key: APIKey = Depends(require_admin)) -> CacheStatsResponse:
    """Return semantic cache statistics. (Stub — wired in Phase 5)"""
    return CacheStatsResponse(
        hit_rate_pct=0.0, total_requests=0,
        cache_hits=0, cache_misses=0, entries=0, size_mb=0.0,
    )


@router.delete("/cache", status_code=204)
async def flush_cache(_key: APIKey = Depends(require_admin)) -> None:
    """Flush the semantic cache. (Stub — wired in Phase 5)"""
    pass
