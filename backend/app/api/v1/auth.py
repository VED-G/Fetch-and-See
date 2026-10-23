"""DocMind — Auth router (API key management)."""
from __future__ import annotations
import secrets
import hashlib
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps import get_db, require_admin
from app.core.exceptions import DocumentNotFound
from app.models.db import APIKey
from app.models.schemas import CreateAPIKeyRequest, APIKeyResponse, APIKeyInfo

router = APIRouter()


def _generate_key() -> str:
    """Generate a high-entropy API key with a recognisable prefix."""
    token = secrets.token_urlsafe(32)
    return f"dm_{token}"


def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


@router.post("/auth/keys", response_model=APIKeyResponse, status_code=201)
async def create_api_key(
    body: CreateAPIKeyRequest,
    db: AsyncSession = Depends(get_db),
    _admin: APIKey = Depends(require_admin),
) -> APIKeyResponse:
    """Create a new API key. Raw key shown ONCE — store it safely."""
    raw_key = _generate_key()
    prefix = raw_key[:7]           # "dm_XXXX"
    key_hash = _hash_key(raw_key)

    api_key = APIKey(
        name=body.name,
        key_hash=key_hash,
        key_prefix=prefix,
        is_admin=body.is_admin,
    )
    db.add(api_key)
    await db.commit()
    await db.refresh(api_key)

    return APIKeyResponse(
        id=api_key.id,
        name=api_key.name,
        api_key=raw_key,
        key_prefix=prefix,
        is_admin=api_key.is_admin,
        created_at=api_key.created_at,
    )


@router.get("/auth/keys", response_model=list[APIKeyInfo])
async def list_api_keys(
    db: AsyncSession = Depends(get_db),
    _admin: APIKey = Depends(require_admin),
) -> list[APIKeyInfo]:
    """List all API keys (no raw keys returned)."""
    result = await db.execute(select(APIKey).order_by(APIKey.created_at.desc()))
    keys = result.scalars().all()
    return [
        APIKeyInfo(
            id=k.id, name=k.name, key_prefix=k.key_prefix,
            is_admin=k.is_admin, is_active=k.is_active,
            last_used_at=k.last_used_at, created_at=k.created_at,
        )
        for k in keys
    ]


@router.delete("/auth/keys/{key_id}", status_code=204)
async def revoke_api_key(
    key_id: str,
    db: AsyncSession = Depends(get_db),
    _admin: APIKey = Depends(require_admin),
) -> None:
    """Revoke (soft-delete) an API key."""
    result = await db.execute(select(APIKey).where(APIKey.id == key_id))
    key = result.scalar_one_or_none()
    if not key:
        raise DocumentNotFound(f"API key {key_id} not found.")
    key.is_active = False
    await db.commit()
