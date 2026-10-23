"""
Bootstrap script: create an admin API key for first-time setup.

Usage:
    cd backend
    python scripts/create_api_key.py --name "my-key"

The script prints the raw key ONCE. Store it — it cannot be recovered
(only the bcrypt hash is saved to the DB).
"""

import argparse
import asyncio
import hashlib
import os
import secrets
import sys
from pathlib import Path

# Allow running from the backend/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))


async def create_key(name: str, is_admin: bool) -> str:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
    from app.config import get_settings
    from app.models.db import APIKey, Base

    settings = get_settings()
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)

    raw_key = f"dm_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(
        (settings.SECRET_KEY + raw_key).encode()
    ).hexdigest()

    async with factory() as db:
        api_key = APIKey(
            name=name,
            key_hash=key_hash,
            key_prefix=raw_key[:8],
            is_admin=is_admin,
        )
        db.add(api_key)
        await db.commit()
        await db.refresh(api_key)

    await engine.dispose()
    return raw_key


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Fetch and See API key")
    parser.add_argument("--name", default="default", help="Label for this key")
    parser.add_argument("--admin", action="store_true", help="Grant admin privileges")
    args = parser.parse_args()

    key = asyncio.run(create_key(args.name, args.admin))
    print()
    print(f"  API key created: {key}")
    print()
    print("  Add to your .env:")
    print(f"    # or pass as Authorization: Bearer {key}")
    print()
    print("  This key will NOT be shown again.")
    print()


if __name__ == "__main__":
    main()
