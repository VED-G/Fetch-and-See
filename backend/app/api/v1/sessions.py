"""
Sessions router: retrieve conversation history and delete sessions.

Sessions are created automatically on the first query. Clients store the
session_id from the X-Session-ID response header and pass it back in
subsequent requests to maintain conversation memory.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, require_auth
from app.core.exceptions import DocMindError
from app.models.db import APIKey, Message, Session
from app.models.schemas import SessionHistoryResponse, SessionInfo

log = structlog.get_logger(__name__)
router = APIRouter()


@router.get(
    "/sessions/{session_id}",
    response_model=SessionHistoryResponse,
    summary="Get conversation history for a session",
)
async def get_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    _key: APIKey = Depends(require_auth),
) -> SessionHistoryResponse:
    """Return all messages in a session, oldest first."""
    result = await db.execute(select(Session).where(Session.id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        raise DocMindError(f"Session {session_id} not found.")

    msgs_result = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at)
    )
    messages = msgs_result.scalars().all()

    return SessionHistoryResponse(
        session_id=session_id,
        collection_id=session.collection_id,
        created_at=session.created_at,
        messages=[
            {"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()}
            for m in messages
        ],
    )


@router.delete("/sessions/{session_id}", status_code=204, summary="Delete a session")
async def delete_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    _key: APIKey = Depends(require_auth),
) -> None:
    """Delete a session and all its messages."""
    await db.execute(delete(Message).where(Message.session_id == session_id))
    await db.execute(delete(Session).where(Session.id == session_id))
    await db.commit()
    log.info("session.deleted", session_id=session_id)


@router.get("/sessions/", response_model=list[SessionInfo], summary="List sessions")
async def list_sessions(
    collection_id: str = "default",
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
    _key: APIKey = Depends(require_auth),
) -> list[SessionInfo]:
    """List recent sessions for a collection, newest first."""
    result = await db.execute(
        select(Session)
        .where(Session.collection_id == collection_id)
        .order_by(Session.created_at.desc())
        .limit(limit)
    )
    sessions = result.scalars().all()
    return [
        SessionInfo(
            session_id=s.id,
            collection_id=s.collection_id,
            created_at=s.created_at,
        )
        for s in sessions
    ]
