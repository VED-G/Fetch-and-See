"""
DocMind — Query Endpoint (SSE Streaming)
==========================================
POST /api/v1/query   → text/event-stream (SSE)
POST /api/v1/query/batch  → JSON (non-streaming, multiple questions)
"""

from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import check_rate_limit, get_db, require_auth
from app.config import get_settings
from app.models.db import APIKey, Message, Session
from app.models.schemas import QueryRequest, QueryResponse
from app.services.generation.pipeline import run_query_pipeline
from app.services.retrieval.hybrid_search import HybridSearcher

log = structlog.get_logger(__name__)
router = APIRouter()
settings = get_settings()


async def _get_or_create_session(
    session_id: str | None, collection_id: str, db: AsyncSession
) -> tuple[str, list[dict]]:
    """Return (session_id, conversation_history) for the request."""
    from sqlalchemy import select

    if session_id:
        result = await db.execute(select(Session).where(Session.id == session_id))
        session = result.scalar_one_or_none()
        if session:
            msgs_result = await db.execute(
                select(Message)
                .where(Message.session_id == session_id)
                .order_by(Message.created_at)
                .limit(10)
            )
            msgs = msgs_result.scalars().all()
            history = [{"role": m.role, "content": m.content} for m in msgs]
            return session_id, history

    # Create new session
    new_id = session_id or str(uuid.uuid4())
    db.add(Session(id=new_id, collection_id=collection_id))
    await db.commit()
    return new_id, []


async def _save_turn(
    session_id: str,
    user_question: str,
    answer: str,
    db: AsyncSession,
) -> None:
    """Persist the user + assistant turn to DB."""
    db.add(Message(session_id=session_id, role="user", content=user_question))
    db.add(Message(session_id=session_id, role="assistant", content=answer))
    await db.commit()


@router.post(
    "/query",
    summary="Ask a question (streaming SSE response)",
    response_description="Server-Sent Events stream with tokens, citations, and metadata",
)
async def query(
    request: Request,
    body: QueryRequest,
    db: AsyncSession = Depends(get_db),
    api_key: APIKey = Depends(require_auth),
    _rl: None = Depends(check_rate_limit),
) -> StreamingResponse:
    """Stream an answer to a question over the indexed documents.

    Returns a `text/event-stream` response.  Each event has the shape:
        data: {"type": "token", "content": "..."}
        data: {"type": "citations", "citations": [...]}
        data: {"type": "meta", "latency": {...}, "confidence": "high"|"low"|"none"}

    When confidence is "none", the answer token will be the "I don't know" message.
    When confidence is "low", the answer is provided but marked as uncertain.
    """
    session_id, history = await _get_or_create_session(
        body.session_id, body.collection_id, db
    )

    # Gather shared services from app state
    hybrid_searcher = HybridSearcher(
        bm25_index=getattr(request.app.state, "bm25_index", None),
        vector_store=request.app.state.vector_store,
        embedder=request.app.state.embedder,
    )
    reranker = request.app.state.reranker   # None if disabled

    from app.services.generation.providers.base import get_provider
    llm_provider = get_provider()

    collected: list[str] = []

    async def _stream_and_save():
        """Wrap the pipeline generator to also save the turn when done."""
        async for event in run_query_pipeline(
            question=body.question,
            collection_id=body.collection_id,
            session_id=session_id,
            conversation_history=history,
            hybrid_searcher=hybrid_searcher,
            reranker=reranker,
            llm_provider=llm_provider,
            cache=None,   # Cache wired in Phase 5
        ):
            # Collect token content for DB save
            import json
            try:
                data = json.loads(event.removeprefix("data: ").strip())
                if data.get("type") == "token":
                    collected.append(data.get("content", ""))
            except Exception:
                pass
            yield event

        # Save turn after stream completes
        full_answer = "".join(collected)
        await _save_turn(session_id, body.question, full_answer, db)

    return StreamingResponse(
        _stream_and_save(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Session-ID": session_id,
        },
    )
