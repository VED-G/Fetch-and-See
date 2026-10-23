"""
Full RAG query pipeline as an async SSE generator.

Stages: cache check -> query rewrite -> hybrid search -> rerank -> generate.
Yields SSE events: {type: token}, {type: citations}, {type: meta}, {type: error}.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import AsyncIterator

import structlog

from app.config import get_settings
from app.models.schemas import Citation, LatencyBreakdown
from app.services.generation.prompt import (
    NO_CONTEXT_RESPONSE,
    build_messages,
    build_rewrite_messages,
)
from app.services.retrieval.reranker import get_confidence_score

log = structlog.get_logger(__name__)
settings = get_settings()


def _parse_citations(answer: str, chunks: list[dict]) -> list[Citation]:
    """Map [1], [2] markers in the answer to Citation objects.

    Extracts citation indices referenced in the answer text,
    then looks up the corresponding chunk metadata.
    """
    import re
    cited_indices: set[int] = set()
    for match in re.finditer(r"\[(\d+)\]", answer):
        idx = int(match.group(1))
        if 1 <= idx <= len(chunks):
            cited_indices.add(idx)

    citations: list[Citation] = []
    for i in sorted(cited_indices):
        chunk = chunks[i - 1]  # 1-indexed → 0-indexed
        citations.append(Citation(
            index=i,
            chunk_id=chunk.get("chunk_id", ""),
            document_id=chunk.get("doc_id", ""),
            filename=chunk.get("filename", "Unknown"),
            page_number=chunk.get("page_number"),
            text_preview=chunk.get("text", "")[:200],
            relevance_score=round(
                chunk.get("reranker_score", chunk.get("rrf_score", 0.0)) or 0.0, 3
            ),
        ))
    return citations


async def run_query_pipeline(
    *,
    question: str,
    collection_id: str,
    session_id: str,
    conversation_history: list[dict[str, str]],
    hybrid_searcher: object,
    reranker: object | None,
    llm_provider: object,
    cache: object | None = None,
) -> AsyncIterator[str]:
    """Full RAG pipeline as an async SSE generator.

    Yields SSE-formatted strings:
        'data: {"type": "token", "content": "..."}\n\n'
        'data: {"type": "citations", "citations": [...]}\n\n'
        'data: {"type": "meta", ...}\n\n'

    Args:
        question:              Raw user question.
        collection_id:         Qdrant collection to search.
        session_id:            For conversation memory.
        conversation_history:  Past messages from the session.
        hybrid_searcher:       HybridSearcher instance.
        reranker:              Reranker instance (or None if disabled).
        llm_provider:          LLMProvider instance.
        cache:                 SemanticCache instance (or None).
    """
    latency = LatencyBreakdown()
    full_answer = ""

    def _sse(event_type: str, **data: object) -> str:
        payload = json.dumps({"type": event_type, **data})
        return f"data: {payload}\n\n"

    # ── Stage 0: Cache check ───────────────────────────────────────────────
    t0 = time.monotonic()
    if cache is not None:
        cached = await cache.get(question)  # type: ignore[union-attr]
        latency.cache_check_ms = int((time.monotonic() - t0) * 1000)
        if cached:
            latency.total_ms = latency.cache_check_ms
            yield _sse("token", content=cached["answer"])
            yield _sse("citations", citations=[c for c in cached.get("citations", [])])
            yield _sse("meta", cache_hit=True, latency=latency.model_dump(),
                       confidence=cached.get("confidence", "high"),
                       confidence_score=cached.get("confidence_score", 1.0))
            return

    latency.cache_check_ms = int((time.monotonic() - t0) * 1000)

    # ── Stage 1: Query rewriting ──────────────────────────────────────────
    t1 = time.monotonic()
    rewritten_query = question
    if conversation_history:
        try:
            rewrite_msgs = build_rewrite_messages(question, conversation_history)
            rewritten_query = await llm_provider.complete(  # type: ignore[union-attr]
                rewrite_msgs, max_tokens=100, temperature=0.0
            )
            rewritten_query = rewritten_query.strip() or question
            log.debug("query.rewritten", original=question, rewritten=rewritten_query)
        except Exception:
            rewritten_query = question   # Fail gracefully

    latency.query_rewrite_ms = int((time.monotonic() - t1) * 1000)

    # ── Stage 2: Hybrid retrieval ─────────────────────────────────────────
    t2 = time.monotonic()
    candidates = await hybrid_searcher.search(  # type: ignore[union-attr]
        rewritten_query, top_k=settings.RERANKER_CANDIDATE_K
    )
    latency.retrieval_ms = int((time.monotonic() - t2) * 1000)

    # ── Stage 3: Reranking (optional) ─────────────────────────────────────
    t3 = time.monotonic()
    if reranker is not None and candidates:
        top_chunks = await reranker.rerank(  # type: ignore[union-attr]
            rewritten_query, candidates, top_k=settings.RERANKER_TOP_K
        )
    else:
        top_chunks = candidates[:settings.RERANKER_TOP_K]
    latency.rerank_ms = int((time.monotonic() - t3) * 1000)

    # ── Stage 4: Confidence gate ──────────────────────────────────────────
    confidence_score = get_confidence_score(top_chunks, reranker is not None)

    if confidence_score < settings.NO_ANSWER_THRESHOLD:
        # Below NO_ANSWER threshold → skip LLM entirely
        log.info(
            "confidence.no_answer",
            score=confidence_score,
            threshold=settings.NO_ANSWER_THRESHOLD,
        )
        latency.total_ms = int((time.monotonic() - t0) * 1000)
        yield _sse("token", content=NO_CONTEXT_RESPONSE)
        yield _sse("citations", citations=[])
        yield _sse("meta",
                   cache_hit=False,
                   latency=latency.model_dump(),
                   confidence="none",
                   confidence_score=round(confidence_score, 3),
                   rewritten_query=rewritten_query)
        return

    # Determine confidence label
    confidence_label = (
        "low" if confidence_score < settings.LOW_CONFIDENCE_THRESHOLD else "high"
    )

    # ── Stage 5: Generation ───────────────────────────────────────────────
    t4 = time.monotonic()
    messages = build_messages(rewritten_query, top_chunks, conversation_history)

    collected_tokens: list[str] = []
    try:
        async for token in await llm_provider.stream(  # type: ignore[union-attr]
            messages,
            max_tokens=settings.GROQ_MAX_TOKENS,
            temperature=settings.GROQ_TEMPERATURE,
        ):
            collected_tokens.append(token)
            yield _sse("token", content=token)
    except Exception as exc:
        log.error("generation.failed", error=str(exc))
        yield _sse("error", message=f"Generation failed: {exc}")
        return

    full_answer = "".join(collected_tokens)
    latency.generation_ms = int((time.monotonic() - t4) * 1000)
    latency.total_ms = int((time.monotonic() - t0) * 1000)

    # ── Stage 6: Parse citations & send metadata ──────────────────────────
    citations = _parse_citations(full_answer, top_chunks)
    yield _sse("citations", citations=[c.model_dump() for c in citations])
    yield _sse(
        "meta",
        cache_hit=False,
        latency=latency.model_dump(),
        confidence=confidence_label,
        confidence_score=round(confidence_score, 3),
        rewritten_query=rewritten_query,
        provider=settings.LLM_PROVIDER,
        session_id=session_id,
    )

    # ── Stage 7: Store in cache ───────────────────────────────────────────
    if cache is not None and full_answer and confidence_label == "high":
        await cache.set(  # type: ignore[union-attr]
            question,
            {
                "answer": full_answer,
                "citations": [c.model_dump() for c in citations],
                "confidence": confidence_label,
                "confidence_score": confidence_score,
            },
        )

    log.info(
        "query.done",
        confidence=confidence_label,
        score=round(confidence_score, 3),
        total_ms=latency.total_ms,
        provider=settings.LLM_PROVIDER,
    )
