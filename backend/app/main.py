"""
FastAPI application factory.

create_app() builds and returns the app instance, making it easy to create
isolated instances in tests. The lifespan context manager handles startup
(DB init, Qdrant setup, model warm-up) and shutdown (connection cleanup).
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings
from app.core.exceptions import DocMindError, RateLimitExceeded
from app.core.logging import setup_logging
from app.models.db import Base

log = structlog.get_logger(__name__)
settings = get_settings()


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Run startup tasks before yield; shutdown tasks after yield."""

    # 1. Configure logging
    setup_logging(
        log_level="DEBUG" if settings.APP_DEBUG else "INFO",
        json_logs=settings.is_production,
    )
    log.info("docmind.starting", env=settings.APP_ENV, provider=settings.LLM_PROVIDER)

    # 2. Create database tables (idempotent)
    engine = create_async_engine(settings.DATABASE_URL, echo=settings.APP_DEBUG)
    async_session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("database.ready", url=settings.DATABASE_URL)

    # 3. Store shared resources in app.state (accessible via request.app.state)
    app.state.db_engine = engine
    app.state.db_session_factory = async_session_factory

    # 4. Initialise Qdrant collection (lazy — done here so workers reuse client)
    from app.services.retrieval.vector_store import VectorStore
    vector_store = VectorStore()
    await vector_store.ensure_collection()
    app.state.vector_store = vector_store
    log.info("qdrant.ready", mode=settings.QDRANT_MODE, collection=settings.QDRANT_COLLECTION)

    # 5. Load embedding model (warm-up: first call is slow, subsequent calls fast)
    from app.services.retrieval.embedder import Embedder
    embedder = Embedder()
    await embedder.warm_up()
    app.state.embedder = embedder
    log.info(
        "embedder.ready",
        model=settings.EMBEDDING_MODEL,
        device=settings.EMBEDDING_DEVICE,
        dim=settings.EMBEDDING_DIM,
    )

    # 6. Load reranker if enabled
    if settings.ENABLE_RERANKER:
        from app.services.retrieval.reranker import Reranker
        reranker = Reranker()
        await reranker.warm_up()
        app.state.reranker = reranker
        log.info("reranker.ready", model=settings.RERANKER_MODEL)
    else:
        app.state.reranker = None
        log.info("reranker.disabled", reason="ENABLE_RERANKER=false")

    # 7. Build BM25 index from existing chunks in DB
    from app.services.retrieval.bm25_index import BM25Index
    bm25_index = BM25Index()
    async with async_session_factory() as db:
        from sqlalchemy import select as sa_select
        from app.models.db import Chunk as ChunkModel
        result = await db.execute(sa_select(ChunkModel.id, ChunkModel.text, ChunkModel.document_id))
        rows = result.all()
        if rows:
            bm25_index.add_chunks([
                {"chunk_id": r.id, "text": r.text, "doc_id": r.document_id}
                for r in rows
            ])
    app.state.bm25_index = bm25_index
    log.info("bm25.ready", chunks_indexed=len(rows) if rows else 0)

    log.info("docmind.ready", host=settings.APP_HOST, port=settings.APP_PORT)

    yield  # ← Application runs here

    # ── Shutdown ──────────────────────────────────────────────────────────────
    log.info("docmind.shutting_down")
    await engine.dispose()
    log.info("docmind.stopped")


# ─── App Factory ──────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="DocMind API",
        description=(
            "Production-style RAG API: upload documents, ask questions, "
            "get grounded answers with citations."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],  # Streamlit
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request ID Middleware ──────────────────────────────────────────────────
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next: object) -> Response:
        request_id = request.headers.get("X-Request-ID") or f"req-{uuid.uuid4().hex[:8]}"
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        start = time.monotonic()
        response: Response = await call_next(request)  # type: ignore[operator]
        duration_ms = int((time.monotonic() - start) * 1000)

        response.headers["X-Request-ID"] = request_id
        log.info(
            "http.request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
        return response

    # ── Exception Handlers ────────────────────────────────────────────────────
    @app.exception_handler(DocMindError)
    async def docmind_error_handler(request: Request, exc: DocMindError) -> JSONResponse:
        request_id = request.headers.get("X-Request-ID", "unknown")
        headers = {}
        if isinstance(exc, RateLimitExceeded):
            headers["Retry-After"] = str(exc.retry_after)
        return JSONResponse(
            status_code=exc.http_status,
            content={
                "error": exc.error_code,
                "message": exc.message,
                "detail": exc.detail,
                "request_id": request_id,
            },
            headers=headers,
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled.exception", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalError",
                "message": "An unexpected error occurred.",
                "detail": str(exc) if settings.APP_DEBUG else None,
                "request_id": request.headers.get("X-Request-ID", "unknown"),
            },
        )

    # ── Routers ───────────────────────────────────────────────────────────────
    from app.api.v1 import documents, query, search, sessions, auth, cache
    from app.api import health

    app.include_router(health.router, tags=["System"])
    app.include_router(documents.router, prefix="/api/v1", tags=["Documents"])
    app.include_router(query.router, prefix="/api/v1", tags=["Query"])
    app.include_router(search.router, prefix="/api/v1", tags=["Search"])
    app.include_router(sessions.router, prefix="/api/v1", tags=["Sessions"])
    app.include_router(auth.router, prefix="/api/v1", tags=["Auth"])
    app.include_router(cache.router, prefix="/api/v1", tags=["Cache"])

    if settings.ENABLE_VOICE:
        from app.api.v1 import voice
        app.include_router(voice.router, prefix="/api/v1", tags=["Voice"])

    if settings.ENABLE_METRICS:
        from app.api import metrics
        app.include_router(metrics.router, tags=["System"])

    return app


# ─── Entry point (used by uvicorn) ────────────────────────────────────────────
app = create_app()
