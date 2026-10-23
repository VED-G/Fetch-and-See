"""
Fetch and See — Celery Ingestion Task
==================================
The async task dispatched when a document is uploaded.
Runs in a Celery worker process, separate from the FastAPI server.

How the task accesses app services:
  The Celery worker is a separate process.  It can't access FastAPI's
  app.state.  Instead, it creates its own DB session and service instances
  using the same config.  This is fine because:
    - Qdrant client is stateless (each call is an HTTP/local file op)
    - BM25 index: we rebuild from DB on worker startup, then update in-place
    - Embedder: loaded once per worker process (expensive but necessary)
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import structlog
from celery import Task

from app.config import get_settings
from app.tasks.celery_app import celery_app

log = structlog.get_logger(__name__)
settings = get_settings()

# Module-level singletons — initialised on first task run, reused thereafter
_embedder = None
_vector_store = None
_bm25_index = None


def _get_services():
    """Lazily initialise and cache service objects for the worker process."""
    global _embedder, _vector_store, _bm25_index

    if _embedder is None:
        from app.services.retrieval.embedder import Embedder
        _embedder = Embedder()
        # Synchronous warm-up for the worker (not in an async context)
        asyncio.run(_embedder.warm_up())

    if _vector_store is None:
        from app.services.retrieval.vector_store import VectorStore
        _vector_store = VectorStore()
        asyncio.run(_vector_store.ensure_collection())

    if _bm25_index is None:
        from app.services.retrieval.bm25_index import BM25Index
        _bm25_index = BM25Index()
        # BM25 index starts empty — it's populated as chunks are ingested
        # (A production system would rebuild from DB on startup)

    return _embedder, _vector_store, _bm25_index


@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    name="app.tasks.ingestion_tasks.ingest_document",
)
def ingest_document(
    self: Task,
    job_id: str,
    doc_id: str,
    file_path: str,
    filename: str,
    collection_id: str,
) -> dict:
    """Celery task: run full ingestion pipeline for one document.

    This function is synchronous (Celery doesn't support async tasks natively
    without a plugin).  We use asyncio.run() to run the async pipeline.

    On failure, Celery retries up to max_retries times with exponential backoff.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    log.info("task.start", job_id=job_id, doc_id=doc_id, filename=filename)

    # Sync DB session for the Celery worker
    # (We use sync SQLAlchemy here because asyncio.run() manages the event loop)
    sync_db_url = settings.DATABASE_URL.replace("+aiosqlite", "")

    embedder, vector_store, bm25_index = _get_services()

    async def _run():
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
        engine = create_async_engine(settings.DATABASE_URL, echo=False)
        factory = async_sessionmaker(engine, expire_on_commit=False)

        async with factory() as db:
            from app.services.ingestion.pipeline import run_ingestion
            try:
                result = await run_ingestion(
                    job_id=job_id,
                    doc_id=doc_id,
                    file_path=Path(file_path),
                    filename=filename,
                    collection_id=collection_id,
                    db=db,
                    embedder=embedder,
                    vector_store=vector_store,
                    bm25_index=bm25_index,
                )
                return result
            except Exception as exc:
                # Update job status to failed
                from sqlalchemy import select, update
                from app.models.db import IngestionJob
                await db.execute(
                    update(IngestionJob)
                    .where(IngestionJob.id == job_id)
                    .values(status="failed", error_message=str(exc))
                )
                await db.commit()
                raise
        await engine.dispose()

    try:
        result = asyncio.run(_run())
        log.info("task.done", job_id=job_id, **result)
        return result
    except Exception as exc:
        log.error("task.failed", job_id=job_id, error=str(exc))
        try:
            raise self.retry(exc=exc, countdown=2 ** self.request.retries * 10)
        except self.MaxRetriesExceededError:
            log.error("task.max_retries", job_id=job_id)
            return {"error": str(exc)}
