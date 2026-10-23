"""
DocMind — Ingestion Pipeline Orchestrator
==========================================
Ties together: extract → language detect → chunk → deduplicate → embed → store.
Called by the Celery task.

The pipeline updates job progress in Redis so the polling endpoint
(/api/v1/documents/jobs/{job_id}) can show real-time progress to the user.

Interview talking point:
  "I designed the pipeline as a sequence of pure functions with side effects
   only at the boundaries (DB writes, Qdrant writes).  This makes each step
   independently testable — I can unit-test the chunker with fake pages
   without touching a real database."
"""

from __future__ import annotations

import hashlib
import time
import uuid
from pathlib import Path

import structlog
from langdetect import LangDetectException, detect
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.db import Chunk as ChunkModel
from app.models.db import Document, IngestionJob, ParentChunk as ParentChunkModel
from app.services.ingestion.chunker import Chunk, ParentChunk, chunk_pages
from app.services.ingestion.extractor import extract
from app.services.retrieval.bm25_index import BM25Index
from app.services.retrieval.embedder import Embedder
from app.services.retrieval.vector_store import VectorStore

log = structlog.get_logger(__name__)
settings = get_settings()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _detect_language(text: str) -> str:
    """Detect language of text.  Returns ISO 639-1 code, default 'en'."""
    try:
        sample = text[:2000]   # Only sample first 2000 chars for speed
        return detect(sample)
    except LangDetectException:
        return "en"


async def run_ingestion(
    *,
    job_id: str,
    doc_id: str,
    file_path: Path,
    filename: str,
    collection_id: str,
    db: AsyncSession,
    embedder: Embedder,
    vector_store: VectorStore,
    bm25_index: BM25Index,
    redis_client: object | None = None,
) -> dict:
    """Full ingestion pipeline.  Returns summary dict.

    Steps:
        1. Update job status → PROCESSING
        2. Extract text from file
        3. Detect language
        4. Chunk pages (parent + child)
        5. Deduplicate chunks (SHA-256 check in DB)
        6. Embed child chunks in batches
        7. Upsert to Qdrant
        8. Save to SQLite
        9. Update BM25 index
       10. Update job status → COMPLETED
    """
    start_time = time.monotonic()

    async def _update_job(status: str, progress: int, **kwargs: object) -> None:
        from sqlalchemy import select
        result = await db.execute(select(IngestionJob).where(IngestionJob.id == job_id))
        job = result.scalar_one_or_none()
        if job:
            job.status = status
            job.progress_pct = progress
            for k, v in kwargs.items():
                setattr(job, k, v)
            await db.commit()

    await _update_job("processing", 5)
    log.info("pipeline.start", job_id=job_id, doc_id=doc_id, filename=filename)

    # ── Step 1: Extract text ──────────────────────────────────────────────────
    extraction = extract(file_path, filename)
    language = _detect_language(extraction.full_text)
    await _update_job("processing", 20)
    log.info("pipeline.extracted", pages=len(extraction.pages), language=language)

    # Update document with language
    from sqlalchemy import select, update
    await db.execute(
        update(Document).where(Document.id == doc_id).values(language=language)
    )
    await db.commit()

    # ── Step 2: Chunk ─────────────────────────────────────────────────────────
    parent_chunks, child_chunks = chunk_pages(extraction.pages)
    await _update_job("processing", 35, total_chunks=len(child_chunks))
    log.info("pipeline.chunked", parents=len(parent_chunks), children=len(child_chunks))

    # ── Step 3: Deduplicate ───────────────────────────────────────────────────
    existing_hashes: set[str] = set()
    existing_result = await db.execute(
        __import__("sqlalchemy", fromlist=["select"]).select(ChunkModel.content_hash)
    )
    existing_hashes = {row[0] for row in existing_result.all()}

    new_children = [c for c in child_chunks if _sha256(c.text) not in existing_hashes]
    skipped = len(child_chunks) - len(new_children)
    if skipped:
        log.info("pipeline.dedup", skipped=skipped, unique=len(new_children))

    # ── Step 4: Save parent chunks to DB ─────────────────────────────────────
    parent_id_map: dict[str, str] = {}   # in-memory id → DB id
    for parent in parent_chunks:
        db_parent = ParentChunkModel(
            id=parent.id,
            document_id=doc_id,
            text=parent.text,
            page_number=parent.page_number,
            chunk_index=parent.chunk_index,
        )
        db.add(db_parent)
        parent_id_map[parent.id] = parent.id
    await db.commit()

    # ── Step 5: Embed + Upsert (batched) ─────────────────────────────────────
    batch_size = settings.EMBEDDING_BATCH_SIZE
    embedded_count = 0
    bm25_texts: list[str] = []
    bm25_metas: list[dict] = []

    for batch_start in range(0, len(new_children), batch_size):
        batch = new_children[batch_start : batch_start + batch_size]
        texts = [c.text for c in batch]

        # Embed
        vectors = await embedder.embed_texts(texts)

        # Build Qdrant payloads
        qdrant_ids = [c.id for c in batch]
        payloads = [
            {
                "doc_id": doc_id,
                "filename": filename,
                "collection_id": collection_id,
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
                "language": language,
                "text": c.text,          # stored in payload for BM25 fallback
                "parent_chunk_id": c.parent_id,
            }
            for c in batch
        ]

        # Upsert to Qdrant
        await vector_store.upsert_chunks(
            chunk_ids=qdrant_ids,
            vectors=vectors,
            payloads=payloads,
        )

        # Save to SQLite
        for chunk, payload in zip(batch, payloads):
            db_chunk = ChunkModel(
                id=chunk.id,
                document_id=doc_id,
                parent_chunk_id=chunk.parent_id,
                qdrant_point_id=chunk.id,
                text=chunk.text,
                content_hash=_sha256(chunk.text),
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                language=language,
                embedding_model=settings.EMBEDDING_MODEL,
                token_count=chunk.token_count,
            )
            db.add(db_chunk)
            bm25_texts.append(chunk.text)
            bm25_metas.append({
                "chunk_id": chunk.id,
                "doc_id": doc_id,
                "filename": filename,
                "page_number": chunk.page_number,
                "text": chunk.text,
            })

        await db.commit()
        embedded_count += len(batch)

        # Update progress
        progress = 35 + int(60 * embedded_count / max(len(new_children), 1))
        await _update_job("processing", min(progress, 90), chunks_processed=embedded_count)

    # ── Step 6: Update BM25 index ────────────────────────────────────────────
    if bm25_texts:
        bm25_index.add_chunks(bm25_texts, bm25_metas)

    # ── Step 7: Update document chunk count and mark complete ─────────────────
    duration = time.monotonic() - start_time
    await db.execute(
        update(Document).where(Document.id == doc_id).values(
            chunk_count=embedded_count,
            status="completed",
        )
    )
    await _update_job(
        "completed", 100,
        chunks_processed=embedded_count,
        duration_seconds=duration,
    )

    log.info(
        "pipeline.done",
        job_id=job_id,
        doc_id=doc_id,
        chunks=embedded_count,
        duration_s=f"{duration:.1f}",
    )

    return {
        "doc_id": doc_id,
        "chunks_indexed": embedded_count,
        "chunks_skipped": skipped,
        "pages": len(extraction.pages),
        "language": language,
        "duration_seconds": duration,
    }
