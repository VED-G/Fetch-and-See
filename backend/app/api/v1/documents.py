"""
DocMind — Document Upload & Management Endpoints
=================================================
POST /api/v1/documents/upload  → 202 {job_id}
GET  /api/v1/documents/jobs/{job_id}
GET  /api/v1/documents/
DELETE /api/v1/documents/{doc_id}
GET  /api/v1/documents/{doc_id}/chunks
"""

from __future__ import annotations

import hashlib
import shutil
import uuid
from pathlib import Path

import structlog
from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import check_rate_limit, get_db, require_auth
from app.config import get_settings
from app.core.exceptions import (
    DocumentNotFound,
    FileTooLarge,
    JobNotFound,
    UnsupportedFileType,
)
from app.models.db import APIKey, Chunk, Document, IngestionJob
from app.models.schemas import (
    ChunkPreview,
    DocumentInfo,
    DocumentUploadResponse,
    JobStatusResponse,
)

log = structlog.get_logger(__name__)
router = APIRouter()
settings = get_settings()


@router.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=202,
    summary="Upload a document for ingestion",
)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    collection_id: str = Form(default="default"),
    db: AsyncSession = Depends(get_db),
    api_key: APIKey = Depends(require_auth),
    _rl: None = Depends(check_rate_limit),
) -> DocumentUploadResponse:
    """Upload a PDF, DOCX, TXT, or MD file.

    Returns immediately with a job_id.  Poll /documents/jobs/{job_id} for progress.
    """
    # Validate file type
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower().lstrip(".")
    if suffix not in settings.allowed_extensions_set:
        raise UnsupportedFileType(
            f"File type '.{suffix}' is not supported. Allowed: {settings.ALLOWED_EXTENSIONS}"
        )

    # Read file content and validate size
    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise FileTooLarge(
            f"File is {len(content) / 1_048_576:.1f} MB, max is {settings.MAX_UPLOAD_SIZE_MB} MB"
        )

    # Save to disk
    doc_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    safe_filename = f"{doc_id}_{filename}"
    file_path = settings.upload_path / safe_filename
    file_path.write_bytes(content)

    # Compute content hash for deduplication at document level
    content_hash = hashlib.sha256(content).hexdigest()

    # Create DB records
    doc = Document(
        id=doc_id,
        filename=safe_filename,
        original_filename=filename,
        file_type=suffix,
        file_size_bytes=len(content),
        content_hash=content_hash,
        collection_id=collection_id,
        status="pending",
    )
    job = IngestionJob(id=job_id, document_id=doc_id, status="pending")
    db.add(doc)
    db.add(job)
    await db.commit()

    # Dispatch Celery task
    from app.tasks.ingestion_tasks import ingest_document
    ingest_document.apply_async(
        kwargs={
            "job_id": job_id,
            "doc_id": doc_id,
            "file_path": str(file_path),
            "filename": filename,
            "collection_id": collection_id,
        },
        queue="ingestion",
    )

    log.info("document.uploaded", doc_id=doc_id, job_id=job_id, filename=filename)
    return DocumentUploadResponse(job_id=job_id, document_id=doc_id)


@router.get(
    "/documents/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Poll ingestion job status",
)
async def get_job_status(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    _key: APIKey = Depends(require_auth),
) -> JobStatusResponse:
    result = await db.execute(select(IngestionJob).where(IngestionJob.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise JobNotFound(f"Job {job_id} not found.")
    return JobStatusResponse(
        job_id=job.id,
        document_id=job.document_id,
        status=job.status,
        progress_pct=job.progress_pct,
        chunks_processed=job.chunks_processed,
        total_chunks=job.total_chunks,
        error_message=job.error_message,
        duration_seconds=job.duration_seconds,
        created_at=job.created_at,
        completed_at=job.completed_at,
    )


@router.get(
    "/documents/",
    response_model=list[DocumentInfo],
    summary="List documents in a collection",
)
async def list_documents(
    collection_id: str = "default",
    db: AsyncSession = Depends(get_db),
    _key: APIKey = Depends(require_auth),
) -> list[DocumentInfo]:
    result = await db.execute(
        select(Document)
        .where(Document.collection_id == collection_id)
        .order_by(Document.created_at.desc())
    )
    docs = result.scalars().all()
    return [
        DocumentInfo(
            id=d.id, filename=d.original_filename, file_type=d.file_type,
            file_size_bytes=d.file_size_bytes, language=d.language,
            chunk_count=d.chunk_count, collection_id=d.collection_id,
            status=d.status, created_at=d.created_at,
        )
        for d in docs
    ]


@router.delete("/documents/{doc_id}", status_code=204, summary="Delete a document")
async def delete_document(
    doc_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
    _key: APIKey = Depends(require_auth),
) -> None:
    result = await db.execute(select(Document).where(Document.id == doc_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise DocumentNotFound(f"Document {doc_id} not found.")

    # Delete from Qdrant
    vector_store = request.app.state.vector_store
    await vector_store.delete_by_doc_id(doc_id)

    # Remove from BM25 index
    # (bm25 is in app state if set up there, else skip gracefully)
    bm25 = getattr(request.app.state, "bm25_index", None)
    if bm25:
        bm25.remove_by_doc_id(doc_id)

    # Delete file from disk
    file_path = settings.upload_path / doc.filename
    if file_path.exists():
        file_path.unlink()

    # Delete from DB (cascades to chunks)
    await db.delete(doc)
    await db.commit()
    log.info("document.deleted", doc_id=doc_id)


@router.get(
    "/documents/{doc_id}/chunks",
    response_model=list[ChunkPreview],
    summary="Preview extracted chunks (debug)",
)
async def get_chunks(
    doc_id: str,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
    _key: APIKey = Depends(require_auth),
) -> list[ChunkPreview]:
    result = await db.execute(
        select(Chunk).where(Chunk.document_id == doc_id).limit(limit)
    )
    chunks = result.scalars().all()
    if not chunks:
        raise DocumentNotFound(f"No chunks found for document {doc_id}.")
    return [
        ChunkPreview(
            chunk_id=c.id, text=c.text, page_number=c.page_number,
            chunk_index=c.chunk_index, language=c.language,
            token_count=c.token_count,
        )
        for c in chunks
    ]
