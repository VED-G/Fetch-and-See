"""
Pydantic request/response schemas (separate from ORM models).

Keeping schemas and DB models in separate layers prevents leaking internal
fields (e.g. key_hash) in API responses and allows independent evolution.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ─── Common ───────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Consistent error shape returned by all exception handlers."""

    error: str
    message: str
    detail: Any | None = None
    request_id: str | None = None


class HealthResponse(BaseModel):
    status: str                        # ok | degraded | down
    version: str
    environment: str
    services: dict[str, str]           # {qdrant: ok, redis: ok, groq: ok}
    timestamp: datetime


# ─── Documents ────────────────────────────────────────────────────────────────

class DocumentUploadResponse(BaseModel):
    """Returned immediately after upload (202 Accepted)."""

    job_id: str
    document_id: str
    message: str = "Ingestion started. Poll /jobs/{job_id} for progress."


class JobStatusResponse(BaseModel):
    """Polling response for async ingestion job."""

    job_id: str
    document_id: str | None
    status: str                        # pending|processing|completed|failed
    progress_pct: int = Field(ge=0, le=100)
    chunks_processed: int
    total_chunks: int
    error_message: str | None = None
    duration_seconds: float | None = None
    created_at: datetime
    completed_at: datetime | None = None


class DocumentInfo(BaseModel):
    """Summary of an ingested document."""

    id: str
    filename: str
    file_type: str
    file_size_bytes: int
    language: str | None
    chunk_count: int
    collection_id: str
    status: str
    created_at: datetime


class ChunkPreview(BaseModel):
    """A single chunk returned by the debug /chunks endpoint."""

    chunk_id: str
    text: str
    page_number: int | None
    chunk_index: int
    language: str | None
    token_count: int


# ─── Query ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Request body for POST /api/v1/query."""

    question: str = Field(min_length=1, max_length=2000)
    collection_id: str = Field(default="default", max_length=255)
    session_id: str | None = None      # Pass to enable conversation memory
    stream: bool = True                # SSE streaming (default) or JSON
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question cannot be blank or whitespace only")
        return v.strip()


class Citation(BaseModel):
    """One citation entry in a query response."""

    index: int                         # [1], [2], etc. as referenced in answer text
    chunk_id: str
    document_id: str
    filename: str
    page_number: int | None
    text_preview: str                  # First 200 chars of the chunk
    relevance_score: float = Field(ge=0.0, le=1.0)


class QueryResponse(BaseModel):
    """Non-streaming query response (when stream=False)."""

    answer: str
    confidence: str                    # high | low | none
    confidence_score: float = Field(ge=0.0, le=1.0)
    citations: list[Citation]
    session_id: str
    latency_breakdown: LatencyBreakdown
    cache_hit: bool = False
    provider: str


class LatencyBreakdown(BaseModel):
    """Per-stage latency in milliseconds — key for performance analysis."""

    cache_check_ms: int = 0
    query_rewrite_ms: int = 0
    retrieval_ms: int = 0              # BM25 + vector search combined
    rerank_ms: int = 0                 # 0 if reranker disabled
    generation_ms: int = 0
    total_ms: int = 0


# ─── Sessions ─────────────────────────────────────────────────────────────────

class MessageOut(BaseModel):
    """One message in a conversation history."""

    id: str
    role: str                          # user | assistant
    content: str
    citations: list[Citation] | None = None
    confidence_score: float | None = None
    provider: str | None = None
    created_at: datetime


class SessionHistoryResponse(BaseModel):
    session_id: str
    collection_id: str
    messages: list[MessageOut]
    created_at: datetime
    last_active_at: datetime


# ─── Search (Debug) ───────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    collection_id: str = "default"
    top_k: int = Field(default=10, ge=1, le=50)
    include_bm25: bool = True
    include_vector: bool = True


class SearchResultItem(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_number: int | None
    text: str
    score: float
    source: str                        # bm25 | vector | reranker


class SearchDebugResponse(BaseModel):
    query: str
    rewritten_query: str | None
    bm25_results: list[SearchResultItem]
    vector_results: list[SearchResultItem]
    fused_results: list[SearchResultItem]
    reranked_results: list[SearchResultItem]
    latency_breakdown: LatencyBreakdown


# ─── Auth ─────────────────────────────────────────────────────────────────────

class CreateAPIKeyRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    is_admin: bool = False


class APIKeyResponse(BaseModel):
    """Returned once at creation — the raw key is NEVER shown again."""

    id: str
    name: str
    api_key: str                       # Raw key shown once
    key_prefix: str                    # e.g. "dm_a3f2"
    is_admin: bool
    created_at: datetime


class APIKeyInfo(BaseModel):
    """Safe listing response — no raw key."""

    id: str
    name: str
    key_prefix: str
    is_admin: bool
    is_active: bool
    last_used_at: datetime | None
    created_at: datetime


# ─── Cache ────────────────────────────────────────────────────────────────────

class CacheStatsResponse(BaseModel):
    hit_rate_pct: float
    total_requests: int
    cache_hits: int
    cache_misses: int
    entries: int
    size_mb: float


# ─── Voice ────────────────────────────────────────────────────────────────────

class TranscriptionResponse(BaseModel):
    text: str
    language: str
    confidence: float | None = None
    duration_seconds: float | None = None


# Forward reference resolution
QueryResponse.model_rebuild()
