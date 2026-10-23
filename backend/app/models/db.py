"""
SQLAlchemy async ORM models.

Tables: documents, chunks, parent_chunks, api_keys, sessions, messages, ingestion_jobs.
Uses SQLite by default; swap DATABASE_URL to Postgres without changing any model code.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


# ─── Documents ────────────────────────────────────────────────────────────────

class Document(Base):
    """One row per uploaded file."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_type: Mapped[str] = mapped_column(String(10), nullable=False)   # pdf|docx|txt|md
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    language: Mapped[str | None] = mapped_column(String(10))             # ISO 639-1
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False) # SHA-256
    collection_id: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), default="pending"
    )  # pending|processing|completed|failed
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow
    )

    # Relationships
    chunks: Mapped[list[Chunk]] = relationship(
        "Chunk", back_populates="document", cascade="all, delete-orphan"
    )
    parent_chunks: Mapped[list[ParentChunk]] = relationship(
        "ParentChunk", back_populates="document", cascade="all, delete-orphan"
    )


# ─── Chunks ───────────────────────────────────────────────────────────────────

class ParentChunk(Base):
    """Large context chunk (1536 tokens) stored for LLM consumption.

    When a child chunk matches during retrieval, we fetch its parent chunk
    so the LLM gets fuller context — this is the 'parent-child' RAG pattern.
    """

    __tablename__ = "parent_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)

    document: Mapped[Document] = relationship("Document", back_populates="parent_chunks")
    children: Mapped[list[Chunk]] = relationship(
        "Chunk", back_populates="parent_chunk"
    )


class Chunk(Base):
    """Small embedding chunk (512 tokens) — the unit indexed in Qdrant.

    Each chunk stores:
      - qdrant_point_id: the UUID used as Qdrant point ID (for cross-referencing)
      - content_hash: SHA-256 of the text for deduplication
      - embedding_model: which model produced the vector (so we can re-embed if we upgrade)
    """

    __tablename__ = "chunks"
    __table_args__ = (
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_content_hash", "content_hash"),
        Index("ix_chunks_qdrant_point_id", "qdrant_point_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    parent_chunk_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("parent_chunks.id", ondelete="SET NULL")
    )
    qdrant_point_id: Mapped[str] = mapped_column(String(36), nullable=False, unique=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    language: Mapped[str | None] = mapped_column(String(10))
    embedding_model: Mapped[str] = mapped_column(String(255), nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    document: Mapped[Document] = relationship("Document", back_populates="chunks")
    parent_chunk: Mapped[ParentChunk | None] = relationship(
        "ParentChunk", back_populates="children"
    )


# ─── API Keys ─────────────────────────────────────────────────────────────────

class APIKey(Base):
    """Hashed API keys for request authentication.

    We never store the raw key — only the bcrypt hash.
    The raw key is shown to the user exactly once at creation time.
    """

    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    key_prefix: Mapped[str] = mapped_column(String(8), nullable=False)   # dm_xxxx
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


# ─── Sessions & Messages ──────────────────────────────────────────────────────

class Session(Base):
    """A conversation session — holds a list of messages for memory/rewriting."""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    collection_id: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    last_active_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    messages: Mapped[list[Message]] = relationship(
        "Message", back_populates="session", cascade="all, delete-orphan",
        order_by="Message.created_at"
    )


class Message(Base):
    """One turn (user or assistant) inside a session."""

    __tablename__ = "messages"
    __table_args__ = (Index("ix_messages_session_id", "session_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)   # user|assistant
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Store raw citations list as JSON string
    citations_json: Mapped[str | None] = mapped_column(Text)

    # Metrics for the assistant turn
    confidence_score: Mapped[float | None] = mapped_column(Float)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    provider: Mapped[str | None] = mapped_column(String(50))

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    session: Mapped[Session] = relationship("Session", back_populates="messages")


# ─── Ingestion Jobs ───────────────────────────────────────────────────────────

class IngestionJob(Base):
    """Tracks the status of async ingestion tasks dispatched to Celery.

    The API returns job_id on upload (202 Accepted).
    The client polls GET /api/v1/documents/jobs/{job_id} to check progress.
    """

    __tablename__ = "ingestion_jobs"
    __table_args__ = (Index("ix_ingestion_jobs_document_id", "document_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    document_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="SET NULL")
    )
    status: Mapped[str] = mapped_column(
        String(20), default="pending"
    )  # pending|processing|completed|failed
    progress_pct: Mapped[int] = mapped_column(Integer, default=0)
    chunks_processed: Mapped[int] = mapped_column(Integer, default=0)
    total_chunks: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text)
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)
