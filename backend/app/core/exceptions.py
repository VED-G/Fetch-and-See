"""
DocMind — Custom Exception Hierarchy
======================================
All application exceptions inherit from DocMindError.
FastAPI exception handlers (registered in main.py) convert these into
consistent JSON responses:

    {
        "error": "DocumentNotFound",
        "message": "Document abc123 does not exist",
        "detail": null,
        "request_id": "req-xyz"
    }

Interview talking point:
  "A shared exception base class means every error the API returns has the
   same shape — no surprises for API consumers.  Each HTTP status code maps
   to a specific exception type, so you can grep the codebase for, say,
   RateLimitExceeded and find every place it's raised."
"""

from __future__ import annotations


class DocMindError(Exception):
    """Base class for all DocMind application errors."""

    http_status: int = 500
    error_code: str = "InternalError"

    def __init__(self, message: str, detail: object = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


# ── 400 Bad Request ───────────────────────────────────────────────────────────

class ValidationError(DocMindError):
    """Request payload or parameters failed validation."""

    http_status = 400
    error_code = "ValidationError"


class UnsupportedFileType(DocMindError):
    """Uploaded file extension is not in the allowed list."""

    http_status = 400
    error_code = "UnsupportedFileType"


class FileTooLarge(DocMindError):
    """Uploaded file exceeds MAX_UPLOAD_SIZE_MB."""

    http_status = 400
    error_code = "FileTooLarge"


class InvalidQuery(DocMindError):
    """Query string is empty or malformed."""

    http_status = 400
    error_code = "InvalidQuery"


# ── 401 Unauthorized ─────────────────────────────────────────────────────────

class AuthenticationError(DocMindError):
    """Missing or invalid API key."""

    http_status = 401
    error_code = "AuthenticationError"


# ── 403 Forbidden ────────────────────────────────────────────────────────────

class PermissionDenied(DocMindError):
    """API key exists but lacks permission for this action."""

    http_status = 403
    error_code = "PermissionDenied"


# ── 404 Not Found ────────────────────────────────────────────────────────────

class DocumentNotFound(DocMindError):
    """Requested document ID does not exist."""

    http_status = 404
    error_code = "DocumentNotFound"


class SessionNotFound(DocMindError):
    """Requested conversation session does not exist."""

    http_status = 404
    error_code = "SessionNotFound"


class JobNotFound(DocMindError):
    """Requested ingestion job ID does not exist."""

    http_status = 404
    error_code = "JobNotFound"


# ── 409 Conflict ─────────────────────────────────────────────────────────────

class DuplicateDocument(DocMindError):
    """A document with the same content hash already exists."""

    http_status = 409
    error_code = "DuplicateDocument"


# ── 422 Unprocessable Entity ─────────────────────────────────────────────────

class ExtractionError(DocMindError):
    """Text could not be extracted from the uploaded file."""

    http_status = 422
    error_code = "ExtractionError"


# ── 429 Too Many Requests ────────────────────────────────────────────────────

class RateLimitExceeded(DocMindError):
    """Request rate limit exceeded.

    Attributes:
        retry_after: Seconds until the client may retry.
    """

    http_status = 429
    error_code = "RateLimitExceeded"

    def __init__(self, message: str, retry_after: int = 60) -> None:
        super().__init__(message)
        self.retry_after = retry_after


# ── 503 Service Unavailable ───────────────────────────────────────────────────

class LLMProviderError(DocMindError):
    """LLM provider (Groq/Ollama/OpenAI) returned an error or is unreachable."""

    http_status = 503
    error_code = "LLMProviderError"


class VectorStoreError(DocMindError):
    """Qdrant is unavailable or returned an unexpected error."""

    http_status = 503
    error_code = "VectorStoreError"


class EmbeddingError(DocMindError):
    """Embedding model failed to encode text."""

    http_status = 503
    error_code = "EmbeddingError"
