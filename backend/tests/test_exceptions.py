"""
Tests for the exception hierarchy and HTTP error shapes.
"""

from __future__ import annotations

import pytest
from app.core.exceptions import (
    AuthenticationError,
    DocumentNotFound,
    DocMindError,
    FileTooLarge,
    InvalidFileType,
    RateLimitExceeded,
    VectorStoreError,
)


def test_base_error_defaults() -> None:
    err = DocMindError("something broke")
    assert err.http_status == 500
    assert err.error_code == "InternalError"
    assert err.message == "something broke"


def test_document_not_found() -> None:
    err = DocumentNotFound("doc-abc")
    assert err.http_status == 404
    assert err.error_code == "DocumentNotFound"
    assert "doc-abc" in err.message


def test_authentication_error() -> None:
    err = AuthenticationError()
    assert err.http_status == 401


def test_rate_limit_exceeded() -> None:
    err = RateLimitExceeded(retry_after=30)
    assert err.http_status == 429
    assert err.retry_after == 30


def test_file_too_large() -> None:
    err = FileTooLarge(size_mb=75, max_mb=50)
    assert err.http_status == 413
    assert "75" in err.message or "50" in err.message


def test_invalid_file_type() -> None:
    err = InvalidFileType("exe")
    assert err.http_status == 415
    assert "exe" in err.message


def test_vector_store_error() -> None:
    err = VectorStoreError("connection refused")
    assert err.http_status == 503


def test_all_errors_inherit_base() -> None:
    for exc_cls in [
        AuthenticationError, DocumentNotFound, FileTooLarge,
        InvalidFileType, RateLimitExceeded, VectorStoreError,
    ]:
        assert issubclass(exc_cls, DocMindError)
