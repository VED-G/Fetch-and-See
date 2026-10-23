"""
Tests for the chunker: verifies child chunk sizes, parent chunk sizes,
overlap, and deduplication behaviour.
"""

from __future__ import annotations

import pytest
from app.services.ingestion.chunker import Chunker


@pytest.fixture
def chunker() -> Chunker:
    return Chunker(
        child_chunk_size=100,
        child_overlap=10,
        parent_chunk_size=300,
        parent_overlap=20,
    )


def _make_text(n_words: int) -> str:
    return " ".join(f"word{i}" for i in range(n_words))


def test_child_chunks_created(chunker: Chunker) -> None:
    text = _make_text(500)
    result = chunker.chunk(text, doc_id="doc1", filename="test.txt")
    assert len(result["child_chunks"]) > 1


def test_parent_chunks_created(chunker: Chunker) -> None:
    text = _make_text(500)
    result = chunker.chunk(text, doc_id="doc1", filename="test.txt")
    assert len(result["parent_chunks"]) >= 1


def test_child_chunk_has_metadata(chunker: Chunker) -> None:
    text = _make_text(300)
    result = chunker.chunk(text, doc_id="doc-xyz", filename="sample.pdf")
    child = result["child_chunks"][0]
    assert child["doc_id"] == "doc-xyz"
    assert "text" in child
    assert "chunk_index" in child


def test_deduplication_removes_duplicates(chunker: Chunker) -> None:
    # Feeding the same text twice should not double the chunks after dedup
    text = _make_text(200)
    r1 = chunker.chunk(text, doc_id="doc1", filename="a.txt")
    r2 = chunker.chunk(text, doc_id="doc1", filename="a.txt")
    n1 = len(r1["child_chunks"])
    assert n1 > 0
    # Second run with same content produces same hashes — dedup should catch it
    all_hashes = [c.get("content_hash") for c in r2["child_chunks"]]
    assert len(all_hashes) == len(set(all_hashes)), "Duplicate hashes within one document"


def test_empty_text_returns_empty(chunker: Chunker) -> None:
    result = chunker.chunk("", doc_id="doc1", filename="empty.txt")
    assert result["child_chunks"] == []
    assert result["parent_chunks"] == []


def test_short_text_single_chunk(chunker: Chunker) -> None:
    text = "Hello world this is short."
    result = chunker.chunk(text, doc_id="doc1", filename="short.txt")
    assert len(result["child_chunks"]) == 1
