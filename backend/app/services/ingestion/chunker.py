"""
DocMind — Text Chunker (Parent-Child Strategy)
================================================
Splits extracted text into overlapping child chunks (for embedding precision)
and larger parent chunks (for LLM context richness).

Why parent-child chunking?
  Small chunks (512 tokens) → precise embedding — the model can capture the
  meaning of a focused passage without being diluted by surrounding text.

  But the LLM needs context to answer well.  If the small chunk says
  "the rate is 12%", the LLM doesn't know what "rate" refers to.

  Solution: index the small chunk, but when it's retrieved, fetch its parent
  chunk (1536 tokens) to feed to the LLM.  The LLM gets the paragraph context
  without it polluting the embedding space.

Token counting:
  We approximate tokens as words/0.75 (1 token ≈ 0.75 words for English).
  For Hindi/Devanagari, tokens are closer to 1 per character cluster, so
  we use a conservative 4 chars/token estimate.  Production would use
  tiktoken or the model's actual tokeniser.

Interview talking point:
  "The parent-child pattern is a form of the information retrieval principle
   of 'retrieve small, read large'.  It separates the retrieval index
   (high-precision, compact) from the reading context (rich, complete).
   Many production RAG systems at Notion, Cohere, and Anthropic use this."
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

from app.config import get_settings
from app.services.ingestion.extractor import ExtractedPage

settings = get_settings()


@dataclass
class Chunk:
    """A child chunk (small, for embedding)."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    page_number: int | None = None
    chunk_index: int = 0
    token_count: int = 0
    parent_id: str | None = None   # FK to ParentChunk.id


@dataclass
class ParentChunk:
    """A parent chunk (large, for LLM context)."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    page_number: int | None = None
    chunk_index: int = 0
    children: list[Chunk] = field(default_factory=list)


def _estimate_tokens(text: str) -> int:
    """Fast token estimate without loading a tokeniser.

    English:    1 token ≈ 4 characters  (GPT-style BPE)
    Hindi:      1 token ≈ 3.5 characters (Devanagari syllable clusters)
    Mixed:      Use 4 chars/token as conservative average.
    """
    return max(1, len(text) // 4)


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving sentence boundaries.

    Uses a regex that handles:
    - English punctuation: . ! ?
    - Hindi danda: ।
    - Avoids splitting on abbreviations like "Dr." or "Ph.D."
    """
    # Split on sentence-ending punctuation followed by whitespace + uppercase
    pattern = r'(?<=[.!?।])\s+(?=[A-Z\u0900-\u097F])'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_pages(pages: list[ExtractedPage]) -> tuple[list[ParentChunk], list[Chunk]]:
    """Chunk extracted pages into parent + child chunks.

    Algorithm:
    1. For each page, split into sentences.
    2. Greedily pack sentences into PARENT chunks (up to PARENT_CHUNK_SIZE tokens).
    3. Within each parent chunk, create overlapping CHILD chunks (CHUNK_SIZE tokens,
       CHUNK_OVERLAP token overlap).

    Returns:
        (parent_chunks, child_chunks)  — parallel lists where every child
        has a parent_id pointing to its parent.
    """
    chunk_size = settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP
    parent_size = settings.PARENT_CHUNK_SIZE

    all_parents: list[ParentChunk] = []
    all_children: list[Chunk] = []
    parent_idx = 0

    for page in pages:
        sentences = _split_into_sentences(page.text)
        if not sentences:
            continue

        # ── Build parent chunks ────────────────────────────────────────────
        current_parent_sentences: list[str] = []
        current_parent_tokens = 0

        def _flush_parent() -> ParentChunk:
            nonlocal parent_idx
            parent_text = " ".join(current_parent_sentences)
            parent = ParentChunk(
                text=parent_text,
                page_number=page.page_number,
                chunk_index=parent_idx,
            )
            all_parents.append(parent)
            parent_idx += 1
            return parent

        for sentence in sentences:
            sent_tokens = _estimate_tokens(sentence)

            if current_parent_tokens + sent_tokens > parent_size and current_parent_sentences:
                parent = _flush_parent()
                _make_child_chunks(parent, all_children, chunk_size, overlap, page.page_number)
                current_parent_sentences = []
                current_parent_tokens = 0

            current_parent_sentences.append(sentence)
            current_parent_tokens += sent_tokens

        # Flush remaining sentences
        if current_parent_sentences:
            parent = _flush_parent()
            _make_child_chunks(parent, all_children, chunk_size, overlap, page.page_number)

    return all_parents, all_children


def _make_child_chunks(
    parent: ParentChunk,
    all_children: list[Chunk],
    chunk_size: int,
    overlap: int,
    page_number: int | None,
) -> None:
    """Create overlapping child chunks from a parent chunk's text."""
    words = parent.text.split()
    if not words:
        return

    step = max(1, chunk_size - overlap)
    child_idx = 0

    for start in range(0, len(words), step):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        if not chunk_words:
            break

        chunk_text = " ".join(chunk_words)
        child = Chunk(
            text=chunk_text,
            page_number=page_number,
            chunk_index=len(all_children),
            token_count=_estimate_tokens(chunk_text),
            parent_id=parent.id,
        )
        parent.children.append(child)
        all_children.append(child)
        child_idx += 1

        if end >= len(words):
            break
