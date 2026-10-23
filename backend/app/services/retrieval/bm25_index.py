"""
DocMind — BM25 Sparse Retrieval Index
========================================
Wraps the rank-bm25 library for keyword-based (sparse) retrieval.
BM25 is the gold standard for exact-match search — it catches things like
error codes, product names, and proper nouns that vector search misses.

How BM25 works (interview version):
  BM25 scores documents by term frequency (TF) and inverse document frequency
  (IDF), with saturation to prevent single-term spam from dominating.
  Formula: sum over query terms of  IDF(t) * (tf * (k1+1)) / (tf + k1*(1-b+b*dl/avgdl))
  where k1 controls term frequency saturation (~1.5) and b controls document
  length normalisation (~0.75).

Why both BM25 AND dense vectors?
  Dense vectors: "cardiac arrest" matches "heart attack" ✓, "ERR-4092" → misses ✗
  BM25:          "cardiac arrest" misses "heart attack" ✗, "ERR-4092" → finds it ✓
  Hybrid (both): best of both worlds, fused via RRF.

Persistence:
  The BM25 index is rebuilt from the SQLite chunk table on startup.
  This is fast (pure Python, seconds for thousands of chunks) and avoids
  storing a separate serialised file.

Interview talking point:
  "I chose rank-bm25 over Elasticsearch for a laptop-scale project because
   it's pure Python, zero infrastructure, and rebuilds from the DB in seconds.
   At scale I'd swap in OpenSearch or Qdrant's built-in sparse vectors."
"""

from __future__ import annotations

import re
import unicodedata

import structlog
from rank_bm25 import BM25Okapi

from app.config import get_settings

log = structlog.get_logger(__name__)
settings = get_settings()


def _tokenize(text: str) -> list[str]:
    """Simple multilingual tokeniser.

    Steps:
    1. Unicode NFKC normalisation (handles full-width chars, ligatures)
    2. Lowercase
    3. Split on whitespace and punctuation
    4. Filter tokens shorter than 2 chars

    Works for Hindi (Devanagari) because Unicode normalisation preserves
    the script, and BM25 treats each Unicode word as a token regardless
    of script. A production system would use langdetect + script-aware
    tokenisers (e.g., Indic NLP library for Hindi morphological analysis).
    """
    text = unicodedata.normalize("NFKC", text).lower()
    tokens = re.split(r"[\s\p{P}\p{Z}]+", text) if False else re.split(r"[\s,.!?;:\"'()\[\]{}<>/\\|@#$%^&*+=~`]+", text)
    return [t for t in tokens if len(t) >= 2]


class BM25Index:
    """In-memory BM25 index over all indexed chunks.

    The index is rebuilt at startup from the SQLite DB and updated
    incrementally as new documents are ingested.

    Thread safety:
      Reads (search) are safe concurrently.
      Writes (add_chunks) should be serialised — Celery workers run one at
      a time per queue (worker_prefetch_multiplier=1), so this is fine.
    """

    def __init__(self) -> None:
        self._corpus: list[list[str]] = []   # tokenised chunks
        self._metadata: list[dict] = []      # parallel metadata list
        self._index: BM25Okapi | None = None

    def build(self, texts: list[str], metadata: list[dict]) -> None:
        """Build index from scratch (called at startup).

        Args:
            texts:    Raw text of each chunk.
            metadata: Parallel list of {chunk_id, doc_id, filename, page_number, ...}
        """
        self._corpus = [_tokenize(t) for t in texts]
        self._metadata = metadata
        if self._corpus:
            self._index = BM25Okapi(self._corpus)
            log.info("bm25.built", chunk_count=len(self._corpus))
        else:
            log.info("bm25.empty")

    def add_chunks(self, texts: list[str], metadata: list[dict]) -> None:
        """Incrementally add new chunks to the index.

        BM25Okapi doesn't support incremental updates natively, so we
        rebuild the index after each batch.  At laptop scale (< 100K chunks)
        rebuilding takes < 1 second and is perfectly acceptable.
        """
        new_tokenised = [_tokenize(t) for t in texts]
        self._corpus.extend(new_tokenised)
        self._metadata.extend(metadata)
        self._index = BM25Okapi(self._corpus)
        log.info("bm25.updated", total_chunks=len(self._corpus))

    def remove_by_doc_id(self, doc_id: str) -> None:
        """Remove all chunks belonging to a document and rebuild."""
        paired = [
            (tokens, meta)
            for tokens, meta in zip(self._corpus, self._metadata)
            if meta.get("doc_id") != doc_id
        ]
        if paired:
            self._corpus, self._metadata = zip(*paired)  # type: ignore[assignment]
            self._corpus = list(self._corpus)
            self._metadata = list(self._metadata)
        else:
            self._corpus, self._metadata = [], []

        self._index = BM25Okapi(self._corpus) if self._corpus else None
        log.info("bm25.removed", doc_id=doc_id, remaining=len(self._corpus))

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """Return top-k chunks by BM25 score.

        Returns a list of dicts with the same shape as VectorStore.search():
            {chunk_id, score, doc_id, filename, page_number, text, ...}

        Scores are normalised to [0, 1] so they can be meaningfully compared
        with vector similarity scores (though RRF doesn't need normalisation —
        it only uses ranks).
        """
        if self._index is None or not query.strip():
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        raw_scores = self._index.get_scores(query_tokens)

        # Get top-k indices sorted by score descending
        top_indices = sorted(
            range(len(raw_scores)),
            key=lambda i: raw_scores[i],
            reverse=True,
        )[:top_k]

        max_score = max(raw_scores[i] for i in top_indices) if top_indices else 1.0
        if max_score == 0:
            return []

        results = []
        for idx in top_indices:
            score = raw_scores[idx]
            if score <= 0:
                continue
            result = {
                **self._metadata[idx],
                "score": float(score / max_score),  # normalise to [0, 1]
                "raw_bm25_score": float(score),
            }
            results.append(result)

        return results

    @property
    def size(self) -> int:
        return len(self._corpus)
