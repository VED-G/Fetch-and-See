"""
DocMind — Embedding Service
=============================
Wraps sentence-transformers to produce dense vectors for both documents
(at ingestion time) and queries (at query time).

Model default: paraphrase-multilingual-MiniLM-L12-v2
  - 118M parameters, ~450 MB RAM on CPU
  - 384-dimensional output vectors
  - Supports 50+ languages including Hindi and English
  - Fast: ~300 sentences/sec on a modern CPU

Upgrade path (set EMBEDDING_MODEL env var):
  - BAAI/bge-m3 → 1024-dim, 100+ langs, 8k context, dense+sparse from one model
    (needs ~1.5 GB RAM)

Interview talking point:
  "I separated the embedder into a singleton loaded at startup (warm_up) and
   reused across requests.  Loading a 118M-parameter model takes ~2 seconds;
   doing that per-request would add 2s to every query.  The warm_up() call
   in the FastAPI lifespan ensures the model is hot before the first request."
"""

from __future__ import annotations

import asyncio
from functools import lru_cache

import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.core.exceptions import EmbeddingError

log = structlog.get_logger(__name__)
settings = get_settings()


class Embedder:
    """Thread-safe wrapper around SentenceTransformer.

    All encode calls are run in a threadpool executor so they don't block
    the asyncio event loop (PyTorch inference is CPU-bound, not I/O-bound).
    """

    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None
        self._model_name = settings.EMBEDDING_MODEL
        self._device = settings.EMBEDDING_DEVICE
        self._batch_size = settings.EMBEDDING_BATCH_SIZE
        self._dim = settings.EMBEDDING_DIM

    async def warm_up(self) -> None:
        """Load model weights into memory.  Call once at startup."""
        log.info("embedder.loading", model=self._model_name, device=self._device)
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(
            None, self._load_model
        )
        log.info("embedder.loaded", model=self._model_name, dim=self._dim)

    def _load_model(self) -> SentenceTransformer:
        return SentenceTransformer(
            self._model_name,
            device=self._device,
        )

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts.  Returns list of float vectors.

        Runs in a threadpool to avoid blocking the event loop.
        Batching is handled internally by sentence-transformers.
        """
        if not texts:
            return []
        if self._model is None:
            raise EmbeddingError("Embedder not initialised. Call warm_up() first.")

        try:
            loop = asyncio.get_event_loop()
            vectors: np.ndarray = await loop.run_in_executor(
                None,
                lambda: self._model.encode(  # type: ignore[union-attr]
                    texts,
                    batch_size=self._batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,   # Normalise → cosine sim = dot product
                    convert_to_numpy=True,
                ),
            )
            return vectors.tolist()
        except Exception as exc:
            raise EmbeddingError(f"Embedding failed: {exc}") from exc

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string (common shorthand)."""
        results = await self.embed_texts([query])
        return results[0]

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Return a module-level cached Embedder instance.

    Note: warm_up() must be called before the first embed call.
    The FastAPI lifespan does this automatically.
    """
    return Embedder()
