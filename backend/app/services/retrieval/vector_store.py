"""
DocMind — Qdrant Vector Store Wrapper
=======================================
Thin async wrapper around the Qdrant client that handles:
  - Collection creation with correct vector config
  - Upserting points with dense vectors + metadata payload
  - Dense similarity search with optional metadata filtering
  - Deleting points by document ID

Why Qdrant?
  Written in Rust → very low CPU and memory overhead compared to Go-based
  alternatives.  Supports dense + sparse vectors in the same collection,
  which is exactly what hybrid search needs.  Local file mode means zero
  Docker needed in development; same client code works against Qdrant Cloud.

Interview talking point:
  "Qdrant uses HNSW (Hierarchical Navigable Small World) graphs for approximate
   nearest neighbour search.  HNSW trades a tiny recall penalty for O(log n)
   search instead of O(n) brute-force.  At 1M vectors the difference is
   milliseconds vs. minutes."
"""

from __future__ import annotations

import uuid

import structlog
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import get_settings
from app.core.exceptions import VectorStoreError

log = structlog.get_logger(__name__)
settings = get_settings()


class VectorStore:
    """Async Qdrant client wrapper."""

    def __init__(self) -> None:
        if settings.QDRANT_MODE == "local":
            # Local file-based storage — no server needed
            self._client = AsyncQdrantClient(
                path=str(settings.qdrant_local_path)
            )
        else:
            # Remote Qdrant server (Docker or Qdrant Cloud)
            self._client = AsyncQdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY or None,
            )
        self._collection = settings.QDRANT_COLLECTION

    async def ensure_collection(self) -> None:
        """Create the collection if it doesn't exist.

        We use cosine distance because our embedder normalises vectors
        (so cosine = dot product = faster computation).
        """
        try:
            exists = await self._client.collection_exists(self._collection)
            if not exists:
                await self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=models.VectorParams(
                        size=settings.EMBEDDING_DIM,
                        distance=models.Distance.COSINE,
                        on_disk=True,   # Keep vectors on disk for low-RAM machines
                    ),
                )
                log.info("qdrant.collection_created", name=self._collection)
            else:
                log.info("qdrant.collection_exists", name=self._collection)
        except Exception as exc:
            raise VectorStoreError(f"Failed to create Qdrant collection: {exc}") from exc

    async def upsert_chunks(
        self,
        chunk_ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict],
    ) -> None:
        """Upsert a batch of chunks into Qdrant.

        Args:
            chunk_ids:  UUIDs used as Qdrant point IDs (also stored in SQLite).
            vectors:    Dense embeddings, one per chunk.
            payloads:   Metadata dicts: {doc_id, filename, page_number, text, ...}
        """
        if not chunk_ids:
            return
        try:
            points = [
                models.PointStruct(
                    id=chunk_id,
                    vector=vector,
                    payload=payload,
                )
                for chunk_id, vector, payload in zip(chunk_ids, vectors, payloads)
            ]
            await self._client.upsert(
                collection_name=self._collection,
                points=points,
                wait=True,   # Wait for indexing to complete before returning
            )
            log.info("qdrant.upserted", count=len(points))
        except Exception as exc:
            raise VectorStoreError(f"Qdrant upsert failed: {exc}") from exc

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 20,
        filter_doc_ids: list[str] | None = None,
    ) -> list[dict]:
        """Dense vector similarity search.

        Returns a list of dicts:
            {chunk_id, score, doc_id, filename, page_number, text, ...}

        Args:
            query_vector:    Query embedding from the Embedder.
            top_k:           Number of results to return.
            filter_doc_ids:  If provided, restrict search to these documents.
        """
        try:
            query_filter = None
            if filter_doc_ids:
                query_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchAny(any=filter_doc_ids),
                        )
                    ]
                )

            results = await self._client.search(
                collection_name=self._collection,
                query_vector=query_vector,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )

            return [
                {
                    "chunk_id": str(hit.id),
                    "score": hit.score,
                    **hit.payload,  # type: ignore[arg-type]
                }
                for hit in results
            ]
        except Exception as exc:
            raise VectorStoreError(f"Qdrant search failed: {exc}") from exc

    async def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all points belonging to a document."""
        try:
            await self._client.delete(
                collection_name=self._collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="doc_id",
                                match=models.MatchValue(value=doc_id),
                            )
                        ]
                    )
                ),
            )
            log.info("qdrant.deleted", doc_id=doc_id)
        except Exception as exc:
            raise VectorStoreError(f"Qdrant delete failed: {exc}") from exc

    async def get_collection_info(self) -> dict:
        """Return basic collection stats."""
        try:
            info = await self._client.get_collection(self._collection)
            return {
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": str(info.status),
            }
        except Exception as exc:
            raise VectorStoreError(f"Qdrant info failed: {exc}") from exc
