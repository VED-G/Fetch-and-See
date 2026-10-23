"""
DocMind — /metrics Endpoint (Prometheus format)
=================================================
Exposes application metrics in the Prometheus text format.
Grafana or any Prometheus-compatible scraper can pull from this endpoint.

Metrics exposed:
  docmind_requests_total{method, endpoint, status}   — request count
  docmind_request_duration_seconds{endpoint}         — latency histogram
  docmind_documents_total                            — total documents ingested
  fas_chunks_total                               — total chunks indexed
  docmind_cache_hits_total                           — semantic cache hits
  docmind_cache_misses_total                         — semantic cache misses
  docmind_llm_tokens_total{provider}                 — tokens generated

Interview talking point:
  "Prometheus metrics + Grafana is the industry standard for observability.
   The histogram gives you p50/p95/p99 latency without storing every request —
   it uses pre-defined buckets and is O(1) memory per metric."
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

router = APIRouter()

# ─── Metric definitions ───────────────────────────────────────────────────────
# These are module-level singletons — defined once, incremented anywhere.

REQUEST_COUNT = Counter(
    "fas_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_DURATION = Histogram(
    "fas_request_duration_seconds",
    "HTTP request duration in seconds",
    ["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

DOCUMENTS_INGESTED = Counter(
    "fas_documents_total",
    "Total documents successfully ingested",
)

CHUNKS_INDEXED = Counter(
    "fas_chunks_total",
    "Total chunks indexed into the vector store",
)

CACHE_HITS = Counter(
    "fas_cache_hits_total",
    "Semantic cache hits (query answered from cache)",
)

CACHE_MISSES = Counter(
    "fas_cache_misses_total",
    "Semantic cache misses (query required retrieval + LLM)",
)

LLM_TOKENS = Counter(
    "fas_llm_tokens_total",
    "Total LLM output tokens generated",
    ["provider"],
)

RETRIEVAL_DURATION = Histogram(
    "fas_retrieval_duration_seconds",
    "Retrieval pipeline duration (BM25 + vector + rerank)",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

GENERATION_DURATION = Histogram(
    "fas_generation_duration_seconds",
    "LLM generation duration",
    ["provider"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)


# ─── Endpoint ─────────────────────────────────────────────────────────────────

@router.get("/metrics", include_in_schema=False)
async def metrics_endpoint() -> PlainTextResponse:
    """Return Prometheus-format metrics."""
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )
