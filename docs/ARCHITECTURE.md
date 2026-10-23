# Fetch and See — Architecture

This document explains the system design decisions and data flow. Read alongside `DESIGN.md` for the full rationale.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                             │
│   Streamlit UI (port 8501)   ·   curl / any HTTP client         │
└──────────────────────┬──────────────────────────────────────────┘
                       │ HTTP / SSE
┌──────────────────────▼──────────────────────────────────────────┐
│                    FastAPI Backend (port 8000)                   │
│                                                                  │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌────────────────┐  │
│  │/documents│  │ /query   │  │/search   │  │ /sessions      │  │
│  │ upload   │  │ SSE      │  │/debug    │  │ /auth /health  │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────────────┘  │
└───────┼─────────────┼─────────────┼───────────────────────────┘
        │             │             │
        ▼             │             │
┌───────────────┐     │             │
│ Celery Worker │     │             │
│               │     │             │
│  Extract      │     │             │
│  ↓ Chunk      │     │             │
│  ↓ Embed      │     │             │
│  ↓ Upsert     │     │             │
└───────┬───────┘     │             │
        │             │             │
        ▼             ▼             ▼
┌────────────────────────────────────────────────────────────────┐
│                       Service Layer                             │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  Query Pipeline (hot path)                │  │
│  │                                                           │  │
│  │  query ──► embed ──► BM25 + vector ──► RRF ──► rerank   │  │
│  │                                              ↓            │  │
│  │           confidence gate (0.15 / 0.35)  ◄──┘            │  │
│  │                    ↓                                       │  │
│  │          build_messages (citation [N] tags)                │  │
│  │                    ↓                                       │  │
│  │            Groq LLM → stream tokens → SSE                 │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────────┬─────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│    Qdrant    │  │     Redis    │  │    SQLite    │
│ Vector store │  │ Celery queue │  │  ORM + jobs  │
│ (dense vecs) │  │ Rate limiter │  │              │
└──────────────┘  │ Semantic     │  └──────────────┘
                  │ cache (TTL)  │
                  └──────────────┘
```

---

## Ingestion Pipeline

```
PDF / DOCX / TXT / MD
        │
        ▼ extractor.py
   Raw text + metadata
   (page numbers, section headers)
        │
        ▼ chunker.py
   Parent chunks (1536 tokens, overlap 50)
        └─► Child chunks (512 tokens, overlap 50)
                │
                ▼ Deduplication (SHA-256 of text)
                │
                ▼ embedder.py (MiniLM, batch=16)
           Dense vectors (384-dim)
                │
         ┌──────┴──────┐
         ▼              ▼
    Qdrant upsert   BM25 index update
    (child chunks)  (in-memory, rebuilt on restart)
         │
         ▼
    SQLite: documents + chunks + parent_chunks tables
```

---

## Query Pipeline (Hot Path)

```
User question
      │
      ▼  1. Semantic cache check (Redis, threshold=0.97)
      │     ─── HIT: return cached answer + citations
      │
      ▼  2. Query rewrite (if session history exists)
      │     Groq rewrites to be standalone (removes "it", "they", etc.)
      │
      ▼  3. Embed query (MiniLM → 384-dim vector)
      │
      ├──► 4a. BM25 search  (top-20 candidates)
      ├──► 4b. Vector search (top-20 candidates, Qdrant)
      │
      ▼  5. RRF Fusion  k=60
      │     score(d) = Σ 1/(k + rank_i(d))
      │
      ▼  6. Reranker (optional, bge-reranker-v2-m3)
      │     Cross-encoder over top-30 RRF candidates → top-5
      │
      ▼  7. Confidence gate
      │     score < 0.15  →  "I don't know" (skip LLM)
      │     score < 0.35  →  LLM + low-confidence warning
      │     score ≥ 0.35  →  LLM + confident citations
      │
      ▼  8. Prompt builder
      │     [SOURCE 1]: {chunk_text} (filename, page N)
      │     [SOURCE 2]: ...
      │     Q: {question}
      │     Rule: cite using [1], [2], etc.
      │
      ▼  9. Groq streaming
      │     Yield SSE: {type: token, content: "..."}
      │
      ▼ 10. Citation parser + SSE flush
           Yield SSE: {type: citations, citations: [...]}
           Yield SSE: {type: meta, confidence, latency}
```

---

## Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| LLM | Groq (llama-3.1-8b-instant) | Free tier, 300-500 tok/s on LPU, no GPU needed |
| Embedding | MiniLM-L12-v2 | 384-dim, ~450 MB RAM, multilingual, fast on CPU |
| Vector DB | Qdrant | Rust, local-file mode (no Docker required in dev) |
| Retrieval | BM25 + dense hybrid | Exact-match + semantic, RRF avoids score normalisation |
| Chunking | Parent-child (512/1536t) | Retrieve small, read large — better citation precision |
| Confidence | Two-level gate (0.15/0.35) | Graceful degradation vs. hard cutoff |
| Async | asyncio + threadpool | CPU-bound inference runs in executor, never blocks event loop |
| Tasks | Celery + Redis | Ingestion survives restarts, retries on failure |
| Config | Pydantic Settings | Typed, validated, 12-factor, one-line env override |
| Schemas | Separate from ORM | Prevents leaking internal fields, allows independent evolution |

---

## File Map

```
docmind/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── v1/
│   │   │   │   ├── documents.py     upload, job poll, list, delete, chunk preview
│   │   │   │   ├── query.py         streaming SSE query + session persistence
│   │   │   │   ├── search.py        raw retrieval debug endpoint (no LLM)
│   │   │   │   ├── sessions.py      history, delete, list
│   │   │   │   ├── auth.py          API key create / revoke
│   │   │   │   └── cache.py         cache stats / flush
│   │   │   ├── deps.py              DB session, auth, rate limiter (shared deps)
│   │   │   ├── health.py            GET /health
│   │   │   └── metrics.py           GET /metrics (Prometheus)
│   │   ├── core/
│   │   │   ├── exceptions.py        DocMindError hierarchy → consistent JSON
│   │   │   └── logging.py           structlog JSON structured logging
│   │   ├── models/
│   │   │   ├── db.py                SQLAlchemy async ORM models
│   │   │   └── schemas.py           Pydantic request/response schemas
│   │   ├── services/
│   │   │   ├── ingestion/
│   │   │   │   ├── extractor.py     PDF/DOCX/TXT/MD → raw text + metadata
│   │   │   │   ├── chunker.py       parent-child chunking with overlap
│   │   │   │   └── pipeline.py      orchestrates extract→chunk→embed→store
│   │   │   ├── retrieval/
│   │   │   │   ├── embedder.py      SentenceTransformer (async, threadpool)
│   │   │   │   ├── vector_store.py  Qdrant async client (local + server mode)
│   │   │   │   ├── bm25_index.py    rank-bm25 with multilingual tokeniser
│   │   │   │   ├── hybrid_search.py RRF fusion
│   │   │   │   └── reranker.py      cross-encoder (bge-reranker-v2-m3, optional)
│   │   │   └── generation/
│   │   │       ├── providers/
│   │   │       │   ├── base.py      LLMProvider Protocol + factory
│   │   │       │   ├── groq.py      Groq async streaming (primary)
│   │   │       │   ├── ollama.py    Ollama local fallback
│   │   │       │   └── openai_compat.py  OpenAI-compatible wrapper
│   │   │       ├── prompt.py        citation-aware prompt builder
│   │   │       └── pipeline.py      full RAG pipeline (7 stages, SSE output)
│   │   ├── tasks/
│   │   │   ├── celery_app.py        Celery app config
│   │   │   └── ingestion_tasks.py   async ingestion Celery task
│   │   ├── config.py                Pydantic Settings (all env vars)
│   │   └── main.py                  FastAPI app factory + lifespan
│   ├── scripts/
│   │   ├── create_api_key.py        first-time setup bootstrap
│   │   ├── eval_ragas.py            RAGAS evaluation (4 metrics, 3 modes)
│   │   ├── run_ablation.py          all-mode ablation table
│   │   ├── bench_query.py           p50/p95/p99 query latency
│   │   └── bench_embeddings.py      embedding throughput by batch size
│   └── tests/
│       ├── conftest.py              pytest fixtures (test app, test DB, mock LLM)
│       └── fixtures/
│           └── eval_dataset.json    ground-truth QA pairs for RAGAS
├── frontend/
│   ├── app.py                       Home page + shared sidebar config
│   ├── .streamlit/config.toml       Dark theme
│   └── pages/
│       ├── 1_💬_Chat.py             Streaming chat with citations
│       ├── 2_📤_Upload.py           Upload + progress + doc list
│       └── 3_🔬_Debug.py            Raw retrieval debug panel
├── docker-compose.yml               Redis + Qdrant + API + Worker
├── backend/Dockerfile               Python 3.11 slim
├── README.md                        Quickstart + API examples
└── .github/workflows/ci.yml         Lint + typecheck + tests on push
```
