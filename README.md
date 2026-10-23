# Fetch and See

A production-style RAG (Retrieval-Augmented Generation) API. Upload PDFs, DOCX, TXT, or Markdown files and ask questions — you get grounded answers with inline citations and page numbers, a per-query latency breakdown, and a confidence score.

**Stack:** FastAPI · Qdrant · sentence-transformers · Groq (LLM) · Redis · Celery · SQLite · Python 3.11

---

## Features

- **Hybrid search** — BM25 + dense vector retrieval fused via Reciprocal Rank Fusion (RRF)
- **Cross-encoder reranking** — optional, set `ENABLE_RERANKER=true`
- **Two-level confidence gating** — skips the LLM if context is irrelevant; warns if it's borderline
- **Streaming SSE responses** — tokens appear as they're generated, citations sent at the end
- **Conversation memory** — multi-turn sessions with automatic query rewriting
- **Async ingestion** — file upload returns 202 immediately; Celery worker handles the rest
- **Multilingual** — embedding model supports English + Hindi (50+ languages)
- **No paid APIs required** — Groq has a free tier (14k RPM on llama-3.1-8b-instant)

---

## Quick Start (no Docker)

> Works on a low-end PC. Groq is free. No GPU needed.

### 1. Clone and set up

```bash
git clone https://github.com/VED-G/Fetch-and-See.git
cd Fetch-and-See
```

### 2. Create .env

```bash
cp .env.example .env
# Edit .env and set your GROQ_API_KEY (free at console.groq.com)
```

### 3. Install dependencies

```bash
cd backend
pip install -e ".[all]"
```

### 4. Create your first API key

```bash
python scripts/create_api_key.py --name "dev" --admin
# Copy the printed key — you'll need it for all API calls
```

### 5. Start Redis (needed for Celery)

```bash
# Option A: Docker (easiest)
docker run -d -p 6379:6379 redis:7-alpine

# Option B: Windows — download from https://github.com/microsoftarchive/redis/releases
```

### 6. Start the API

```bash
uvicorn app.main:app --reload
# API docs at http://localhost:8000/docs
```

### 7. Start the Celery worker (separate terminal)

```bash
cd backend
celery -A app.tasks.celery_app.celery_app worker --loglevel=info -Q ingestion --concurrency=1
```

---

## Quick Start (Docker Compose)

```bash
cp .env.example .env
# Set GROQ_API_KEY in .env

docker compose up -d
docker compose exec api python scripts/create_api_key.py --name "dev" --admin
```

---

## API Usage

### Upload a document

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -H "Authorization: Bearer dm_YOUR_KEY" \
  -F "file=@your_document.pdf" \
  -F "collection_id=default"
# Returns: {"job_id": "...", "document_id": "..."}
```

### Poll ingestion status

```bash
curl http://localhost:8000/api/v1/documents/jobs/{job_id} \
  -H "Authorization: Bearer dm_YOUR_KEY"
# Returns: {"status": "completed", "chunks_processed": 47, ...}
```

### Ask a question (streaming)

```bash
curl -N -X POST http://localhost:8000/api/v1/query \
  -H "Authorization: Bearer dm_YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?", "collection_id": "default"}'

# Response stream (one event per line):
# data: {"type": "token", "content": "The refund"}
# data: {"type": "token", "content": " policy allows"}
# data: {"type": "citations", "citations": [{"index": 1, "filename": "policy.pdf", "page_number": 3}]}
# data: {"type": "meta", "confidence": "high", "confidence_score": 0.82, "latency": {...}}
```

### Debug retrieval (no LLM)

```bash
curl -X POST http://localhost:8000/api/v1/search/debug \
  -H "Authorization: Bearer dm_YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "refund policy", "collection_id": "default"}'
```

---

## Configuration

All settings are in `.env`. Key ones:

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | **Required.** Free at console.groq.com |
| `LLM_PROVIDER` | `groq` | `groq` \| `ollama` \| `openai` |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Model name |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | ~450 MB RAM |
| `ENABLE_RERANKER` | `false` | Set `true` for better quality (+550 MB RAM) |
| `QDRANT_MODE` | `local` | `local` (no Docker) or `server` |
| `NO_ANSWER_THRESHOLD` | `0.15` | Below this → skip LLM, return "I don't know" |
| `LOW_CONFIDENCE_THRESHOLD` | `0.35` | Below this → answer with warning badge |

See [`.env.example`](.env.example) for the full list.

---

## Project Structure

```
docmind/
├── backend/
│   ├── app/
│   │   ├── api/v1/          # FastAPI routers (documents, query, search, sessions, auth)
│   │   ├── core/            # Exceptions, structured logging
│   │   ├── models/          # SQLAlchemy ORM models + Pydantic schemas
│   │   ├── services/
│   │   │   ├── ingestion/   # Extractor, chunker, pipeline
│   │   │   ├── retrieval/   # Embedder, Qdrant, BM25, hybrid search, reranker
│   │   │   └── generation/  # LLM providers (Groq/Ollama/OpenAI), prompt, pipeline
│   │   ├── tasks/           # Celery task definitions
│   │   └── config.py        # Pydantic Settings
│   ├── scripts/             # create_api_key.py, bench_query.py, eval_ragas.py
│   └── tests/
├── docker-compose.yml
├── DESIGN.md                # Architecture decisions
└── .env.example
```

---

## Architecture

```
Upload → FastAPI → Celery task
                      ↓
              Extract (PDF/DOCX/TXT/MD)
                      ↓
              Chunk (parent 1536t / child 512t)
                      ↓
              Embed (MiniLM, batch=16)
                      ↓
              Qdrant upsert + BM25 update + SQLite

Query → FastAPI → cache check
                      ↓ (miss)
              Query rewrite (if session history)
                      ↓
              BM25 search ──┐
              Vector search ──→ RRF fusion
                                    ↓
                              Reranker (optional)
                                    ↓
                          Confidence gate (0.15 / 0.35)
                                    ↓
                         Prompt build ([SOURCE N] tags)
                                    ↓
                          Groq LLM → stream tokens
                                    ↓
                       Parse [1][2] → Citation objects → SSE
```

---

## Evaluation

```bash
# Run RAGAS eval on 30 QA pairs (Phase 6)
cd backend
python scripts/eval_ragas.py --dataset tests/fixtures/eval_dataset.json

# Latency benchmark (Phase 7)
python scripts/bench_query.py --queries 100 --concurrency 5
```

---

## License

MIT — see [LICENSE](LICENSE).
