# DocMind — Build Progress

> One file to track every phase. Update the status emoji as each phase completes.
> Phases are cumulative — each one builds directly on the last.

---

## Status Key

| Symbol | Meaning |
|--------|---------|
| ✅ | Complete & committed |
| 🔄 | In progress |
| ⬜ | Not started |

---

## ✅ Phase 1 — Design & Stack Decision
**Commit:** `3655b63` · `1a8495e`

**Goal:** Decide every technology before writing a line of code. Document the "why" not just the "what."

| Deliverable | File |
|---|---|
| Full architecture design doc | `DESIGN.md` |
| Git repo initialised | `.gitignore`, `LICENSE`, `.env.example` |
| Makefile with dev/test/lint targets | `Makefile` |

**Key decisions made:**
- Groq as primary LLM (free, 300-500 tok/s on LPU, no GPU needed)
- Qdrant as vector DB (Rust, local-file mode = zero Docker for dev)
- MiniLM as default embedder (~450 MB RAM, 50+ languages)
- Parent-child chunking (retrieve small 512t, read large 1536t)
- Two-level confidence threshold (0.15 no-answer / 0.35 low-conf)
- BM25 + dense hybrid via RRF (no score normalisation needed)

---

## ✅ Phase 2 — Project Scaffolding
**Commits:** `479c26d` → `320a16b` → `634b0a9`

**Goal:** Full project skeleton that type-checks and lints cleanly before any feature code.

| Deliverable | File |
|---|---|
| Pinned deps, ruff + pyright config | `backend/pyproject.toml` |
| Pydantic Settings (all config typed) | `backend/app/config.py` |
| Custom exception hierarchy | `backend/app/core/exceptions.py` |
| Structured JSON logging (structlog) | `backend/app/core/logging.py` |
| SQLAlchemy async ORM models | `backend/app/models/db.py` |
| Pydantic request/response schemas | `backend/app/models/schemas.py` |
| FastAPI app factory + lifespan | `backend/app/main.py` |
| Celery app config | `backend/app/tasks/celery_app.py` |
| Health, metrics, auth, deps routers | `backend/app/api/` |
| Pytest conftest + fixtures | `backend/tests/conftest.py` |

---

## ✅ Phase 3 — Core Services (Retrieval + Ingestion + Generation)
**Commits:** `752113f` → `2d8baf6` → `e13b7b5`

**Goal:** Every non-trivial service implemented from scratch — no LangChain, no magic.

### Retrieval
| Deliverable | File |
|---|---|
| SentenceTransformer wrapper (async, threadpool) | `services/retrieval/embedder.py` |
| Qdrant async client wrapper (local + server mode) | `services/retrieval/vector_store.py` |
| BM25 index (multilingual tokeniser, incremental) | `services/retrieval/bm25_index.py` |
| Hybrid search via RRF | `services/retrieval/hybrid_search.py` |
| Cross-encoder reranker (optional, lazy-loaded) | `services/retrieval/reranker.py` |

### Ingestion
| Deliverable | File |
|---|---|
| Multi-format extractor (PDF/DOCX/TXT/MD) | `services/ingestion/extractor.py` |
| Parent-child chunker | `services/ingestion/chunker.py` |
| Pipeline orchestrator (extract→chunk→dedup→embed→store) | `services/ingestion/pipeline.py` |
| Celery async task with retry | `tasks/ingestion_tasks.py` |

### Generation
| Deliverable | File |
|---|---|
| LLM Protocol + factory (structural typing) | `services/generation/providers/base.py` |
| Groq provider (streaming SSE) | `services/generation/providers/groq.py` |
| Ollama provider (local fallback) | `services/generation/providers/ollama.py` |
| OpenAI-compatible provider | `services/generation/providers/openai_compat.py` |
| Citation-aware prompt builder | `services/generation/prompt.py` |
| Full RAG SSE pipeline (all 7 stages) | `services/generation/pipeline.py` |

### API Routers
| Deliverable | File |
|---|---|
| Documents: upload (202), job poll, list, delete, chunk preview | `api/v1/documents.py` |
| Query: streaming SSE endpoint with session persistence | `api/v1/query.py` |

---

## ✅ Phase 4 — Make It Runnable
**Commit:** `14c5e57`

**Goal:** Everything wired up end-to-end. Someone can clone → run → upload → query in under 5 minutes.

| Deliverable | File |
|---|---|
| BM25 index rebuilt from DB on startup → `app.state` | `app/main.py` (lifespan step 7) |
| Sessions router (history, delete, list) | `api/v1/sessions.py` |
| Search debug router (raw retrieval, no LLM) | `api/v1/search.py` |
| Session + search debug schemas | `models/schemas.py` |
| Docker Compose (Redis + Qdrant + API + Worker) | `docker-compose.yml` |
| Backend Dockerfile | `backend/Dockerfile` |
| First-run API key bootstrap script | `scripts/create_api_key.py` |
| README with quickstart, curl examples, config table | `README.md` |

---

## ⬜ Phase 5 — Evaluation (RAGAS)
**Goal:** Measure what you built. Produce real numbers to fill resume bullets.

| Planned Deliverable | File |
|---|---|
| RAGAS eval script (faithfulness, relevancy, precision, recall) | `scripts/eval_ragas.py` |
| Ground-truth QA fixture (30–50 hand-labelled QA pairs) | `tests/fixtures/eval_dataset.json` |
| Ablation runner (vector-only vs hybrid vs hybrid+rerank) | `scripts/run_ablation.py` |
| Latency benchmark (p50/p95/p99 per stage, 100 queries) | `scripts/bench_query.py` |
| Embedding throughput benchmark | `scripts/bench_embeddings.py` |
| Results output (ablation chart + numbers table) | `docs/eval_results.md` |

**Target metrics (fill in after running):**
- Faithfulness: `___` (hybrid+rerank) vs `___` (vector-only)
- Recall@5: `___` (hybrid+rerank) vs `___` (vector-only)
- Query p95 latency: `___ ms` (cache miss) / `___ ms` (cache hit)
- Ingestion throughput: `___ chunks/sec`

---

## ⬜ Phase 6 — Streamlit UI
**Goal:** A visual demo that makes the project instantly understandable in an interview screen share.

| Planned Deliverable | Description |
|---|---|
| Upload page | Drag-and-drop, progress bar polling job status |
| Chat page | SSE token streaming, citation cards with page numbers |
| Debug panel | Show retrieval scores, reranker scores, confidence meter |
| Eval dashboard | Display RAGAS metrics from Phase 5 |
| Latency panel | Live p50/p95 chart from `/metrics` |

---

## ⬜ Phase 7 — Polish & Push
**Goal:** Production-ready repo that a hiring manager can open, understand, and run in 10 minutes.

| Planned Deliverable | Description |
|---|---|
| Push to GitHub (`VED-G/docmind`) | Public repo |
| GitHub Actions CI | ruff lint + pyright + pytest on every push |
| `docs/` folder | Architecture diagram (Mermaid), eval results, benchmark charts |
| Fill resume bullets | Use real Phase 5 numbers |
| Update `INTERVIEW.md` | Add Phase 5/6/7 questions |

---

## Overall Progress

```
Phase 1 ████████████████████ 100%  Design & Stack
Phase 2 ████████████████████ 100%  Scaffolding
Phase 3 ████████████████████ 100%  Core Services
Phase 4 ████████████████████ 100%  Runnable
Phase 5 ░░░░░░░░░░░░░░░░░░░░   0%  Evaluation
Phase 6 ░░░░░░░░░░░░░░░░░░░░   0%  UI
Phase 7 ░░░░░░░░░░░░░░░░░░░░   0%  Polish & Push
```

**Total: 4 / 7 phases complete (~57%)**

---

## Git Log Reference

```
14c5e57  phase 4 — runnable
3bcdbd6  refactor — clean code files, INTERVIEW.md
e13b7b5  phase 3 — generation layer + API routers
2d8baf6  phase 3 — ingestion pipeline + Celery task
752113f  phase 3 — retrieval services
634b0a9  phase 2 — DB models + schemas
320a16b  phase 2 — FastAPI factory + routers + config
4c5ec20  phase 2 — exceptions + logging
06f3daf  phase 2 — Pydantic settings
479c26d  phase 2 — pyproject.toml
1a8495e  phase 1 — Makefile + data dirs
3655b63  phase 1 — init + gitignore + LICENSE + .env.example
```
