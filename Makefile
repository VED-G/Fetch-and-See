# ═══════════════════════════════════════════════════════════════════════════════
# DocMind Makefile
# Usage: make <target>
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: help install dev test lint format typecheck bench eval seed \
        docker-up docker-down clean setup-env

BACKEND_DIR := backend
PYTHON      := python

help:  ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ─── Setup ────────────────────────────────────────────────────────────────────

setup-env:  ## Copy .env.example to .env (first-time setup)
	@if not exist .env (copy .env.example .env && echo .env created — fill in your API keys) else (echo .env already exists)

install:  ## Install Python dependencies (run inside backend/)
	cd $(BACKEND_DIR) && pip install -e ".[dev]"

# ─── Development ──────────────────────────────────────────────────────────────

dev:  ## Start API server with hot reload
	cd $(BACKEND_DIR) && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

worker:  ## Start Celery ingestion worker
	cd $(BACKEND_DIR) && celery -A app.tasks.celery_app worker -Q ingestion --loglevel=info

ui:  ## Start Streamlit UI
	cd frontend && streamlit run app.py --server.port 8501

seed:  ## Load sample documents into the running API
	cd $(BACKEND_DIR) && $(PYTHON) scripts/seed_documents.py

create-key:  ## Create an admin API key (needs running server)
	cd $(BACKEND_DIR) && $(PYTHON) scripts/create_api_key.py

# ─── Quality ──────────────────────────────────────────────────────────────────

lint:  ## Run ruff linter
	cd $(BACKEND_DIR) && ruff check app/ tests/

format:  ## Run ruff formatter (auto-fix)
	cd $(BACKEND_DIR) && ruff format app/ tests/

typecheck:  ## Run pyright type checker
	cd $(BACKEND_DIR) && pyright app/

# ─── Tests ────────────────────────────────────────────────────────────────────

test:  ## Run all tests with coverage
	cd $(BACKEND_DIR) && pytest tests/ -v --cov=app --cov-report=term-missing

test-unit:  ## Run unit tests only
	cd $(BACKEND_DIR) && pytest tests/unit/ -v

test-integration:  ## Run integration tests only (needs running services)
	cd $(BACKEND_DIR) && pytest tests/integration/ -v

# ─── Evaluation ───────────────────────────────────────────────────────────────

eval:  ## Run RAGAS evaluation on the eval set
	cd $(BACKEND_DIR) && $(PYTHON) ../evaluation/run_eval.py

ablation:  ## Run retrieval ablation study
	cd $(BACKEND_DIR) && $(PYTHON) ../evaluation/ablations.py

# ─── Benchmarks ───────────────────────────────────────────────────────────────

bench-ingest:  ## Benchmark ingestion throughput
	cd $(BACKEND_DIR) && $(PYTHON) ../benchmarks/bench_ingestion.py

bench-query:  ## Benchmark query latency (p50/p95/p99)
	cd $(BACKEND_DIR) && $(PYTHON) ../benchmarks/bench_query.py

bench-embed:  ## Compare embedding model quantizations
	cd $(BACKEND_DIR) && $(PYTHON) ../benchmarks/bench_embeddings.py

bench-load:  ## Run locust concurrent load test (headless, 60s)
	locust -f benchmarks/locustfile.py --headless -u 20 -r 2 -t 60s \
	  --host http://localhost:8000 --html benchmarks/results/concurrent_load.html

# ─── Docker ───────────────────────────────────────────────────────────────────

docker-up:  ## Start all services via Docker Compose
	docker compose up -d

docker-down:  ## Stop all Docker Compose services
	docker compose down

docker-logs:  ## Tail Docker logs
	docker compose logs -f

# ─── Cleanup ──────────────────────────────────────────────────────────────────

clean:  ## Remove caches and generated files
	cd $(BACKEND_DIR) && \
	  $(PYTHON) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['__pycache__','.pytest_cache','.ruff_cache','.mypy_cache','htmlcov']]"
	@echo Cleaned.
