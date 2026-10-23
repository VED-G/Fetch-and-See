"""
DocMind — Application Configuration
====================================
All configuration is read from environment variables (or a .env file).
Using Pydantic-Settings means:
  - Every setting is typed and validated at startup.
  - Missing required settings fail loudly with a clear error message.
  - You can override any value with an environment variable.

Interview talking point:
  "Using Pydantic Settings is idiomatic 12-factor app design — config comes
   from the environment, not hard-coded values, so the same image runs in
   dev, staging, and prod without rebuilding."
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central settings object.  Loaded once and cached via get_settings()."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # silently ignore unknown env vars
    )

    # ── App ───────────────────────────────────────────────────────────────────
    APP_ENV: Literal["development", "production", "test"] = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    APP_DEBUG: bool = True
    SECRET_KEY: str = Field(
        default="CHANGE-ME-IN-PRODUCTION",
        description="Used for hashing API keys. Must be 32+ chars in production.",
    )

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"

    # ── LLM Provider ──────────────────────────────────────────────────────────
    LLM_PROVIDER: Literal["groq", "ollama", "openai"] = "groq"

    # Groq (Primary — free, fast, cloud)
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    GROQ_MAX_TOKENS: int = 1024
    GROQ_TEMPERATURE: float = 0.1   # Low temperature for factual RAG answers

    # Ollama (Optional local fallback)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1:8b"

    # OpenAI (Optional paid alternative)
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"

    # ── Embedding Model ───────────────────────────────────────────────────────
    # Default: small multilingual model friendly to low-end PCs (~450 MB RAM).
    # Upgrade: set EMBEDDING_MODEL=BAAI/bge-m3 if you have 8+ GB free RAM.
    EMBEDDING_MODEL: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    EMBEDDING_DEVICE: Literal["cpu", "cuda", "mps"] = "cpu"
    EMBEDDING_BATCH_SIZE: int = 16       # Keep small on low-end PCs
    EMBEDDING_DIM: int = 384             # 384 for MiniLM; 1024 for BGE-M3

    # ── Reranker ──────────────────────────────────────────────────────────────
    # Disabled by default — uses ~600 MB RAM.
    # Enable with ENABLE_RERANKER=true once you know you have headroom.
    ENABLE_RERANKER: bool = False
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANKER_TOP_K: int = 5              # Chunks fed to LLM after reranking
    RERANKER_CANDIDATE_K: int = 30       # Candidates reranked from retrieval pool

    # ── Retrieval ─────────────────────────────────────────────────────────────
    VECTOR_TOP_K: int = 20
    BM25_TOP_K: int = 20
    RRF_K: int = 60                      # Reciprocal Rank Fusion constant

    # ── Confidence Thresholds & "I Don't Know" ────────────────────────────────
    #
    # DocMind uses a two-level confidence system:
    #
    #   Score range        Behaviour
    #   ─────────────────────────────────────────────────────────────────────
    #   score < NO_ANSWER  Skip LLM.  Return structured "I don't know" with
    #                      the best retrieved chunk shown as evidence (so the
    #                      user can judge whether the question is simply out
    #                      of scope vs. a retrieval failure).
    #
    #   NO_ANSWER ≤ score  Call LLM but tag the response with
    #     < LOW_CONF       confidence="low" and a UI warning badge.
    #                      Still provides the best-effort answer.
    #
    #   score ≥ LOW_CONF   Normal confident answer with citations.
    #
    # Scores are normalised to [0, 1]:
    #   - With reranker ON:  reranker logit score after sigmoid
    #   - With reranker OFF: cosine similarity from Qdrant vector search
    #
    # Why two thresholds instead of one hard cutoff?
    #   A hard cutoff hides potentially useful answers from the user.
    #   A warning badge lets them decide whether to trust a borderline answer,
    #   which is better UX and more honest than silently discarding results.
    #
    NO_ANSWER_THRESHOLD: float = 0.15    # Below this → skip LLM entirely
    LOW_CONFIDENCE_THRESHOLD: float = 0.35  # Below this → warn but still answer

    @field_validator("NO_ANSWER_THRESHOLD", "LOW_CONFIDENCE_THRESHOLD")
    @classmethod
    def validate_thresholds(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("Confidence thresholds must be between 0 and 1 (exclusive)")
        return v

    # ── Vector Database ───────────────────────────────────────────────────────
    QDRANT_MODE: Literal["local", "server"] = "local"
    QDRANT_LOCAL_PATH: str = "./data/qdrant_storage"
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "docmind_chunks"

    # ── Redis ─────────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL_SECONDS: int = 3600
    CACHE_SIMILARITY_THRESHOLD: float = 0.97

    # ── Celery ────────────────────────────────────────────────────────────────
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # ── Chunking ──────────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    PARENT_CHUNK_SIZE: int = 1536

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    RATE_LIMIT_REQUESTS: int = 60
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    # ── Feature Flags ─────────────────────────────────────────────────────────
    ENABLE_VOICE: bool = False
    ENABLE_METRICS: bool = True

    # ── File Upload ───────────────────────────────────────────────────────────
    UPLOAD_DIR: str = "./data/uploads"
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: str = "pdf,docx,txt,md"

    @property
    def allowed_extensions_set(self) -> set[str]:
        return {ext.strip().lower() for ext in self.ALLOWED_EXTENSIONS.split(",")}

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/docmind.sqlite"

    # ── Data directories ──────────────────────────────────────────────────────
    @property
    def upload_path(self) -> Path:
        p = Path(self.UPLOAD_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def qdrant_local_path(self) -> Path:
        p = Path(self.QDRANT_LOCAL_PATH)
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance.

    Using @lru_cache means we read and validate env vars exactly once at
    startup — not on every request.  The cache is cleared in tests by calling
    get_settings.cache_clear().
    """
    return Settings()
