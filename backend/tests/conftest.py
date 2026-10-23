"""
Fetch and See — Test Fixtures (conftest.py)
=======================================
Shared pytest fixtures used by both unit and integration tests.

Key fixtures:
  settings_override  — replaces real settings with test values (in-memory DB, no Redis)
  client             — async TestClient for the FastAPI app
  mock_embedder      — returns deterministic fake embeddings (no model needed in tests)
  mock_llm           — returns a scripted LLM response (no Groq calls in tests)
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import Settings, get_settings
from app.models.db import Base


TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Override settings for tests: in-memory DB, test env, no real API keys."""
    get_settings.cache_clear()
    settings = Settings(
        APP_ENV="test",
        APP_DEBUG=True,
        SECRET_KEY="test-secret-key-32-chars-minimum!!",
        DATABASE_URL=TEST_DB_URL,
        LLM_PROVIDER="groq",
        GROQ_API_KEY="test-groq-key",
        EMBEDDING_MODEL="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        EMBEDDING_DEVICE="cpu",
        EMBEDDING_BATCH_SIZE=4,
        ENABLE_RERANKER=False,
        QDRANT_MODE="local",
        QDRANT_LOCAL_PATH=":memory:",
        REDIS_URL="redis://localhost:6379/15",  # separate DB for tests
        CACHE_TTL_SECONDS=60,
        ENABLE_VOICE=False,
        ENABLE_METRICS=False,
    )
    return settings


@pytest_asyncio.fixture(scope="function")
async def db_session(test_settings: Settings) -> AsyncSession:  # type: ignore[misc]
    """Provide an isolated in-memory DB session per test function."""
    engine = create_async_engine(TEST_DB_URL, echo=False)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with factory() as session:
        yield session
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def client(test_settings: Settings) -> AsyncClient:  # type: ignore[misc]
    """Async HTTP test client for the FastAPI app."""
    # Monkeypatch get_settings so the app uses test settings
    import app.config as config_module
    config_module.get_settings = lambda: test_settings  # type: ignore[assignment]

    from app.main import create_app
    test_app = create_app()

    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as ac:
        yield ac

    config_module.get_settings = get_settings  # restore
