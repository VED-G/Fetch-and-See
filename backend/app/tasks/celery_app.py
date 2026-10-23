"""
DocMind — Celery Application
==============================
Celery handles async document ingestion so the API doesn't block.

Why Celery + Redis?
  PDF parsing + chunking + embedding a 50-page document takes 30-90 seconds
  on a low-end CPU.  A synchronous upload endpoint would hold the HTTP
  connection open that whole time — terrible UX and resource waste.

  With Celery:
    1. API receives file → saves to disk → dispatches Celery task → returns 202
    2. Celery worker runs ingestion in background
    3. Client polls GET /api/v1/documents/jobs/{job_id} to check progress

Interview talking point:
  "I chose Celery over Python's asyncio.create_task() for ingestion because
   Celery tasks survive worker restarts (stored in Redis), can be retried
   automatically on failure, and can be distributed across multiple workers
   if throughput needs to scale."
"""

from __future__ import annotations

from celery import Celery

from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "docmind",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.tasks.ingestion_tasks"],
)

celery_app.conf.update(
    # Serialization — JSON is safe and human-readable
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Timezone
    timezone="UTC",
    enable_utc=True,
    # Retry behavior
    task_acks_late=True,               # Acknowledge only after task completes (safer)
    task_reject_on_worker_lost=True,   # Re-queue if worker dies mid-task
    worker_prefetch_multiplier=1,      # One task at a time per worker (CPU-heavy tasks)
    # Result expiry (48 hours — enough time to poll job status)
    result_expires=172_800,
    # Routing
    task_routes={
        "app.tasks.ingestion_tasks.ingest_document": {"queue": "ingestion"},
    },
)
