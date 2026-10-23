"""
Celery application for async document ingestion.

Ingestion tasks are dispatched here so the upload endpoint returns 202 immediately.
Redis is the broker; tasks survive worker restarts and are retried on failure.
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
