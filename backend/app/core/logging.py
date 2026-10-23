"""
Structured logging using structlog.

Dev mode: pretty-printed coloured console output.
Production: JSON output parseable by Loki, Elasticsearch, or CloudWatch.
All log entries include a request_id via contextvars.
"""

from __future__ import annotations

import logging
import sys
import uuid

import structlog
from structlog.types import EventDict, WrappedLogger


def add_request_id(
    logger: WrappedLogger,  # noqa: ARG001
    method_name: str,       # noqa: ARG001
    event_dict: EventDict,
) -> EventDict:
    """Structlog processor: inject a request_id if not already present.

    The request_id is set per-request in the FastAPI middleware and bound
    to the structlog context using structlog.contextvars.bind_contextvars().
    This processor is a safety net for log calls made outside a request context.
    """
    if "request_id" not in event_dict:
        event_dict["request_id"] = f"req-{uuid.uuid4().hex[:8]}"
    return event_dict


def setup_logging(log_level: str = "INFO", json_logs: bool = False) -> None:
    """Configure structlog and the stdlib logging bridge.

    Args:
        log_level:  One of DEBUG, INFO, WARNING, ERROR.
        json_logs:  True in production for JSON output; False in dev for
                    pretty-printed console output.
    """
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        add_request_id,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if json_logs:
        # Production: JSON output (parseable by log aggregators)
        renderer = structlog.processors.JSONRenderer()
    else:
        # Development: human-friendly coloured output
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Bridge stdlib logging (uvicorn, sqlalchemy, celery) into structlog
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level.upper())

    # Suppress noisy third-party loggers
    for noisy in ("httpx", "httpcore", "hpack", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a structlog logger bound to the given name.

    Usage:
        log = get_logger(__name__)
        log.info("chunk.indexed", doc_id=doc_id, count=42)
    """
    return structlog.get_logger(name)
