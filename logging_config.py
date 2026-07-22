"""Structured JSON logging for EnergyForge Agent.

Every event carries ``timestamp``, ``level``, ``logger``, ``event`` plus any
bound context. Use :func:`bind_log_context` so that all downstream log lines
in the current asyncio task automatically include the cross-cutting fields
required by our observability policy (``asset_id``, ``trace_id``).

Context is propagated with ``contextvars``, which is concurrency-safe across
asyncio tasks — each agent/tool/task sees only its own bound values.
"""

from __future__ import annotations

import logging
import sys
import uuid

import structlog

_configured: bool = False


def configure_logging(level: str = "INFO") -> None:
    """Configure stdlib + structlog for JSON output on stdout. Idempotent."""
    global _configured
    if _configured:
        return
    level_no = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level_no, force=True)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level_no),
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a named structlog logger, configuring logging on first use."""
    if not _configured:
        configure_logging()
    return structlog.get_logger(name)


def new_trace_id() -> str:
    """Mint a new trace id for one orchestrator turn / pipeline run."""
    return uuid.uuid4().hex


def bind_log_context(
    *,
    asset_id: str | None = None,
    trace_id: str | None = None,
    **extra: str | None,
) -> None:
    """Bind ``asset_id`` / ``trace_id`` (and any extras) to the current context.

    All log lines emitted until :func:`clear_log_context` will include these
    fields. Values set to ``None`` are skipped.
    """
    context: dict[str, str] = {}
    if asset_id is not None:
        context["asset_id"] = asset_id
    if trace_id is not None:
        context["trace_id"] = trace_id
    for key, value in extra.items():
        if value is not None:
            context[key] = value
    structlog.contextvars.bind_contextvars(**context)


def clear_log_context() -> None:
    """Drop all bound context (call at the end of a turn to avoid leakage)."""
    structlog.contextvars.clear_contextvars()
