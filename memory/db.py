"""Async SQLAlchemy data layer targeting TimescaleDB (PostgreSQL 16).

Design decisions
----------------
* **Narrow (long) schema** for ``sensor_readings`` — one row per
  (time, asset_id, channel) — so a single hypertable serves all four asset
  types without nullable per-type columns. Readers pivot back to wide form
  (see data/generators and tools/sensor_query).
* ``sensor_readings`` and ``anomaly_events`` are TimescaleDB **hypertables**;
  their primary keys therefore include the partition column (``time``).
* ``work_orders.wo_number`` comes from the Postgres sequence
  ``energyforge_wo_seq`` via :func:`next_work_order_number` — race-free under
  concurrent agents, unlike count-based generation.
* All access goes through :func:`session_scope`, which wraps SQLAlchemy
  errors in :class:`~exceptions.DatabaseError` so callers see only the
  EnergyForge exception hierarchy.

Apply the schema with::

    python main.py migrate        # or: alembic upgrade head
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path

from sqlalchemy import (
    JSON,
    BigInteger,
    Date,
    DateTime,
    Float,
    Index,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from config.settings import Settings, get_settings
from exceptions import DatabaseError
from logging_config import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORK_ORDER_SEQUENCE = "energyforge_wo_seq"


class Base(DeclarativeBase):
    """Declarative base for all EnergyForge ORM models."""


class SensorReading(Base):
    """One sensor value at one instant for one asset channel.

    Hypertable on ``time``; the composite PK satisfies TimescaleDB's rule
    that the partition column appears in every unique constraint.
    """

    __tablename__ = "sensor_readings"

    time: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    asset_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    channel: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (Index("ix_sensor_readings_asset_time", "asset_id", "time"),)


class AnomalyEvent(Base):
    """A detected anomaly with severity and affected channels.

    Hypertable on ``time`` (PK includes ``time`` for the same reason as
    :class:`SensorReading`).
    """

    __tablename__ = "anomaly_events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    time: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    asset_id: Mapped[str] = mapped_column(String(32), nullable=False)
    detector: Mapped[str] = mapped_column(String(64), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    severity: Mapped[str] = mapped_column(String(12), nullable=False)
    channels: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    trace_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (Index("ix_anomaly_events_asset_time", "asset_id", "time"),)


class WorkOrder(Base):
    """A maintenance work order created by the MaintenanceAgent."""

    __tablename__ = "work_orders"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    wo_number: Mapped[str] = mapped_column(String(24), nullable=False, unique=True)
    asset_id: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    priority: Mapped[str] = mapped_column(String(12), nullable=False, default="MEDIUM")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="OPEN")
    due_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    created_by: Mapped[str] = mapped_column(String(64), nullable=False, default="energyforge-agent")
    meta: Mapped[dict[str, object]] = mapped_column(
        "meta", JSONB, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_work_orders_asset", "asset_id"),
        Index("ix_work_orders_status", "status"),
    )


class AuditLog(Base):
    """Append-only audit trail — one row per orchestrator node execution.

    Identity PK gives monotonic insertion order (handy for "≥ N entries
    for this run" assertions in integration tests).
    """

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    trace_id: Mapped[str] = mapped_column(String(64), nullable=False)
    asset_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    actor: Mapped[str] = mapped_column(String(64), nullable=False)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, object]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), nullable=False, default=dict
    )

    __table_args__ = (
        Index("ix_audit_log_trace", "trace_id"),
        Index("ix_audit_log_asset_time", "asset_id", "time"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Engine / session management (per-event-loop, lazily constructed)
#
# asyncpg connections are bound to the event loop that created them. CrewAI
# executes tools in worker threads (each with its own loop) and pytest-asyncio
# gives every test a fresh loop — so ONE global engine would poison subsequent
# loops with "attached to a different loop" failures. We therefore key the
# engine/session-factory caches by running loop identity.
# ─────────────────────────────────────────────────────────────────────────────

_engines: dict[int, tuple[asyncio.AbstractEventLoop, AsyncEngine]] = {}
_session_factories: dict[int, async_sessionmaker[AsyncSession]] = {}


def _running_loop() -> asyncio.AbstractEventLoop:
    try:
        return asyncio.get_running_loop()
    except RuntimeError as exc:
        raise DatabaseError("get_engine() requires a running event loop") from exc


def get_engine(settings: Settings | None = None) -> AsyncEngine:
    """Return the async engine for the CURRENT event loop (created on demand).

    The URL embeds credentials — never log it (we only ever log host/db).
    """
    loop = _running_loop()
    key = id(loop)
    entry = _engines.get(key)
    if entry is None or entry[0] is not loop:
        cfg = settings or get_settings()
        engine = create_async_engine(
            cfg.database_url,
            pool_size=cfg.db_pool_size,
            max_overflow=cfg.db_pool_max_overflow,
            pool_pre_ping=True,
            echo=False,
        )
        _engines[key] = (loop, engine)
        logger.info(
            "db.engine_created",
            host=cfg.postgres_host,
            port=cfg.postgres_port,
            database=cfg.postgres_db,
        )
    return _engines[key][1]


def get_session_factory(settings: Settings | None = None) -> async_sessionmaker[AsyncSession]:
    """Return the async session factory for the CURRENT event loop."""
    loop = _running_loop()
    key = id(loop)
    factory = _session_factories.get(key)
    if factory is None:
        factory = async_sessionmaker(
            bind=get_engine(settings),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
        _session_factories[key] = factory
    return factory


async def dispose_engine() -> None:
    """Dispose the engine bound to the CURRENT loop — test hygiene helper."""
    loop = _running_loop()
    key = id(loop)
    entry = _engines.pop(key, None)
    _session_factories.pop(key, None)
    if entry is not None:
        await entry[1].dispose()


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Transactional session scope: commit on success, rollback on error.

    SQLAlchemy errors are re-raised as :class:`DatabaseError`; all other
    exceptions roll back too and propagate unchanged.

    Yields:
        An active :class:`AsyncSession`.
    """
    session = get_session_factory()()
    try:
        yield session
        await session.commit()
    except SQLAlchemyError as exc:
        await session.rollback()
        raise DatabaseError(
            "database operation failed",
            context={"sqlalchemy_error": type(exc).__name__, "detail": str(exc)[:500]},
        ) from exc
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def record_audit(
    session: AsyncSession,
    *,
    trace_id: str,
    actor: str,
    action: str,
    asset_id: str | None = None,
    payload: dict[str, object] | None = None,
) -> AuditLog:
    """Append one audit row (caller owns the transaction/commit)."""
    entry = AuditLog(
        trace_id=trace_id,
        actor=actor,
        action=action,
        asset_id=asset_id,
        payload=dict(payload or {}),
    )
    session.add(entry)
    await session.flush()
    logger.debug("audit.recorded", actor=actor, action=action, asset_id=asset_id)
    return entry


async def next_work_order_number(session: AsyncSession) -> str:
    """Return the next human-readable WO number (race-free).

    Uses the ``energyforge_wo_seq`` sequence created by migration 0001 —
    a static identifier, no user input, so plain SQL text is safe here.
    """
    value = await session.scalar(text(f"SELECT nextval('{WORK_ORDER_SEQUENCE}')"))
    if value is None:  # pragma: no cover - defensive
        raise DatabaseError("work-order sequence returned NULL")
    return f"WO-{int(value):06d}"


async def health_check() -> bool:
    """Probe TimescaleDB with ``SELECT 1``; raise DatabaseError on failure."""
    try:
        async with get_engine().connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # broad: infra probe must surface any failure mode
        raise DatabaseError(
            "TimescaleDB health check failed",
            context={"error": type(exc).__name__, "detail": str(exc)[:300]},
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Alembic bootstrap (so the demo/CLI can self-provision the schema)
# ─────────────────────────────────────────────────────────────────────────────


def _upgrade_head_sync() -> None:
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    command.upgrade(cfg, "head")


async def run_migrations() -> None:
    """Apply Alembic migrations (``upgrade head``) without blocking the loop."""
    await asyncio.to_thread(_upgrade_head_sync)
    logger.info("db.migrations_applied", head="0001_initial")
