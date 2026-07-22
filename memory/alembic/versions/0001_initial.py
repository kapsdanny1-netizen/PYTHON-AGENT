"""initial schema: sensor_readings, work_orders, anomaly_events, audit_log

Revision ID: 0001_initial
Revises:
Create Date: 2026-07-22

Notes:
* ``sensor_readings`` and ``anomaly_events`` become TimescaleDB hypertables —
  their PKs include the partition column ``time`` (TimescaleDB requirement).
* Secondary compression settings on ``sensor_readings``
  (segmented by asset/channel for chunk-local statistics).
* Work-order numbers come from the ``energyforge_wo_seq`` sequence so
  concurrent agents can never collide on a WO number.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SEVERITIES = ("LOW", "MED", "HIGH", "CRITICAL")
PRIORITIES = ("LOW", "MEDIUM", "HIGH", "URGENT")
WO_STATUSES = ("OPEN", "IN_PROGRESS", "COMPLETED", "CANCELLED")


def upgrade() -> None:
    # ── sensor_readings (hypertable) ─────────────────────────────────────
    op.create_table(
        "sensor_readings",
        sa.Column("time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("asset_id", sa.String(length=32), nullable=False),
        sa.Column("channel", sa.String(length=64), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("time", "asset_id", "channel", name="pk_sensor_readings"),
    )
    op.create_index(
        "ix_sensor_readings_asset_time", "sensor_readings", ["asset_id", "time"]
    )
    op.execute("SELECT create_hypertable('sensor_readings', 'time', if_not_exists => TRUE)")
    op.execute(
        "ALTER TABLE sensor_readings SET ("
        "timescaledb.compress, "
        "timescaledb.compress_segmentby = 'asset_id, channel')"
    )

    # ── work_orders ──────────────────────────────────────────────────────
    op.create_table(
        "work_orders",
        sa.Column(
            "id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("wo_number", sa.String(length=24), nullable=False),
        sa.Column("asset_id", sa.String(length=32), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("priority", sa.String(length=12), nullable=False, server_default="MEDIUM"),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="OPEN"),
        sa.Column("due_date", sa.Date(), nullable=True),
        sa.Column("created_by", sa.String(length=64), nullable=False,
                  server_default="energyforge-agent"),
        sa.Column("meta", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
                  nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
                  nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_work_orders"),
        sa.UniqueConstraint("wo_number", name="uq_work_orders_wo_number"),
        sa.CheckConstraint(f"priority IN {PRIORITIES!r}", name="ck_work_orders_priority"),
        sa.CheckConstraint(f"status IN {WO_STATUSES!r}", name="ck_work_orders_status"),
    )
    op.create_index("ix_work_orders_asset", "work_orders", ["asset_id"])
    op.create_index("ix_work_orders_status", "work_orders", ["status"])

    # ── anomaly_events (hypertable) ──────────────────────────────────────
    op.create_table(
        "anomaly_events",
        sa.Column(
            "id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("asset_id", sa.String(length=32), nullable=False),
        sa.Column("detector", sa.String(length=64), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("severity", sa.String(length=12), nullable=False),
        sa.Column("channels", postgresql.JSONB(), nullable=False,
                  server_default=sa.text("'[]'::jsonb")),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("trace_id", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
                  nullable=False),
        sa.PrimaryKeyConstraint("id", "time", name="pk_anomaly_events"),
        sa.CheckConstraint(f"severity IN {SEVERITIES!r}", name="ck_anomaly_events_severity"),
    )
    op.create_index("ix_anomaly_events_asset_time", "anomaly_events", ["asset_id", "time"])
    op.execute("SELECT create_hypertable('anomaly_events', 'time', if_not_exists => TRUE)")

    # ── audit_log (plain table, identity PK → monotonic order) ──────────
    op.create_table(
        "audit_log",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("time", sa.DateTime(timezone=True), server_default=sa.func.now(),
                  nullable=False),
        sa.Column("trace_id", sa.String(length=64), nullable=False),
        sa.Column("asset_id", sa.String(length=32), nullable=True),
        sa.Column("actor", sa.String(length=64), nullable=False),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False,
                  server_default=sa.text("'{}'::jsonb")),
        sa.PrimaryKeyConstraint("id", name="pk_audit_log"),
    )
    op.create_index("ix_audit_log_trace", "audit_log", ["trace_id"])
    op.create_index("ix_audit_log_asset_time", "audit_log", ["asset_id", "time"])

    # ── race-free work-order numbers ─────────────────────────────────────
    op.execute("CREATE SEQUENCE IF NOT EXISTS energyforge_wo_seq AS BIGINT START WITH 1000")


def downgrade() -> None:
    op.execute("DROP SEQUENCE IF EXISTS energyforge_wo_seq")
    op.drop_index("ix_audit_log_asset_time", table_name="audit_log")
    op.drop_index("ix_audit_log_trace", table_name="audit_log")
    op.drop_table("audit_log")
    op.drop_index("ix_anomaly_events_asset_time", table_name="anomaly_events")
    op.drop_table("anomaly_events")
    op.drop_index("ix_work_orders_status", table_name="work_orders")
    op.drop_index("ix_work_orders_asset", table_name="work_orders")
    op.drop_table("work_orders")
    op.drop_index("ix_sensor_readings_asset_time", table_name="sensor_readings")
    op.drop_table("sensor_readings")
