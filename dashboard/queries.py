"""Read-only dashboard queries (sync wrappers around the async data layer).

All coroutines run on the dashboard bridge loop — see dashboard/bridge.py.
KPI figures are transparent heuristics computed from anomaly_events and
work_orders, documented per function.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
from sqlalchemy import desc, select

from dashboard.bridge import get_bridge
from memory.db import AnomalyEvent, WorkOrder, session_scope
from tools.document_generator import REPORTS_DIR
from tools.sensor_query import fetch_sensor_frame


def fetch_frame_sync(asset_id: str, hours: float) -> tuple[pl.DataFrame, str]:
    """Wide Polars frame + source ("timescaledb" | "synthetic_fallback")."""
    result = get_bridge().run(fetch_sensor_frame(asset_id, hours_back=hours))
    return result.frame, result.source


def fetch_anomaly_events_sync(asset_id: str, hours: float) -> list[dict[str, object]]:
    """Anomaly events for the asset inside the window, newest first."""

    async def _query() -> list[dict[str, object]]:
        async with session_scope() as session:
            rows = await session.execute(
                select(AnomalyEvent)
                .where(
                    AnomalyEvent.asset_id == asset_id,
                    AnomalyEvent.time >= datetime.now(UTC) - timedelta(hours=hours),
                )
                .order_by(desc(AnomalyEvent.time))
                .limit(50)
            )
            events = rows.scalars().all()
            return [
                {"time": e.time.isoformat(), "severity": e.severity, "score": e.score,
                 "channels": ", ".join(e.channels), "description": e.description}
                for e in events
            ]

    try:
        return get_bridge().run(_query())
    except Exception:  # dashboards degrade gracefully without a database
        return []


def fetch_work_orders_sync(open_only: bool = False) -> list[dict[str, object]]:
    """Work orders, newest first (open_only filters OPEN + IN_PROGRESS)."""

    async def _query() -> list[dict[str, object]]:
        async with session_scope() as session:
            stmt = select(WorkOrder).order_by(desc(WorkOrder.created_at)).limit(100)
            if open_only:
                stmt = stmt.where(WorkOrder.status.in_(("OPEN", "IN_PROGRESS")))
            rows = await session.execute(stmt)
            orders = rows.scalars().all()
            return [
                {"wo_number": w.wo_number, "asset_id": w.asset_id, "title": w.title,
                 "priority": w.priority, "status": w.status,
                 "due_date": w.due_date.isoformat() if w.due_date else "",
                 "created_at": w.created_at.isoformat() if w.created_at else ""}
                for w in orders
            ]

    try:
        return get_bridge().run(_query())
    except Exception:
        return []


def kpi_bundle_sync(asset_id: str, hours: float) -> dict[str, object]:
    """KPI cards: availability %, MTBF, active WOs.

    Heuristics (documented for operators):
    * availability = 100 − 2.5×(#HIGH events) − 10×(#CRITICAL), floored at 0
    * MTBF (h)   = window hours ÷ max(1, HIGH+CRITICAL events)
    * active WOs = OPEN + IN_PROGRESS count for the asset (fleet-wide if empty)
    """
    events = fetch_anomaly_events_sync(asset_id, hours)
    high_plus = sum(1 for e in events if e["severity"] in ("HIGH", "CRITICAL"))
    critical = sum(1 for e in events if e["severity"] == "CRITICAL")
    availability = max(0.0, 100.0 - 2.5 * high_plus - 10.0 * critical)
    mtbf = round(hours / max(1, high_plus), 1)
    orders = fetch_work_orders_sync(open_only=True)
    asset_orders = [o for o in orders if o["asset_id"] == asset_id]
    return {
        "availability_pct": round(availability, 1),
        "mtbf_hours": mtbf,
        "active_wos": len(asset_orders),
        "events_in_window": len(events),
    }


def list_reports_sync() -> list[Path]:
    """Generated report artifacts (PDFs), newest first."""
    if not REPORTS_DIR.exists():
        return []
    return sorted(REPORTS_DIR.glob("*.pdf"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
