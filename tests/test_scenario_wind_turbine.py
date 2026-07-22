"""End-to-end integration scenario — WT-07 bearing wear.

Scenario (anchored in data.generators.SCENARIO_WT07_BEARING):

    "Wind turbine WT-07 shows elevated bearing vibration at 3× normal for
     6 hours. Bearing temperature rising 2°C/hr."

The full orchestrator pipeline runs against REAL infrastructure: TimescaleDB
(seeded by fixtures), real tool implementations, real CrewAI agents with
real LLM calls. Nothing internal is mocked; the diagnose→maintain→safety→
report plan performs no external calls beyond the LLM provider
(notifications render to console under ENVIRONMENT=test).

Assertions a–f per the acceptance spec, including the 90-second budget.
Runtime tip: a fast model (e.g. LLM_MODEL=gpt-4o-mini) comfortably meets the
budget; heavyweight reasoning models may not.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, select

from orchestrator import run_turn

QUERY = (
    "Wind turbine WT-07 shows elevated bearing vibration at 3× normal for "
    "6 hours. Bearing temperature rising 2°C/hr."
)

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


async def test_wt07_bearing_wear_end_to_end(
    db_ready: None, seeded_fleet: object, vector_ready: None
) -> None:
    started = time.perf_counter()
    state = await run_turn(QUERY, assets=["WT-07"])
    elapsed = time.perf_counter() - started

    outputs = state["agent_outputs"]
    for agent in ("diagnostics", "maintenance", "safety", "reporting"):
        assert agent in outputs, f"plan did not execute {agent}: {state['events']}"
        assert "error" not in outputs[agent], (
            f"{agent} failed: {outputs[agent]['error']}"
        )

    # ── (a) DiagnosticsAgent identifies bearing degradation, confidence ≥ 0.80
    diagnosis = outputs["diagnostics"]
    root_cause = str(diagnosis["root_cause"]).lower()
    assert "bearing" in root_cause, f"root cause not bearing-related: {root_cause}"
    assert any(
        term in root_cause for term in ("degrad", "wear", "spall", "damage", "defect")
    ), f"root cause lacks a degradation mode: {root_cause}"
    assert float(diagnosis["confidence"]) >= 0.80, (
        f"diagnostics confidence too low: {diagnosis['confidence']}"
    )
    assert diagnosis["severity"] in ("MED", "HIGH"), diagnosis["severity"]

    # ── (b) MaintenanceAgent creates a work order within 7 days ─────────────
    maintenance = outputs["maintenance"]
    assert str(maintenance["wo_id"]).startswith("WO-"), (
        f"no work order created: {maintenance}"
    )
    assert float(maintenance["tta_days"]) <= 7.0, (
        f"time-to-action exceeds 7 days: {maintenance['tta_days']}"
    )

    from memory.db import WorkOrder, session_scope

    async with session_scope() as session:
        stored = (
            await session.execute(
                select(WorkOrder).where(WorkOrder.wo_number == maintenance["wo_id"])
            )
        ).scalars().one()
    assert stored.asset_id == "WT-07"
    today = datetime.now(UTC).date()
    assert stored.due_date is None or stored.due_date <= today + timedelta(days=7), (
        f"WO due date {stored.due_date} is more than 7 days out (today: {today})"
    )

    # ── (c) SafetyAgent does NOT trigger emergency shutdown (severity < CRITICAL)
    safety = outputs["safety"]
    assert diagnosis["severity"] != "CRITICAL"
    assert safety.get("requires_shutdown") is False, (
        f"unexpected emergency shutdown: {safety}"
    )
    assert float(safety["risk_score"]) >= 0.0  # risk model ran

    # ── (d) ReportingAgent produces a valid PDF at the returned path ────────
    report = outputs["reporting"]
    pdf_path = Path(str(report["file_path"]))
    assert pdf_path.suffix == ".pdf" and pdf_path.exists(), (
        f"report path invalid: {report.get('file_path')}"
    )
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))  # raises on malformed PDFs
    assert len(reader.pages) >= 1
    assert (reader.pages[0].extract_text() or "").strip(), "PDF has no readable text"

    # ── (e) Entire pipeline completes in under 90 seconds (real LLM calls) ──
    assert elapsed < 90.0, f"pipeline took {elapsed:.1f}s (> 90s budget)"

    # ── (f) audit_log contains ≥ 5 entries for this run ─────────────────────
    from memory.db import AuditLog

    async with session_scope() as session:
        audit_count = await session.scalar(
            select(func.count())
            .select_from(AuditLog)
            .where(AuditLog.trace_id == state["trace_id"])
        )
    assert (audit_count or 0) >= 5, (
        f"expected ≥ 5 audit entries for trace {state['trace_id']}, got {audit_count}"
    )
