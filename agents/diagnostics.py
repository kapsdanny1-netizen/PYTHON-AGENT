"""DiagnosticsAgent — root-cause analysis for anomalous assets.

Tools (exactly scoped): SensorQuery, AnomalyDetector, RiskCalculator.

The agent's prompt is grounded with semantically retrieved historical RCAs
from the Chroma long-term memory (best effort — absent memory never blocks
diagnosis). A deterministic calibration rule lifts confidence to ≥ 0.82 when
the ensemble detector fired HIGH/CRITICAL: at that point the *tool evidence*
already supports the conclusion and we don't let LLM modesty understate it.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from pydantic import Field

from agents.base import AgentOutputBase, AgentRunResult, run_agent
from config.settings import Settings
from logging_config import get_logger
from memory.vector_store import VectorStore
from tools import AnomalyDetectorTool, RiskCalculatorTool, SensorQueryTool

logger = get_logger(__name__)


class DiagnosisReport(AgentOutputBase):
    """DiagnosticsAgent structured output."""

    root_cause: str = Field(description="Most likely root cause of the anomaly")
    evidence: list[str] = Field(
        default_factory=list, description="Concrete, data-backed evidence items"
    )
    severity: Literal["LOW", "MED", "HIGH", "CRITICAL"]
    confidence: float = Field(ge=0.0, le=1.0)


async def _rca_context(trigger: str, settings: Settings | None) -> str:
    """Top-2 similar historical RCAs from vector memory (best effort)."""
    try:
        hits = await VectorStore(settings).query(trigger, n_results=2, where={"type": "rca"})
        if not hits:
            return "none found"
        return " | ".join(
            f"[{hit.id} sim={hit.similarity:.2f}] {hit.text[:360]}" for hit in hits
        )
    except Exception as exc:  # memory unavailable must never block diagnosis
        logger.warning("diagnostics.rca_lookup_failed", error=str(exc)[:160])
        return "memory unavailable"


async def run_diagnostics(
    *,
    asset_id: str,
    trigger: str,
    settings: Settings | None = None,
) -> AgentRunResult[DiagnosisReport]:
    """Diagnose ``asset_id`` from an operator trigger description."""
    rca_context = await _rca_context(f"{asset_id} {trigger}", settings)
    description = f"""Diagnose asset {asset_id}.

Operator report: "{trigger}"

Similar historical root-cause analyses from long-term memory:
{rca_context}

Procedure (follow strictly, be decisive):
1. Call sensor_query (asset_id={asset_id}, hours_back=24) to see current channel values.
2. Call anomaly_detector (asset_id={asset_id}, hours_back=168, recent_hours=6) for the
   ensemble anomaly score, affected channels, z-scores and severity.
3. Call risk_calculator (asset_id={asset_id}, severity from step 2) for bow-tie context.
4. Correlate the evidence with the historical RCAs above and conclude ONE root cause.

Rules: evidence items must cite actual numbers from tool outputs; severity must
match the anomaly detector unless absolute limits say otherwise; confidence
≥ 0.8 is expected when the detector severity is HIGH or CRITICAL and the pattern
matches a known RCA."""

    result = await run_agent(
        role="Senior Rotating-Equipment Diagnostician",
        goal=("Find the single most probable root cause of the anomaly on the asset, "
              "backed by sensor evidence and historical RCAs."),
        backstory=("You have 20 years diagnosing rotating machinery and power assets "
                   "from SCADA data. You distrust conclusions without numbers, and you "
                   "always check how previous identical failures presented."),
        task_description=description,
        expected_output=(
            "A DiagnosisReport JSON: root_cause (one precise failure mode), evidence "
            "(list of data-backed facts with numbers), severity (LOW/MED/HIGH/CRITICAL), "
            "confidence (0-1)."
        ),
        output_model=DiagnosisReport,
        tools=[SensorQueryTool(settings), AnomalyDetectorTool(settings), RiskCalculatorTool(settings)],
        agent_name="DiagnosticsAgent",
        settings=settings,
    )

    report = result.output
    detector_fired = any(
        r.tool == AnomalyDetectorTool.name and r.ok for r in result.trace
    )
    if detector_fired and report.severity in ("HIGH", "CRITICAL") and report.confidence < 0.82:
        # Calibration floor: strong tool evidence warrants strong confidence.
        report = report.model_copy(update={"confidence": 0.82})
    return replace(result, output=report)
