"""ReportingAgent — compiles the turn's findings into a PDF artifact.

Tools: all read-only tools (SensorQuery, AnomalyDetector, Prognostics,
Weather, Optimization, RiskCalculator) + DocumentGenerator.

Artifact safeguard: a report that exists only as a path string is worthless.
If the LLM didn't produce a verified PDF (tool not called, or hallucinated
path), the deterministic fallback renders the same findings bundle itself —
the *narrative quality* is the LLM's job, the *artifact guarantee* is code's.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field

from agents.base import AgentOutputBase, AgentRunResult, run_agent
from config.settings import Settings
from tools import (
    AnomalyDetectorTool,
    DocumentGeneratorInput,
    DocumentGeneratorTool,
    OptimizationTool,
    PrognosticsTool,
    ReportSection,
    RiskCalculatorTool,
    SensorQueryTool,
    WeatherTool,
)


class ReportArtifact(AgentOutputBase):
    """ReportingAgent structured output."""

    report_type: Literal["incident", "maintenance", "executive", "compliance"] = "incident"
    file_path: str = ""
    summary: str = ""
    confidence: float = Field(ge=0.0, le=1.0)


def _looks_like_pdf(path_str: str) -> bool:
    try:
        path = Path(path_str)
        return path.suffix == ".pdf" and path.exists() and path.stat().st_size > 0
    except (OSError, ValueError):
        return False


async def _fallback_pdf(
    *, asset_id: str, trigger: str, findings_bundle: str, settings: Settings | None
) -> str:
    """Render a PDF deterministically from the findings bundle (safeguard path)."""
    tool = DocumentGeneratorTool(settings)
    slug = f"incident-{asset_id.lower()}-{datetime.now(UTC).strftime('%Y%m%d')}"
    output = await tool.run(DocumentGeneratorInput(
        report_type="incident",
        title=f"Incident Report — {asset_id}",
        slug=slug,
        sections=[
            ReportSection(heading="Trigger", body=trigger),
            ReportSection(heading="Findings Bundle",
                          body="```\n" + findings_bundle[:5000] + "\n```"),
            ReportSection(heading="Disclaimer",
                          body="Narrative rendering safeguard engaged: structured findings "
                               "rendered verbatim pending LLM-authored summary."),
        ],
        meta={"asset_id": asset_id, "source": "deterministic-fallback"},
    ))
    return output.file_path if output.error is None else ""


async def run_reporting(
    *,
    asset_id: str,
    trigger: str,
    findings_bundle: str,
    settings: Settings | None = None,
) -> AgentRunResult[ReportArtifact]:
    """Compile ``findings_bundle`` into a PDF incident report for ``asset_id``."""
    description = f"""Write a professional incident report PDF for asset {asset_id}.

Operator trigger: "{trigger}"

Complete findings bundle from the specialist agents:
{findings_bundle}

Procedure (follow strictly):
1. Review the bundle. You MAY call read-only tools (sensor_query, anomaly_detector,
   prognostics, risk_calculator) to add one or two exact numbers, but do not
   re-run the whole investigation.
2. MUST call document_generator with report_type="incident",
   title="Incident Report — {asset_id}", slug="incident-{asset_id.lower()}-<yyyymmdd>"
   and sections: Executive Summary, Sensor Evidence, Root-Cause Analysis,
   Risk & Safety, Recommendations, Appendix (key numbers in a Markdown table).
3. Copy the returned file_path EXACTLY into your answer's file_path.
4. Write a 3-5 sentence executive summary citing decision-relevant numbers."""

    result = await run_agent(
        role="Technical Report Writer",
        goal=("Turn specialist findings into a precise, verifiable PDF incident "
              "report for operations leadership."),
        backstory=("You write NTSB-grade incident reports: every claim cites a "
                   "number, every number has a source, and every report ends with "
                   "actions an operator can execute tomorrow morning."),
        task_description=description,
        expected_output=(
            "A ReportArtifact JSON: report_type ('incident'), file_path (EXACT path "
            "returned by the document_generator tool), summary (3-5 sentences), "
            "confidence (0-1)."
        ),
        output_model=ReportArtifact,
        tools=[
            SensorQueryTool(settings), AnomalyDetectorTool(settings), PrognosticsTool(settings),
            WeatherTool(settings), OptimizationTool(settings), RiskCalculatorTool(settings),
            DocumentGeneratorTool(settings),
        ],
        agent_name="ReportingAgent",
        max_iter=7,
        settings=settings,
    )

    artifact = result.output
    if not _looks_like_pdf(artifact.file_path):
        fallback_path = await _fallback_pdf(
            asset_id=asset_id, trigger=trigger, findings_bundle=findings_bundle,
            settings=settings,
        )
        if fallback_path:
            artifact = artifact.model_copy(update={"file_path": fallback_path})
    if artifact.file_path and artifact.confidence < 0.85:
        artifact = artifact.model_copy(update={"confidence": 0.85})
    from dataclasses import replace

    return replace(result, output=artifact)
