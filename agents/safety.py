"""SafetyAgent — independent safety review of a diagnosed condition.

Tools (exactly scoped): SensorQuery, RiskCalculator, Notification.

Shutdown doctrine (deterministic, not negotiable by the LLM): an emergency
shutdown is only ever set when severity is CRITICAL or an absolute safety
limit is breached. Anything below CRITICAL yields ``requires_shutdown=False``
enforced by code after the LLM run — the SafetyAgent escalates through
barriers and mitigations instead of tripping healthy-but-degrading assets.
"""

from __future__ import annotations

from dataclasses import replace

from pydantic import Field

from agents.base import AgentOutputBase, AgentRunResult, run_agent
from config.settings import Settings
from tools import NotificationTool, RiskCalculatorTool, SensorQueryTool


class SafetyAssessment(AgentOutputBase):
    """SafetyAgent structured output."""

    violations: list[str] = Field(default_factory=list,
                                  description="Safety violations identified, if any")
    risk_score: float = Field(ge=0.0, le=100.0)
    immediate_actions: list[str] = Field(default_factory=list)
    requires_shutdown: bool = False
    confidence: float = Field(ge=0.0, le=1.0)


async def run_safety(
    *,
    asset_id: str,
    trigger: str,
    diagnosis: str,
    severity: str,
    settings: Settings | None = None,
) -> AgentRunResult[SafetyAssessment]:
    """Assess immediate safety implications for ``asset_id``."""
    severity = severity.strip().upper()
    description = f"""Perform an independent safety review for asset {asset_id}.

Operator report: "{trigger}"
Diagnosis under review: {diagnosis}
Declared severity: {severity}

Procedure (follow strictly):
1. Call sensor_query (asset_id={asset_id}, hours_back=6) and compare latest values
   against absolute limits (bearing_temp_c ≥ 80°C, oil_temp_c ≥ 75°C,
   exhaust_temp_c ≥ 565°C, efficiency ≤ 0.90).
2. Call risk_calculator (asset_id={asset_id}, severity="{severity}") for the
   bow-tie risk score, top barriers and recommended mitigations.
3. Set violations ONLY for actual limit breaches or failed barriers.
4. Set immediate_actions from the risk tool's mitigations, ordered by urgency.
5. Emergency shutdown doctrine: requires_shutdown is allowed ONLY when severity
   is CRITICAL or an absolute limit is breached. If (and only if) you set it,
   you MUST also call the notification tool with CRITICAL severity.

Copy risk_score verbatim from the risk_calculator output."""

    result = await run_agent(
        role="HSE Barrier Analyst",
        goal=("Verify that diagnosed conditions are safe to run with, and specify "
              "barrier-restoring immediate actions."),
        backstory=("You are a process-safety engineer who thinks in bow-ties. You "
                   "shut down only on hard evidence — an unnecessary trip is itself "
                   "a safety incident — but you never wave through a breached limit."),
        task_description=description,
        expected_output=(
            "A SafetyAssessment JSON: violations (list, possibly empty), risk_score "
            "(0-100 from the risk tool), immediate_actions (ordered list), "
            "requires_shutdown (true ONLY if CRITICAL or absolute breach), confidence (0-1)."
        ),
        output_model=SafetyAssessment,
        tools=[SensorQueryTool(settings), RiskCalculatorTool(settings), NotificationTool(settings)],
        agent_name="SafetyAgent",
        settings=settings,
    )

    assessment = result.output
    if severity != "CRITICAL" and assessment.requires_shutdown:
        # Doctrine override — see module docstring.
        assessment = assessment.model_copy(update={"requires_shutdown": False})
    return replace(result, output=assessment)
