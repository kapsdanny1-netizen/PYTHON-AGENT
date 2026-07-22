"""MaintenanceAgent — turns prognosis into an actionable work order.

Tools (exactly scoped): SensorQuery, Prognostics, WorkOrder.

Operational safeguard: a maintenance recommendation without a work order is
worthless, so if the LLM finishes without a valid ``wo_id`` (didn't call the
tool, or failed to echo the WO number), we deterministically create the WO
ourselves with the recommendation's title. The agent's *judgement* (priority,
timing, action) still comes from the LLM; the *bookkeeping guarantee* is code.
"""

from __future__ import annotations

from dataclasses import replace

from pydantic import Field

from agents.base import AgentOutputBase, AgentRunResult, run_agent
from config.settings import Settings
from tools import (
    PrognosticsTool,
    SensorQueryTool,
    WorkOrderInput,
    WorkOrderTool,
)


class MaintenanceRecommendation(AgentOutputBase):
    """MaintenanceAgent structured output."""

    priority: str = Field(description="LOW | MEDIUM | HIGH | URGENT")
    action: str = Field(description="Concrete recommended maintenance action")
    wo_id: str = Field(default="PENDING", description="Work order number, e.g. WO-001000")
    tta_days: float = Field(gt=0.0, description="Recommended time-to-action in days")
    confidence: float = Field(ge=0.0, le=1.0)


async def _ensure_work_order(
    recommendation: MaintenanceRecommendation, asset_id: str, diagnosis: str,
    settings: Settings | None,
) -> MaintenanceRecommendation:
    """Deterministic WO safeguard — see module docstring."""
    if recommendation.wo_id and recommendation.wo_id != "PENDING":
        return recommendation
    tool = WorkOrderTool(settings)
    created = await tool.run(WorkOrderInput(
        action="create",
        asset_id=asset_id,
        title=f"{recommendation.action[:180]}",
        description=(
            f"Auto-created by MaintenanceAgent. Diagnosis context: {diagnosis[:600]}. "
            f"Planned within {recommendation.tta_days} day(s)."
        ),
        priority=recommendation.priority if recommendation.priority in
        ("LOW", "MEDIUM", "HIGH", "URGENT") else "MEDIUM",
        due_in_days=min(recommendation.tta_days, 365.0),
    ))
    wo_id = created.wo_number if created.error is None else "PENDING"
    return recommendation.model_copy(update={"wo_id": wo_id})


async def run_maintenance(
    *,
    asset_id: str,
    trigger: str,
    diagnosis: str,
    settings: Settings | None = None,
) -> AgentRunResult[MaintenanceRecommendation]:
    """Produce a maintenance recommendation (+ guaranteed WO) for ``asset_id``."""
    description = f"""Plan maintenance for asset {asset_id}.

Operator report: "{trigger}"
Current diagnosis: {diagnosis}

Procedure (follow strictly):
1. Call prognostics (asset_id={asset_id}, hours_back=168) and read days_to_failure
   P10/P50/P90, the failure mode and the trend slope.
2. Decide the time-to-action: about 0.8 × P50, clamped to [0.5, 30] days.
3. Choose priority: URGENT if P50 ≤ 1 day, HIGH if ≤ 3, MEDIUM if ≤ 14, else LOW.
4. MUST call work_order with action="create" (asset_id={asset_id}, descriptive
   title, description citing the prognosis, chosen priority, due_in_days = tta),
   then copy the returned wo_number into wo_id.
5. Recommend ONE concrete action (what to inspect/replace and during what window).

Be decisive and cite the prognosis numbers in the action text."""

    result = await run_agent(
        role="Reliability & Maintenance Planner",
        goal=("Convert condition prognosis into the right maintenance intervention "
              "at the right time, with a work order raised."),
        backstory=("You are a former wind-farm maintenance superintendent. You plan "
                   "interventions inside the RUL window and you never let a finding "
                   "leave your desk without a work order attached."),
        task_description=description,
        expected_output=(
            "A MaintenanceRecommendation JSON: priority (LOW/MEDIUM/HIGH/URGENT), "
            "action (concrete), wo_id (WO number from the work_order tool), "
            "tta_days (>0), confidence (0-1)."
        ),
        output_model=MaintenanceRecommendation,
        tools=[SensorQueryTool(settings), PrognosticsTool(settings), WorkOrderTool(settings)],
        agent_name="MaintenanceAgent",
        max_iter=7,
        settings=settings,
    )

    recommendation = await _ensure_work_order(result.output, asset_id, diagnosis, settings)
    return replace(result, output=recommendation)
