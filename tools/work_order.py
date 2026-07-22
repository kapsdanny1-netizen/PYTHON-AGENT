"""WorkOrderTool — create/update records in the ``work_orders`` table.

Create: allocates the human-readable WO number from the Postgres sequence
``energyforge_wo_seq`` (race-free across concurrent agents — see
:func:`memory.db.next_work_order_number`), defaults priority MEDIUM and
due date N days out. Update: status transition on an existing WO number.

All user input is validated by Pydantic (asset id against the fleet
registry, enums for priority/status/action); SQL access is ORM-only.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import ClassVar, Literal

from pydantic import Field
from sqlalchemy import select

from data.generators import get_asset
from exceptions import DatabaseError
from memory.db import WorkOrder, next_work_order_number, session_scope
from tools.base import BaseToolInput, BaseToolOutput, EnergyForgeTool

Priority = Literal["LOW", "MEDIUM", "HIGH", "URGENT"]
WoStatus = Literal["OPEN", "IN_PROGRESS", "COMPLETED", "CANCELLED"]


class WorkOrderInput(BaseToolInput):
    """Input for WorkOrderTool."""

    action: Literal["create", "update"] = Field(description="create | update")
    asset_id: str | None = Field(default=None, description="Required for create")
    title: str = Field(default="", max_length=200)
    description: str = Field(default="", max_length=4000)
    priority: Priority = "MEDIUM"
    due_in_days: float = Field(default=7.0, gt=0.0, le=365.0)
    wo_number: str | None = Field(default=None, description="Required for update, e.g. WO-001000")
    status: WoStatus | None = Field(default=None, description="New status for update")


class WorkOrderOutput(BaseToolOutput):
    """Created/updated work order."""

    action: str = ""
    wo_id: str = ""
    wo_number: str = ""
    asset_id: str = ""
    title: str = ""
    priority: str = ""
    status: str = ""
    due_date: str = ""


class WorkOrderTool(EnergyForgeTool[WorkOrderInput, WorkOrderOutput]):
    """Create or update a maintenance work order.

    Returns the human-readable WO number (from Postgres sequence) and the
    record's UUID. Use after a prognosis or diagnosis justifies intervention.
    """

    name: ClassVar[str] = "work_order"
    description: ClassVar[str] = (
        "Create or update a work order in the maintenance system. For create: "
        "asset_id, title, description, priority (LOW/MEDIUM/HIGH/URGENT), "
        "due_in_days. For update: wo_number and new status "
        "(OPEN/IN_PROGRESS/COMPLETED/CANCELLED). Returns the WO number."
    )
    input_model: ClassVar[type[BaseToolInput]] = WorkOrderInput
    output_model: ClassVar[type[BaseToolOutput]] = WorkOrderOutput

    async def _arun(self, tool_input: WorkOrderInput) -> WorkOrderOutput:
        if tool_input.action == "create":
            return await self._create(tool_input)
        return await self._update(tool_input)

    async def _create(self, tool_input: WorkOrderInput) -> WorkOrderOutput:
        if not tool_input.asset_id:
            return WorkOrderOutput(error="asset_id is required for action=create")
        if not tool_input.title.strip():
            return WorkOrderOutput(error="title is required for action=create")
        asset = get_asset(tool_input.asset_id.strip().upper())
        due = datetime.now(UTC).date() + timedelta(days=int(tool_input.due_in_days))
        async with session_scope() as session:
            wo = WorkOrder(
                wo_number=await next_work_order_number(session),
                asset_id=asset.asset_id,
                title=tool_input.title.strip(),
                description=tool_input.description.strip(),
                priority=tool_input.priority,
                status="OPEN",
                due_date=due,
                created_by="energyforge-agent",
                meta={"due_in_days": tool_input.due_in_days},
            )
            session.add(wo)
            await session.flush()
            wo_id = str(wo.id)
        return WorkOrderOutput(
            action="create", wo_id=wo_id, wo_number=wo.wo_number,
            asset_id=asset.asset_id, title=wo.title, priority=wo.priority,
            status="OPEN", due_date=due.isoformat(), confidence=0.98,
        )

    async def _update(self, tool_input: WorkOrderInput) -> WorkOrderOutput:
        if not tool_input.wo_number:
            return WorkOrderOutput(error="wo_number is required for action=update")
        if tool_input.status is None:
            return WorkOrderOutput(error="status is required for action=update")
        async with session_scope() as session:
            wo = (
                await session.execute(
                    select(WorkOrder).where(WorkOrder.wo_number == tool_input.wo_number)
                )
            ).scalars().first()
            if wo is None:
                raise DatabaseError(
                    "work order not found", context={"wo_number": tool_input.wo_number}
                )
            wo.status = tool_input.status
            await session.flush()
            return WorkOrderOutput(
                action="update", wo_id=str(wo.id), wo_number=wo.wo_number,
                asset_id=wo.asset_id, title=wo.title, priority=wo.priority,
                status=wo.status,
                due_date=wo.due_date.isoformat() if wo.due_date else "",
                confidence=0.98,
            )
