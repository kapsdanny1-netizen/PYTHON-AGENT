"""OptimizationAgent — weather-aware set-point planning.

Tools (exactly scoped): SensorQuery, Weather, Optimization.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from agents.base import AgentOutputBase, AgentRunResult, run_agent
from config.settings import Settings
from data.generators import get_asset
from tools import OptimizationTool, SensorQueryTool, WeatherTool


class OptimizationPlan(AgentOutputBase):
    """OptimizationAgent structured output."""

    asset_id: str = ""
    set_points: dict[str, float] = Field(
        default_factory=dict,
        description="Recommended set-points, e.g. {output_mw: 2.35, reserve_mw: 0.2}",
    )
    delta_mw: float = 0.0
    valid_until: datetime | None = None
    rationale: str = ""
    confidence: float = Field(ge=0.0, le=1.0)


async def run_optimization(
    *,
    asset_id: str,
    trigger: str,
    settings: Settings | None = None,
) -> AgentRunResult[OptimizationPlan]:
    """Produce an optimization plan for ``asset_id``."""
    asset = get_asset(asset_id)
    description = f"""Optimise the operating set-point of asset {asset.asset_id}
({asset.asset_type.value}, rated {asset.rated_mw} MW).

Operator request: "{trigger}"

Procedure (follow strictly):
1. Call sensor_query (asset_id={asset.asset_id}, hours_back=3) for current operating state.
2. Call weather_forecast (latitude={asset.latitude}, longitude={asset.longitude},
   hours_ahead=24) for wind/irradiance capacity factors.
3. Call setpoint_optimizer (asset_id={asset.asset_id}, forecast_cf from step 2 —
   use wind_capacity_factor for wind assets, solar_capacity_factor for solar).
4. Translate the LP result: set_points must include output_mw (recommended_setpoint_mw)
   and reserve_mw; delta_mw = delta_output_mw; valid_until = now + valid_minutes.

Cite the weather forecast and binding constraints in the rationale."""

    return await run_agent(
        role="Energy Dispatch Optimiser",
        goal=("Maximise asset output within equipment-health and ramp constraints "
              "using the weather forecast."),
        backstory=("You dispatch hybrid renewable fleets for a living. You trust "
                   "linear programming over gut feel, and you always state exactly "
                   "which constraint bound your recommendation."),
        task_description=description,
        expected_output=(
            "An OptimizationPlan JSON: asset_id, set_points (dict with output_mw and "
            "reserve_mw), delta_mw, valid_until (ISO 8601), rationale, confidence (0-1)."
        ),
        output_model=OptimizationPlan,
        tools=[SensorQueryTool(settings), WeatherTool(settings), OptimizationTool(settings)],
        agent_name="OptimizationAgent",
        settings=settings,
    )
