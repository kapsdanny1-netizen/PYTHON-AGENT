"""OptimizationTool — LP set-point recommendation via ``scipy.optimize.linprog``.

Decision variables: ``x = [s, r]`` where ``s`` = set-point (MW) and
``r`` = spinning reserve offered (MW). Solved with HiGHS::

    min   −s − 0.05·r
    s.t.  s        ≤ rated · forecast_cf        (weather capability)
          s        ≤ rated · health_factor       (condition derate)
          s + r    ≤ rated · forecast_cf         (reserve from headroom)
          s        ≤ current + ramp_limit         (ramp-up)
          s        ≥ current − ramp_limit         (ramp-down)
          s ∈ [min_load, rated],  r ∈ [0, 0.2·rated]

Any input the caller omits is self-fetched so the LP stays deterministic even
when the LLM passes only an asset id: current output + forecast capacity
factor are estimated from the latest sensor readings, and the health derate
factor comes from the last 24 h of ``anomaly_events`` (LOW 1.0, MED 0.95,
HIGH 0.85, CRITICAL 0.60). Active constraints are reported by name in
``constraint_summary`` (residual < 1e-6).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import ClassVar

import numpy as np
from pydantic import Field
from sqlalchemy import select

from data.generators import AssetType, get_asset
from memory.db import AnomalyEvent, session_scope
from tools.base import BaseToolInput, BaseToolOutput, EnergyForgeTool
from tools.sensor_query import fetch_sensor_frame

_SEVERITY_DERATE = {"LOW": 1.0, "MED": 0.95, "HIGH": 0.85, "CRITICAL": 0.60}


class OptimizationInput(BaseToolInput):
    """Input for OptimizationTool. All numerics optional — see module docstring."""

    asset_id: str = Field(description="Fleet asset id, e.g. WT-07")
    forecast_cf: float | None = Field(default=None, ge=0.0, le=1.2,
                                      description="Forecast capacity factor override")
    health_factor: float | None = Field(default=None, ge=0.1, le=1.0,
                                        description="Condition derate override (1 = healthy)")
    current_output_mw: float | None = Field(default=None, ge=0.0,
                                            description="Current output override (MW)")
    min_load_pct: float = Field(default=0.30, ge=0.0, le=0.9)
    ramp_limit_pct: float = Field(default=0.50, ge=0.05, le=1.0,
                                  description="Max set-point move as fraction of rated")
    horizon_minutes: int = Field(default=60, ge=15, le=1440)


class OptimizationOutput(BaseToolOutput):
    """LP recommendation."""

    asset_id: str = ""
    recommended_setpoint_mw: float = 0.0
    reserve_mw: float = 0.0
    delta_output_mw: float = 0.0
    current_output_mw: float = 0.0
    constraint_summary: list[str] = Field(default_factory=list)
    solver_status: str = ""
    health_factor: float = 1.0
    forecast_cf: float = 0.0
    valid_minutes: int = 0
    note: str = ""


class OptimizationTool(EnergyForgeTool[OptimizationInput, OptimizationOutput]):
    """Recommend an energy set-point via linear programming.

    Combines weather capability, equipment health derating and ramp
    constraints into an LP that maximises output while optionally holding
    spinning reserve. Returns the set-point, reserve, delta vs current
    output, and which constraints were binding.
    """

    name: ClassVar[str] = "setpoint_optimizer"
    description: ClassVar[str] = (
        "Solve a linear program for the optimal power set-point of an asset, "
        "given forecast capacity factor, equipment health derate and ramp "
        "limits. Returns recommended_setpoint_mw, reserve_mw, delta_output_mw "
        "and the binding constraint summary. Input: asset_id plus optional "
        "numeric overrides (defaults auto-fetched)."
    )
    input_model: ClassVar[type[BaseToolInput]] = OptimizationInput
    output_model: ClassVar[type[BaseToolOutput]] = OptimizationOutput

    async def _estimate_current_and_cf(self, asset_id: str) -> tuple[float, float]:
        """Estimate current MW + forecast capacity factor from latest readings."""
        try:
            result = await fetch_sensor_frame(
                asset_id, hours_back=3.0, settings=self._settings
            )
            frame = result.frame
            asset = get_asset(asset_id)
            if asset.asset_type is AssetType.WIND_TURBINE and "rpm" in frame.columns:
                cf = min(1.0, float(frame["rpm"][-1]) / 18.5)
            elif asset.asset_type is AssetType.SOLAR_INVERTER and "ac_power_kw" in frame.columns:
                kw = float(frame["ac_power_kw"][-1])
                return kw / 1000.0, min(1.0, kw / 250.0)
            elif asset.asset_type is AssetType.GAS_TURBINE and "fuel_flow_kg_s" in frame.columns:
                cf = min(1.0, max(0.4, (float(frame["fuel_flow_kg_s"][-1]) - 1.55) / 1.75))
            elif asset.asset_type is AssetType.HV_TRANSFORMER and "load_pct" in frame.columns:
                load = float(frame["load_pct"][-1])
                return asset.rated_mw * load / 100.0, load / 100.0
            else:
                cf = 0.7
            return asset.rated_mw * cf * 0.95, cf
        except Exception:  # estimation is best-effort — LP overrides still work
            asset = get_asset(asset_id)
            return asset.rated_mw * 0.7, 0.7

    async def _health_factor(self, asset_id: str) -> float:
        try:
            async with session_scope() as session:
                rows = await session.execute(
                    select(AnomalyEvent.severity)
                    .where(AnomalyEvent.asset_id == asset_id,
                           AnomalyEvent.time >= datetime.now(UTC) - timedelta(hours=24))
                    .order_by(AnomalyEvent.time.desc())
                    .limit(5)
                )
                severities = [r[0] for r in rows.all()]
            return min((_SEVERITY_DERATE.get(s, 1.0) for s in severities), default=1.0)
        except Exception:
            return 1.0

    async def _arun(self, tool_input: OptimizationInput) -> OptimizationOutput:
        from scipy.optimize import linprog

        asset = get_asset(tool_input.asset_id.strip().upper())
        rated = asset.rated_mw
        current_est, cf_est = await self._estimate_current_and_cf(asset.asset_id)
        current = tool_input.current_output_mw or current_est
        forecast_cf = min(1.0, tool_input.forecast_cf if tool_input.forecast_cf is not None else cf_est)
        health = tool_input.health_factor if tool_input.health_factor is not None else await self._health_factor(asset.asset_id)

        capability = rated * forecast_cf
        derate = rated * health
        ramp = rated * tool_input.ramp_limit_pct
        min_load = rated * tool_input.min_load_pct

        a_ub = np.array([
            [1.0, 0.0],            # s ≤ capability
            [1.0, 0.0],            # s ≤ derate
            [1.0, 1.0],            # s + r ≤ capability
            [1.0, 0.0],            # s ≤ current + ramp
            [-1.0, 0.0],           # -s ≤ -(current - ramp)
        ])
        b_ub = np.array([capability, derate, capability, current + ramp, -(current - ramp)])
        result = linprog(c=np.array([-1.0, -0.05]), A_ub=a_ub, b_ub=b_ub,
                         bounds=[(min_load, rated), (0.0, 0.2 * rated)], method="highs")

        names = ["forecast_capability", "health_derate", "reserve_headroom",
                 "ramp_up_limit", "ramp_down_limit"]
        if result.success:
            setpoint, reserve = float(result.x[0]), float(result.x[1])
            residuals = b_ub - a_ub @ result.x
            summary = [f"{name} (binding)" for name, res in zip(names, residuals, strict=True)
                       if abs(res) < 1e-6]
            status, note, confidence = "optimal", "", 0.9
            if not summary:
                summary = ["objective interior — no binding constraints"]
        else:  # conservative deterministic fallback, still LP-consistent
            setpoint = float(np.clip(min(capability, derate, current + ramp), min_load, rated))
            reserve, status, confidence = 0.0, f"infeasible_fallback ({result.message})", 0.5
            summary = ["fallback: clipped min(capability, derate, current+ramp)"]
            note = "LP infeasible — conservative fallback used."

        return OptimizationOutput(
            asset_id=asset.asset_id,
            recommended_setpoint_mw=round(setpoint, 3),
            reserve_mw=round(reserve, 3),
            delta_output_mw=round(setpoint - current, 3),
            current_output_mw=round(current, 3),
            constraint_summary=summary,
            solver_status=status,
            health_factor=round(health, 3),
            forecast_cf=round(forecast_cf, 3),
            valid_minutes=tool_input.horizon_minutes,
            note=note,
            confidence=confidence,
        )
