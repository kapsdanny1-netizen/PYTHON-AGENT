"""SensorQueryTool — query ``sensor_readings`` by asset_id + time window.

::

    Query:  SELECT time, channel, value FROM sensor_readings
            WHERE asset_id = :asset_id AND time BETWEEN :start AND :end
            ORDER BY time ASC

Returns the wide (pivoted) frame as compact per-channel series plus summary
statistics (mean/min/max/last) — the shape LLM agents reason best over.

Behaviour notes
---------------
* ``asset_id`` is sanitised against the fleet registry — a malicious or
  hallucinated id can never reach SQL (parameterised ORM query regardless).
* Dev convenience: when the window is empty AND ``ENVIRONMENT=dev``, the tool
  regenerates the window with the synthetic generator (incl. the anchored
  WT-07 scenario) and marks ``source="synthetic_fallback"``, so a bare
  checkout without a database still demos end-to-end. In test/prod an empty
  window is an honest error.
* The module-level :func:`fetch_sensor_frame` is reused by other tools
  (anomaly detector, prognostics) so they don't need to round-trip through
  the LLM-facing tool surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import ClassVar

import polars as pl
from pydantic import Field
from sqlalchemy import select

from config.settings import Environment, Settings
from data.generators import SCENARIO_WT07_BEARING, generate_sensor_data, get_asset
from exceptions import DatabaseError
from memory.db import SensorReading, session_scope
from tools.base import BaseToolInput, BaseToolOutput, EnergyForgeTool


class SensorQueryInput(BaseToolInput):
    """Input for SensorQueryTool."""

    asset_id: str = Field(description="Fleet asset id, e.g. WT-07")
    hours_back: float = Field(
        default=24.0, gt=0, le=24 * 90, description="Lookback window in hours"
    )
    channels: list[str] | None = Field(
        default=None, description="Subset of channels; null/omitted → all channels"
    )


class SensorQueryOutput(BaseToolOutput):
    """SensorQueryTool result (wide frame + summary stats)."""

    asset_id: str = ""
    start: str = ""
    end: str = ""
    channels: list[str] = Field(default_factory=list)
    points_per_channel: int = 0
    latest: dict[str, float] = Field(default_factory=dict)
    stats: dict[str, dict[str, float]] = Field(default_factory=dict)
    series: dict[str, list[float]] = Field(default_factory=dict)
    timestamps: list[str] = Field(default_factory=list)
    source: str = "timescaledb"


@dataclass(frozen=True)
class FetchResult:
    """Internal frame-fetch result shared between tools."""

    frame: pl.DataFrame  # wide: time + one column per channel
    source: str


async def fetch_sensor_frame(
    asset_id: str,
    *,
    hours_back: float,
    channels: list[str] | None = None,
    settings: Settings | None = None,
) -> FetchResult:
    """Load one asset's sensor window as a wide Polars frame.

    Raises:
        ValueError: unknown asset id (sanitisation).
        DatabaseError: empty window outside dev mode.
    """
    asset = get_asset(asset_id.strip().upper())
    end = datetime.now(UTC)
    start = end - timedelta(hours=hours_back)

    stmt = (
        select(SensorReading.time, SensorReading.channel, SensorReading.value)
        .where(
            SensorReading.asset_id == asset.asset_id,
            SensorReading.time >= start,
            SensorReading.time <= end,
        )
        .order_by(SensorReading.time.asc())
    )
    if channels:
        stmt = stmt.where(SensorReading.channel.in_(channels))

    records: list[tuple[datetime, str, float]] = []
    async with session_scope() as session:
        rows = await session.execute(stmt)
        records = [(t, c, v) for t, c, v in rows.all()]

    if records:
        long_df = pl.DataFrame(
            {"time": [r[0] for r in records],
             "channel": [r[1] for r in records],
             "value": [r[2] for r in records]}
        )
        wide = long_df.pivot(index="time", on="channel", values="value").sort("time")
        return FetchResult(frame=wide, source="timescaledb")

    cfg = settings or Settings()
    if cfg.environment is Environment.DEV:
        anomalies = [SCENARIO_WT07_BEARING] if asset.asset_id == SCENARIO_WT07_BEARING.asset_id else []
        frame = generate_sensor_data(asset, hours=hours_back, anomalies=anomalies)
        if channels:
            keep = ["time", *[c for c in channels if c in frame.columns]]
            frame = frame.select(keep)
        return FetchResult(frame=frame, source="synthetic_fallback")

    raise DatabaseError(
        "no sensor data in window",
        context={"asset_id": asset.asset_id, "hours_back": hours_back,
                 "hint": "run `python main.py seed-demo`"},
    )


def _channel_stats(frame: pl.DataFrame, channels: list[str]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for channel in channels:
        col = frame[channel].drop_nans()
        if col.is_empty():
            continue
        stats[channel] = {
            "mean": round(float(col.mean()), 4),  # type: ignore[arg-type]
            "min": round(float(col.min()), 4),  # type: ignore[arg-type]
            "max": round(float(col.max()), 4),  # type: ignore[arg-type]
            "last": round(float(col[-1]), 4),
        }
    return stats


class SensorQueryTool(EnergyForgeTool[SensorQueryInput, SensorQueryOutput]):
    """Query sensor history for an asset over a time window.

    Use this first: it tells you what every channel is doing and gives you
    basic statistics. For anomaly significance, follow up with the
    AnomalyDetectorTool.
    """

    name: ClassVar[str] = "sensor_query"
    description: ClassVar[str] = (
        "Query time-series sensor readings for an asset over a lookback window. "
        "Returns per-channel series (pivoted wide), latest values, and summary "
        "statistics (mean/min/max/last). Input: asset_id, hours_back (≤2160), "
        "optional channel subset."
    )
    input_model: ClassVar[type[BaseToolInput]] = SensorQueryInput
    output_model: ClassVar[type[BaseToolOutput]] = SensorQueryOutput

    async def _arun(self, tool_input: SensorQueryInput) -> SensorQueryOutput:
        asset = get_asset(tool_input.asset_id.strip().upper())
        result = await fetch_sensor_frame(
            asset.asset_id,
            hours_back=tool_input.hours_back,
            channels=tool_input.channels,
            settings=self._settings,
        )
        frame = result.frame
        channels = [c for c in frame.columns if c != "time"]
        stats = _channel_stats(frame, channels)
        times = frame["time"]
        return SensorQueryOutput(
            asset_id=asset.asset_id,
            start=str(times[0]),
            end=str(times[-1]),
            channels=channels,
            points_per_channel=frame.height,
            latest={c: stats[c]["last"] for c in stats},
            stats=stats,
            series={c: [round(float(v), 4) for v in frame[c].to_list()[:500]] for c in channels},
            timestamps=[str(t) for t in times.to_list()[:500]],
            source=result.source,
            confidence=0.99 if result.source == "timescaledb" else 0.9,
        )
