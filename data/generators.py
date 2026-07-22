"""Synthetic sensor data generators for the EnergyForge demo fleet.

Four asset types with physically-plausible baseline signals (diurnal cycles,
AR(1) autocorrelated noise, cross-channel coupling):

==================  ============================================================
Asset type          Channels
==================  ============================================================
wind_turbine        rpm, vibration_x/y/z (mm/s), bearing_temp_c, blade_pitch_deg
solar_inverter      dc_voltage_v, ac_power_kw, efficiency (0-1), string_current_a
gas_turbine         exhaust_temp_c, fuel_flow_kg_s, compressor_pressure_bar
hv_transformer      oil_temp_c, load_pct
==================  ============================================================

Anomaly injection (``AnomalySpec.kind`` semantics):

* ``BEARING_WEAR``  — vibration channels multiplied up to ``magnitude`` over
  ``ramp_hours`` then held; ``bearing_temp_c`` rises ``rate`` °C per active hour.
* ``FOULING_DEGRADATION`` — primary performance channel decays ``rate`` per day
  (fractional) with coupled secondary channels (fuel flow up, exhaust up, …).
* ``STEP_CHANGE``   — channel jumps by ``magnitude`` (absolute units) at the
  anomaly start and stays.

Output is a Polars DataFrame sorted by ``time`` (Polars is index-free; the
timezone-aware UTC ``time`` column is the first column and acts as the index)
— plus :func:`store_sensor_dataframe` / :func:`generate_and_store` for DB
persistence. Determinism: everything derives from ``Settings.random_seed``
(override per call); per-asset seeds are derived via CRC32 so results are
stable across processes.

Run directly to seed the demo database::

    python -m data.generators
"""

from __future__ import annotations

import asyncio
import zlib
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TypeAlias

import numpy as np
import polars as pl
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from config.settings import get_settings
from logging_config import bind_log_context, clear_log_context, get_logger, new_trace_id
from memory.db import SensorReading, session_scope

logger = get_logger(__name__)

FloatArray: TypeAlias = NDArray[np.float64]

# ─────────────────────────────────────────────────────────────────────────────
# Fleet registry
# ─────────────────────────────────────────────────────────────────────────────


class AssetType(StrEnum):
    WIND_TURBINE = "wind_turbine"
    SOLAR_INVERTER = "solar_inverter"
    GAS_TURBINE = "gas_turbine"
    HV_TRANSFORMER = "hv_transformer"


class AssetSpec(BaseModel):
    """Static description of a fleet asset."""

    model_config = ConfigDict(frozen=True)

    asset_id: str = Field(pattern=r"^[A-Z0-9][A-Z0-9-]{1,31}$")
    asset_type: AssetType
    rated_mw: float = Field(gt=0)  # MVA for transformers
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


CHANNELS: dict[AssetType, tuple[str, ...]] = {
    AssetType.WIND_TURBINE: (
        "rpm",
        "vibration_x",
        "vibration_y",
        "vibration_z",
        "bearing_temp_c",
        "blade_pitch_deg",
    ),
    AssetType.SOLAR_INVERTER: ("dc_voltage_v", "ac_power_kw", "efficiency", "string_current_a"),
    AssetType.GAS_TURBINE: ("exhaust_temp_c", "fuel_flow_kg_s", "compressor_pressure_bar"),
    AssetType.HV_TRANSFORMER: ("oil_temp_c", "load_pct"),
}

# Demo site: Lagos, NG — small per-asset coordinate offsets.
_SITE_LAT, _SITE_LON = 6.5244, 3.3792
FLEET: dict[str, AssetSpec] = {
    **{
        f"WT-{i:02d}": AssetSpec(
            asset_id=f"WT-{i:02d}",
            asset_type=AssetType.WIND_TURBINE,
            rated_mw=3.0,
            latitude=_SITE_LAT + 0.002 * i,
            longitude=_SITE_LON + 0.003 * i,
        )
        for i in range(1, 9)
    },
    **{
        f"PV-{i:02d}": AssetSpec(
            asset_id=f"PV-{i:02d}",
            asset_type=AssetType.SOLAR_INVERTER,
            rated_mw=0.25,
            latitude=_SITE_LAT - 0.004 * i,
            longitude=_SITE_LON - 0.002 * i,
        )
        for i in range(1, 5)
    },
    **{
        f"GT-{i:02d}": AssetSpec(
            asset_id=f"GT-{i:02d}",
            asset_type=AssetType.GAS_TURBINE,
            rated_mw=42.0,
            latitude=_SITE_LAT + 0.006 * i,
            longitude=_SITE_LON - 0.004 * i,
        )
        for i in range(1, 3)
    },
    **{
        f"TR-{i:02d}": AssetSpec(
            asset_id=f"TR-{i:02d}",
            asset_type=AssetType.HV_TRANSFORMER,
            rated_mw=60.0,
            latitude=_SITE_LAT + 0.001 * i,
            longitude=_SITE_LON + 0.001 * i,
        )
        for i in range(1, 3)
    },
}


def get_asset(asset_id: str) -> AssetSpec:
    """Look up an asset in the fleet registry (input sanitisation point).

    Raises:
        ValueError: if the id is unknown — callers must never pass raw,
            unsanitised ids into SQL or prompts.
    """
    normalised = asset_id.strip().upper()
    try:
        return FLEET[normalised]
    except KeyError:
        valid = ", ".join(sorted(FLEET))
        raise ValueError(f"unknown asset_id {asset_id!r} — valid ids: {valid}") from None


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly specifications
# ─────────────────────────────────────────────────────────────────────────────


class AnomalyKind(StrEnum):
    BEARING_WEAR = "bearing_wear"
    FOULING_DEGRADATION = "fouling_degradation"
    STEP_CHANGE = "step_change"


class AnomalySpec(BaseModel):
    """Declarative anomaly injection.

    Attributes:
        kind: See module docstring for per-kind semantics.
        asset_id: Target asset (must exist in :data:`FLEET`).
        start_offset_hours: Anomaly starts this many hours before window end.
        ramp_hours: Taper-in duration (full effect reached after this).
        magnitude: Peak vibration multiplier (bearing), absolute jump (step),
            unused by fouling.
        rate: Bearing temperature rise °C/hr; fouling fractional decay/day.
        hold: Hold at peak after ramp (False → triangular decay back to normal).
        channels: Explicit channel override; defaults per kind/asset if None.
    """

    model_config = ConfigDict(frozen=True)

    kind: AnomalyKind
    asset_id: str = Field(pattern=r"^[A-Z0-9][A-Z0-9-]{1,31}$")
    start_offset_hours: float = Field(default=48.0, ge=0.0)
    ramp_hours: float = Field(default=6.0, gt=0.0)
    magnitude: float = Field(default=3.0, gt=0.0)
    rate: float = Field(default=0.0, ge=0.0)
    hold: bool = True
    channels: tuple[str, ...] | None = None


# Anchored Phase 6 scenario preset — single source of truth for the
# WT-07 integration test ("vibration 3× normal for 6 h, temp +2 °C/hr").
SCENARIO_WT07_BEARING = AnomalySpec(
    kind=AnomalyKind.BEARING_WEAR,
    asset_id="WT-07",
    start_offset_hours=6.0,
    ramp_hours=1.0,
    magnitude=3.0,
    rate=2.0,
    hold=True,
)

_BEARING_CHANNELS = ("vibration_x", "vibration_y", "vibration_z")
_FOULING_PRIMARY = {
    AssetType.SOLAR_INVERTER: "efficiency",
    AssetType.GAS_TURBINE: "compressor_pressure_bar",
}
_FOULING_DEFAULT_RATE = {AssetType.SOLAR_INVERTER: 0.03, AssetType.GAS_TURBINE: 0.02}
_STEP_DEFAULT = {
    AssetType.WIND_TURBINE: "bearing_temp_c",
    AssetType.SOLAR_INVERTER: "string_current_a",
    AssetType.GAS_TURBINE: "exhaust_temp_c",
    AssetType.HV_TRANSFORMER: "oil_temp_c",
}

# ─────────────────────────────────────────────────────────────────────────────
# Signal construction
# ─────────────────────────────────────────────────────────────────────────────


def _derive_seed(base_seed: int, asset_id: str) -> int:
    """Stable per-asset seed (hash() is process-randomised — CRC32 is not)."""
    return zlib.crc32(f"{base_seed}:{asset_id}".encode()) & 0x7FFFFFFF


def _ar1(rng: np.random.Generator, n: int, sigma: float, phi: float = 0.92) -> FloatArray:
    """AR(1) autocorrelated noise — smooth like real sensor drift."""
    draws = rng.normal(0.0, sigma, n)
    out = np.empty(n, dtype=np.float64)
    level = 0.0
    for i, draw in enumerate(draws):
        level = phi * level + draw
        out[i] = level
    return out


def _wind_signals(hod: FloatArray, rng: np.random.Generator) -> dict[str, FloatArray]:
    n = hod.size
    wind_ms = np.clip(8.6 + 2.8 * np.sin(2 * np.pi * (hod - 3) / 24) + _ar1(rng, n, 0.9), 3.0, 22.0)
    rpm = np.clip(1.62 * wind_ms + _ar1(rng, n, 0.15), 8.0, 18.5)
    blade_pitch_deg = np.clip((wind_ms - 11.0) * 2.8 + _ar1(rng, n, 0.1), 0.0, 25.0)
    vib_base = 0.52 + 0.18 * (rpm / 18.5)
    return {
        "rpm": rpm,
        "vibration_x": np.clip(vib_base + np.abs(_ar1(rng, n, 0.05)), 0.0, None),
        "vibration_y": np.clip(vib_base + np.abs(_ar1(rng, n, 0.05)), 0.0, None),
        "vibration_z": np.clip(vib_base * 0.8 + np.abs(_ar1(rng, n, 0.04)), 0.0, None),
        "bearing_temp_c": 50.0
        + 5.5 * (rpm / 18.5)
        + 1.8 * np.sin(2 * np.pi * (hod - 14) / 24)
        + _ar1(rng, n, 0.25),
        "blade_pitch_deg": blade_pitch_deg,
    }


def _solar_signals(hod: FloatArray, rng: np.random.Generator) -> dict[str, FloatArray]:
    n = hod.size
    # Clear-sky irradiance bell 06:24→19:12 with smooth cloud factor.
    bell = np.clip(np.sin(np.pi * (hod - 6.4) / 12.8), 0.0, None) ** 1.25
    cloud = np.clip(1.0 + _ar1(rng, n, 0.10), 0.5, 1.08)
    irr = np.clip(bell * cloud, 0.0, 1.15)
    efficiency = np.clip(0.983 - 0.012 * (1.0 - irr) + _ar1(rng, n, 0.0015), 0.90, 0.995)
    return {
        "dc_voltage_v": np.clip(592.0 + 235.0 * np.sqrt(irr) + _ar1(rng, n, 6.0), 0.0, 850.0),
        "ac_power_kw": np.clip(250.0 * irr * 0.995 + _ar1(rng, n, 2.0), 0.0, 265.0),
        "efficiency": efficiency,
        "string_current_a": np.clip(11.8 * irr + _ar1(rng, n, 0.15), 0.0, 13.0),
    }


def _gas_signals(hod: FloatArray, rng: np.random.Generator) -> dict[str, FloatArray]:
    n = hod.size
    load = np.clip(0.72 + 0.22 * np.sin(2 * np.pi * (hod - 8) / 24) + _ar1(rng, n, 0.02), 0.5, 1.0)
    return {
        "exhaust_temp_c": 455.0 + 110.0 * load + _ar1(rng, n, 1.6),
        "fuel_flow_kg_s": np.clip(1.55 + 1.75 * load + _ar1(rng, n, 0.03), 0.0, None),
        "compressor_pressure_bar": np.clip(10.5 + 9.2 * load + _ar1(rng, n, 0.12), 0.0, None),
    }


def _transformer_signals(hod: FloatArray, rng: np.random.Generator) -> dict[str, FloatArray]:
    n = hod.size
    load_pct = np.clip(
        58.0 + 18.0 * np.sin(2 * np.pi * (hod - 14) / 24) + _ar1(rng, n, 3.0), 30.0, 95.0
    )
    return {
        "oil_temp_c": 38.0
        + 0.30 * load_pct
        + 2.6 * np.sin(2 * np.pi * (hod - 15) / 24)
        + _ar1(rng, n, 0.4),
        "load_pct": load_pct,
    }


_BuilderFn: TypeAlias = Callable[[FloatArray, np.random.Generator], dict[str, FloatArray]]
_BUILDERS: dict[AssetType, _BuilderFn] = {
    AssetType.WIND_TURBINE: _wind_signals,
    AssetType.SOLAR_INVERTER: _solar_signals,
    AssetType.GAS_TURBINE: _gas_signals,
    AssetType.HV_TRANSFORMER: _transformer_signals,
}

# ─────────────────────────────────────────────────────────────────────────────
# Anomaly application
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_channels(spec: AnomalySpec, asset_type: AssetType) -> tuple[str, ...]:
    """Channels directly manipulated by this anomaly."""
    if spec.channels is not None:
        unknown = set(spec.channels) - set(CHANNELS[asset_type])
        if unknown:
            raise ValueError(
                f"channels {sorted(unknown)} not valid for {asset_type.value}; "
                f"valid: {list(CHANNELS[asset_type])}"
            )
        return spec.channels
    if spec.kind is AnomalyKind.BEARING_WEAR:
        return _BEARING_CHANNELS
    if spec.kind is AnomalyKind.FOULING_DEGRADATION:
        primary = _FOULING_PRIMARY.get(asset_type)
        if primary is None:
            raise ValueError(f"fouling_degradation not modelled for {asset_type.value}")
        return (primary,)
    return (_STEP_DEFAULT[asset_type],)


def _apply_anomaly(
    signals: dict[str, FloatArray],
    times_epoch_s: FloatArray,
    spec: AnomalySpec,
    asset_type: AssetType,
) -> None:
    """Mutate ``signals`` in place according to ``spec``."""
    end_s = float(times_epoch_s[-1])
    t0 = end_s - spec.start_offset_hours * 3600.0
    active = times_epoch_s >= t0
    if not bool(active.any()):
        logger.warning("generator.anomaly_outside_window", anomaly=spec.kind.value)
        return

    ramp_s = spec.ramp_hours * 3600.0
    progress = np.clip((times_epoch_s - t0) / ramp_s, 0.0, 1.0)
    if not spec.hold:  # triangular decay back to normal over another ramp
        progress = np.where(
            times_epoch_s <= t0 + ramp_s,
            progress,
            np.clip(2.0 - (times_epoch_s - t0) / ramp_s, 0.0, 1.0),
        )
    active_seconds = np.clip(times_epoch_s - t0, 0.0, None)

    if spec.kind is AnomalyKind.BEARING_WEAR:
        factor = 1.0 + (spec.magnitude - 1.0) * progress
        for channel in _resolve_channels(spec, asset_type):
            signals[channel] = signals[channel] * factor
        signals["bearing_temp_c"] = signals["bearing_temp_c"] + spec.rate * (
            active_seconds / 3600.0
        )

    elif spec.kind is AnomalyKind.FOULING_DEGRADATION:
        rate = spec.rate if spec.rate > 0 else _FOULING_DEFAULT_RATE.get(asset_type, 0.02)
        loss = 1.0 - rate * (active_seconds / 86400.0)
        if asset_type is AssetType.SOLAR_INVERTER:
            signals["efficiency"] = signals["efficiency"] * loss
            signals["ac_power_kw"] = signals["ac_power_kw"] * loss
            signals["string_current_a"] = signals["string_current_a"] * np.clip(loss, 0.0, 1.0)
        elif asset_type is AssetType.GAS_TURBINE:
            signals["compressor_pressure_bar"] = signals["compressor_pressure_bar"] * loss
            signals["fuel_flow_kg_s"] = signals["fuel_flow_kg_s"] * (1.0 + (1.0 - loss) * 0.65)
            signals["exhaust_temp_c"] = signals["exhaust_temp_c"] + (1.0 - loss) * 28.0
        else:  # guarded by _resolve_channels, defensive double-check
            raise ValueError(f"fouling_degradation not modelled for {asset_type.value}")

    else:  # STEP_CHANGE — absolute jump at t0, held while active
        (channel,) = _resolve_channels(spec, asset_type)[:1]
        signals[channel] = signals[channel] + spec.magnitude * progress


# ─────────────────────────────────────────────────────────────────────────────
# Public generation API
# ─────────────────────────────────────────────────────────────────────────────


def generate_sensor_data(
    asset: AssetSpec | str,
    *,
    end: datetime | None = None,
    hours: float = 168.0,
    freq_minutes: int = 10,
    seed: int | None = None,
    anomalies: Sequence[AnomalySpec] = (),
) -> pl.DataFrame:
    """Generate one asset's sensor history as a Polars DataFrame.

    Args:
        asset: Fleet :class:`AssetSpec` or asset_id string (sanitised via
            :func:`get_asset`).
        end: Window end (UTC assumed when naive); defaults to *now*.
        hours: Lookback window length in hours.
        freq_minutes: Sampling interval in minutes.
        seed: Base seed; defaults to ``Settings.random_seed``.
        anomalies: Anomaly specs to inject (only those matching this asset).

    Returns:
        Polars DataFrame, first column timezone-aware UTC ``time`` (sorted,
        unique — Polars is index-free, this column is the effective index),
        followed by one float column per channel for the asset type.
    """
    spec_asset = asset if isinstance(asset, AssetSpec) else get_asset(asset)
    cfg = get_settings()
    end = end or datetime.now(UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    start = end - timedelta(hours=hours)

    times = pl.datetime_range(
        start, end, interval=f"{freq_minutes}m", closed="both", eager=True, time_zone="UTC"
    )
    hod = (
        times.dt.hour() + times.dt.minute() / 60.0 + times.dt.second() / 3600.0
    ).to_numpy().astype(np.float64)
    epoch_s = times.dt.epoch("s").to_numpy().astype(np.float64)

    rng = np.random.default_rng(_derive_seed(seed if seed is not None else cfg.random_seed, spec_asset.asset_id))
    signals = _BUILDERS[spec_asset.asset_type](hod, rng)

    applied: list[str] = []
    for anomaly in anomalies:
        if anomaly.asset_id == spec_asset.asset_id:
            _apply_anomaly(signals, epoch_s, anomaly, spec_asset.asset_type)
            applied.append(anomaly.kind.value)

    df = pl.DataFrame({"time": times, **signals}).sort("time")
    bind_log_context(asset_id=spec_asset.asset_id)
    logger.info(
        "generator.data_generated",
        rows=df.height,
        channels=df.width - 1,
        anomalies_applied=applied,
        window_hours=hours,
    )
    clear_log_context()
    return df


def dataframe_to_sensor_rows(df: pl.DataFrame, asset_id: str) -> list[dict[str, object]]:
    """Melt a wide generator frame into sensor_readings row dicts."""
    get_asset(asset_id)  # sanitise
    long_df = df.unpivot(index="time", variable_name="channel", value_name="value")
    return [
        {"time": time_val, "asset_id": asset_id, "channel": channel, "value": float(value)}
        for time_val, channel, value in long_df.iter_rows()
    ]


async def store_sensor_dataframe(
    df: pl.DataFrame, asset_id: str, *, batch_size: int = 5000
) -> int:
    """Write a generator frame to ``sensor_readings``. Idempotent.

    Uses ``ON CONFLICT DO NOTHING`` on the hypertable PK so re-seeding the
    same window never duplicates rows. Returns the number of rows offered.
    """
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    records = dataframe_to_sensor_rows(df, asset_id)
    if not records:
        return 0
    stmt = pg_insert(SensorReading).on_conflict_do_nothing(
        index_elements=["time", "asset_id", "channel"]
    )
    async with session_scope() as session:
        for offset in range(0, len(records), batch_size):
            await session.execute(stmt, records[offset : offset + batch_size])
    logger.info("generator.stored", asset_id=asset_id, rows=len(records))
    return len(records)


# ─────────────────────────────────────────────────────────────────────────────
# Bulk orchestration (demo seeding)
# ─────────────────────────────────────────────────────────────────────────────


class GenerateRequest(BaseModel):
    """Bulk generation request for the whole fleet (or a subset)."""

    asset_ids: list[str] | None = None  # None → every asset in FLEET
    hours: float = Field(default=168.0, gt=0.0, le=24 * 90)
    freq_minutes: int = Field(default=10, ge=1, le=60)
    seed: int | None = None  # None → Settings.random_seed
    include_wt07_scenario: bool = True  # inject the anchored WT-07 anomaly
    anomalies: list[AnomalySpec] = Field(default_factory=list)  # extra anomalies


class GenerationSummary(BaseModel):
    """Result of :func:`generate_and_store`."""

    trace_id: str
    window_hours: float
    freq_minutes: int
    per_asset_rows: dict[str, int]
    total_rows: int
    anomalies_applied: list[str]
    started_at: datetime
    finished_at: datetime


async def generate_and_store(request: GenerateRequest | None = None) -> GenerationSummary:
    """Generate the fleet history (with anomalies) and persist it to TimescaleDB."""
    req = request or GenerateRequest()
    cfg = get_settings()
    trace_id = new_trace_id()
    bind_log_context(trace_id=trace_id)

    asset_ids = req.asset_ids or list(FLEET)
    anomalies: list[AnomalySpec] = list(req.anomalies)
    if req.include_wt07_scenario:
        anomalies.append(SCENARIO_WT07_BEARING)

    started = datetime.now(UTC)
    per_asset_rows: dict[str, int] = {}
    for asset_id in asset_ids:
        bind_log_context(asset_id=asset_id)
        df = generate_sensor_data(
            get_asset(asset_id),
            hours=req.hours,
            freq_minutes=req.freq_minutes,
            seed=req.seed if req.seed is not None else cfg.random_seed,
            anomalies=anomalies,
        )
        per_asset_rows[asset_id] = await store_sensor_dataframe(df, asset_id)

    summary = GenerationSummary(
        trace_id=trace_id,
        window_hours=req.hours,
        freq_minutes=req.freq_minutes,
        per_asset_rows=per_asset_rows,
        total_rows=sum(per_asset_rows.values()),
        anomalies_applied=[a.kind.value for a in anomalies],
        started_at=started,
        finished_at=datetime.now(UTC),
    )
    logger.info(
        "generator.fleet_seeded",
        assets=len(per_asset_rows),
        total_rows=summary.total_rows,
        anomalies=summary.anomalies_applied,
    )
    clear_log_context()
    return summary


if __name__ == "__main__":  # sync wrapper allowed only at CLI entry
    asyncio.run(generate_and_store())
