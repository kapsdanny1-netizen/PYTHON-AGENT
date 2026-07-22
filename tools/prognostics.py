"""PrognosticsTool — Prophet + XGBoost Remaining-Useful-Life estimator.

Pipeline
--------
1. Fetch the asset window and select the **degradation channel** for the
   asset type (wind → bearing_temp_c @ 80 °C alarm, transformer → oil_temp_c
   @ 75 °C, gas → exhaust_temp_c @ 565 °C, solar → efficiency @ 0.90).
2. **Prophet** on the 1-hourly resampled series → 60-day forecast; RUL
   quantiles from threshold crossings: ``yhat_upper`` → P10 (pessimistic),
   ``yhat`` → P50, ``yhat_lower`` → P90 (optimistic, possibly "never").
3. **XGBoost quantile regressors** (α = 0.1/0.5/0.9,
   ``objective="reg:quantileerror"``) trained on a seeded synthetic library
   of degradation curves with known crossing times. Features: current value,
   recent slope, window trend, distance-to-threshold, daily amplitude.
4. Both estimates are blended per quantile (median); confidence reflects
   model agreement.

When no degradation trend exists (non-positive slope), RUL reports the
365-day horizon cap with a candid "no active degradation" note instead of a
fake precise number. Heavy CPU work (Prophet/XGB fitting) is offloaded with
``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import ClassVar

import numpy as np
from pydantic import Field

from data.generators import AssetType, get_asset
from tools.base import BaseToolInput, BaseToolOutput, EnergyForgeTool
from tools.sensor_query import fetch_sensor_frame

# Degradation channel + failure threshold per asset type (from equipment manuals).
_FAILURE_MODES: dict[AssetType, tuple[str, float]] = {
    AssetType.WIND_TURBINE: ("bearing_temp_c", 80.0),
    AssetType.SOLAR_INVERTER: ("efficiency", 0.90),
    AssetType.GAS_TURBINE: ("exhaust_temp_c", 565.0),
    AssetType.HV_TRANSFORMER: ("oil_temp_c", 75.0),
}
_RUL_CAP_DAYS = 365.0
_FORECAST_DAYS = 60

for noisy in ("prophet", "cmdstanpy", "prophet.plot"):  # keep logs JSON-clean
    logging.getLogger(noisy).setLevel(logging.WARNING)


class PrognosticsInput(BaseToolInput):
    """Input for PrognosticsTool."""

    asset_id: str = Field(description="Fleet asset id, e.g. WT-07")
    hours_back: float = Field(default=168.0, gt=6, le=24 * 90,
                              description="Training window in hours")


class PrognosticsOutput(BaseToolOutput):
    """RUL estimate."""

    asset_id: str = ""
    days_to_failure_p10: float = _RUL_CAP_DAYS
    days_to_failure_p50: float = _RUL_CAP_DAYS
    days_to_failure_p90: float = _RUL_CAP_DAYS
    failure_mode: str = ""
    failure_threshold: float = 0.0
    current_value: float = 0.0
    slope_per_day: float = 0.0
    method: str = "prophet+xgboost-quantile blend"
    note: str = ""


def _fit_prophet_crossings(ds: np.ndarray, y: np.ndarray, threshold: float) -> dict[str, float]:
    """Fit Prophet, return threshold-crossing days for lower/point/upper bands."""
    import pandas as pd
    from prophet import Prophet

    frame = pd.DataFrame({"ds": pd.to_datetime(ds, unit="s"), "y": y})
    model = Prophet(
        daily_seasonality=True, weekly_seasonality=False, yearly_seasonality=False,
        changepoint_prior_scale=0.08, interval_width=0.8,
    )
    model.fit(frame)
    future = model.make_future_dataframe(periods=_FORECAST_DAYS * 24, freq="h")
    forecast = model.predict(future)

    horizon = forecast.tail(_FORECAST_DAYS * 24).reset_index(drop=True)
    now_hour = len(frame)

    def crossing(col: str) -> float:
        above = horizon.index[horizon[col] >= threshold]
        if len(above) == 0:
            return _RUL_CAP_DAYS
        return float(above[0]) / 24.0

    return {
        "p10": crossing("yhat_upper"),
        "p50": crossing("yhat"),
        "p90": crossing("yhat_lower"),
        "_points": float(now_hour),
    }


def _xgboost_quantiles(features: np.ndarray, threshold: float, seed: int) -> dict[str, float]:
    """Quantile RUL from XGBoost trained on a synthetic degradation library."""
    from xgboost import XGBRegressor

    rng = np.random.default_rng(seed)
    n_train = 400
    # Synthetic wear library: linear-to-accelerating trends with noise, known RUL.
    train_x = np.zeros((n_train, features.shape[1]))
    train_y = np.zeros(n_train)
    for i in range(n_train):
        slope = rng.uniform(0.0, 3.5)          # units per hour
        current = rng.uniform(0.82, 0.995) * threshold
        noise_lvl = rng.uniform(0.002, 0.03) * threshold
        rul_hours = max(0.5, (threshold - current) / max(slope, 1e-3))
        train_x[i] = [
            current / threshold, slope, slope * 24.0,
            1.0 - current / threshold, noise_lvl / threshold,
        ]
        train_y[i] = min(_RUL_CAP_DAYS, rul_hours / 24.0)
    qx = np.array([
        features[0] / threshold, features[1], features[1] * 24.0,
        1.0 - features[0] / threshold, features[4] / threshold,
    ]).reshape(1, -1)

    out: dict[str, float] = {}
    for label, alpha in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9)):
        model = XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=alpha,
            n_estimators=120, max_depth=3, learning_rate=0.08,
            subsample=0.9, random_state=seed, n_jobs=1, verbosity=0,
        )
        model.fit(train_x, train_y)
        out[label] = float(np.clip(model.predict(qx)[0], 0.25, _RUL_CAP_DAYS))
    # enforce monotone quantiles
    out["p10"], out["p50"], out["p90"] = sorted((out["p10"], out["p50"], out["p90"]))
    return out


def _compute_rul(
    epoch_s: np.ndarray, values: np.ndarray, threshold: float, seed: int
) -> dict[str, float]:
    """CPU-bound portion: resample → Prophet + XGBoost → blended quantiles."""
    # 1-hourly resample for Prophet stability/speed.
    bucket = (epoch_s // 3600.0).astype(np.int64)
    _, first = np.unique(bucket, return_index=True)
    ds = epoch_s[first]
    y = np.array([values[bucket == b].mean() for b in bucket[first]])

    # Full-window trend + recent trend, both in units/hour (cadence-agnostic).
    slope_per_hour = float(np.polyfit((epoch_s - epoch_s[0]) / 3600.0, values, 1)[0])
    recent_n = min(24, len(values))
    recent_t = (epoch_s[-recent_n:] - epoch_s[-recent_n]) / 3600.0
    recent_v = values[-recent_n:]
    slope_recent = (
        float(np.polyfit(recent_t, recent_v, 1)[0]) if recent_t[-1] > 0 else 0.0
    )
    daily_amp = float(values.std())
    features = np.array([values[-1], slope_recent, slope_per_hour,
                         threshold - values[-1], daily_amp])

    if slope_per_hour <= 1e-6 and slope_recent <= 1e-6:
        return {"p10": _RUL_CAP_DAYS, "p50": _RUL_CAP_DAYS, "p90": _RUL_CAP_DAYS,
                "slope": slope_per_hour, "agreement": 1.0, "trend": 0.0}

    prophet_q = _fit_prophet_crossings(ds, y, threshold)
    try:
        xgb_q = _xgboost_quantiles(features, threshold, seed)
        blended = {k: float(np.median([prophet_q[k], xgb_q[k]])) for k in ("p10", "p50", "p90")}
        denom = max(1.0, blended["p50"])
        agreement = 1.0 - min(1.0, abs(prophet_q["p50"] - xgb_q["p50"]) / denom)
    except Exception:  # XGBoost quantile objective unavailable — Prophet-only fallback
        blended = {k: prophet_q[k] for k in ("p10", "p50", "p90")}
        agreement = 0.6
    p10, p50, p90 = sorted((blended["p10"], blended["p50"], blended["p90"]))
    return {"p10": p10, "p50": p50, "p90": p90,
            "slope": slope_per_hour, "agreement": agreement, "trend": 1.0}


class PrognosticsTool(EnergyForgeTool[PrognosticsInput, PrognosticsOutput]):
    """Estimate remaining useful life (RUL) of an asset's degradation mode.

    Blends a Prophet forecast of the failure channel with XGBoost quantile
    regression over synthetic wear libraries. Returns days-to-failure as a
    P10/P50/P90 distribution plus a confidence reflecting model agreement.
    """

    name: ClassVar[str] = "prognostics"
    description: ClassVar[str] = (
        "Estimate Remaining Useful Life for an asset's dominant failure mode "
        "(e.g. bearing temperature to its 80°C alarm). Returns days_to_failure "
        "as P10/P50/P90 quantiles, the failure threshold, current value and "
        "trend slope. Input: asset_id, hours_back."
    )
    input_model: ClassVar[type[BaseToolInput]] = PrognosticsInput
    output_model: ClassVar[type[BaseToolOutput]] = PrognosticsOutput

    async def _arun(self, tool_input: PrognosticsInput) -> PrognosticsOutput:
        asset = get_asset(tool_input.asset_id.strip().upper())
        channel, threshold = _FAILURE_MODES[asset.asset_type]
        result = await fetch_sensor_frame(
            asset.asset_id, hours_back=tool_input.hours_back,
            channels=[channel], settings=self._settings,
        )
        frame = result.frame.drop_nans()
        if frame.height < 48:
            return PrognosticsOutput(
                asset_id=asset.asset_id,
                error=f"insufficient {channel} data ({frame.height} rows, need ≥ 48)",
            )
        epoch_s = frame["time"].dt.epoch("s").to_numpy().astype(np.float64)
        values = frame[channel].to_numpy().astype(np.float64)

        rul = await asyncio.to_thread(
            _compute_rul, epoch_s, values, threshold, self._settings.random_seed
        )
        no_trend = rul["trend"] == 0.0
        confidence = 0.6 if no_trend else round(0.55 + 0.4 * rul["agreement"], 3)
        note = (
            f"No active degradation trend on {channel}; horizon-capped RUL."
            if no_trend
            else f"Degradation trend +{rul['slope']:.3f} {channel}/h toward {threshold} threshold."
        )
        return PrognosticsOutput(
            asset_id=asset.asset_id,
            days_to_failure_p10=round(rul["p10"], 2),
            days_to_failure_p50=round(rul["p50"], 2),
            days_to_failure_p90=round(rul["p90"], 2),
            failure_mode=f"{channel} → {threshold}",
            failure_threshold=threshold,
            current_value=round(float(values[-1]), 3),
            slope_per_day=round(rul["slope"] * 24.0, 4),
            note=note,
            confidence=confidence,
        )
