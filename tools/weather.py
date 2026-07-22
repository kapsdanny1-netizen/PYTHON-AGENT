"""WeatherTool — Open-Meteo forecast (free API, no key required).

Endpoint: https://api.open-meteo.com/v1/forecast
Hourly variables: temperature_2m, wind_speed_10m, shortwave_radiation.

Returns the next-24 h aggregates the OptimizationAgent needs: mean/max wind
speed, mean + peak irradiance, mean temperature, plus heuristic capacity
factors:

* wind  ≈ mean( clip(ws / 11 m/s, 1)³ )     — cube-law against rated wind
* solar ≈ mean_irradiance / 1000 W/m²

Retried once on transient failure (tenacity); network/parse problems are
surfaced through the standard ``error`` field (the tool never raises).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import ClassVar

import httpx
from pydantic import Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from exceptions import WeatherAPIError
from tools.base import BaseToolInput, BaseToolOutput, EnergyForgeTool

_OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
_HOURLY_VARS = "temperature_2m,wind_speed_10m,shortwave_radiation"


class WeatherInput(BaseToolInput):
    """Input for WeatherTool."""

    latitude: float = Field(ge=-90, le=90, description="Asset latitude")
    longitude: float = Field(ge=-180, le=180, description="Asset longitude")
    hours_ahead: int = Field(default=24, ge=1, le=72, description="Forecast horizon")
    asset_id: str | None = Field(default=None, description="Optional asset id for context")


class WeatherOutput(BaseToolOutput):
    """Weather forecast aggregates."""

    asset_id: str | None = None
    latitude: float = 0.0
    longitude: float = 0.0
    horizon_hours: int = 0
    wind_speed_avg_ms: float = 0.0
    wind_speed_max_ms: float = 0.0
    irradiance_avg_wm2: float = 0.0
    irradiance_peak_wm2: float = 0.0
    temperature_avg_c: float = 0.0
    wind_capacity_factor: float = 0.0
    solar_capacity_factor: float = 0.0
    generated_at: str = ""


class WeatherTool(EnergyForgeTool[WeatherInput, WeatherOutput]):
    """Fetch wind/irradiance/temperature forecast for an asset's coordinates.

    Calls the free Open-Meteo API (no API key). Returns the forecast
    aggregates used for dispatch and set-point optimisation.
    """

    name: ClassVar[str] = "weather_forecast"
    description: ClassVar[str] = (
        "Get the weather forecast for an asset location (Open-Meteo, no key). "
        "Returns next-24h averages: wind speed (avg/max m/s), shortwave "
        "irradiance (avg/peak W/m²), temperature (°C), plus estimated wind and "
        "solar capacity factors. Input: latitude, longitude, hours_ahead."
    )
    input_model: ClassVar[type[BaseToolInput]] = WeatherInput
    output_model: ClassVar[type[BaseToolOutput]] = WeatherOutput

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_fixed(1),
        retry=retry_if_exception_type(WeatherAPIError),
        reraise=True,
    )
    async def _fetch(self, tool_input: WeatherInput) -> dict[str, list[float]]:
        params = {
            "latitude": tool_input.latitude,
            "longitude": tool_input.longitude,
            "hourly": _HOURLY_VARS,
            "forecast_days": 3,
            "timezone": "UTC",
        }
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.get(_OPEN_METEO_URL, params=params)
                response.raise_for_status()
                payload = response.json()
            return payload["hourly"]  # type: ignore[no-any-return]
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise WeatherAPIError(
                f"Open-Meteo request failed: {exc}",
                context={"latitude": tool_input.latitude, "longitude": tool_input.longitude},
            ) from exc

    async def _arun(self, tool_input: WeatherInput) -> WeatherOutput:
        hourly = await self._fetch(tool_input)

        times = hourly.get("time", [])
        now = datetime.now(UTC)
        cutoff = [i for i, t in enumerate(times)
                  if datetime.fromisoformat(t).replace(tzinfo=UTC) >= now]
        window = cutoff[: tool_input.hours_ahead] or list(range(tool_input.hours_ahead))

        def series(name: str) -> list[float]:
            values = hourly.get(name, [])
            return [
                float(values[i]) for i in window
                if i < len(values) and values[i] is not None
            ]

        wind = series("wind_speed_10m")
        ghi = series("shortwave_radiation")
        temp = series("temperature_2m")
        if not wind or not ghi or not temp:
            raise WeatherAPIError("Open-Meteo payload missing series")

        wind_avg = sum(wind) / len(wind)
        ghi_avg = sum(ghi) / len(ghi)
        wind_cf = sum(min(1.0, w / 11.0) ** 3 for w in wind) / len(wind) * 0.95
        return WeatherOutput(
            asset_id=tool_input.asset_id,
            latitude=tool_input.latitude,
            longitude=tool_input.longitude,
            horizon_hours=len(window),
            wind_speed_avg_ms=round(wind_avg, 2),
            wind_speed_max_ms=round(max(wind), 2),
            irradiance_avg_wm2=round(ghi_avg, 1),
            irradiance_peak_wm2=round(max(ghi), 1),
            temperature_avg_c=round(sum(temp) / len(temp), 1),
            wind_capacity_factor=round(wind_cf, 3),
            solar_capacity_factor=round(min(1.0, ghi_avg / 1000.0), 3),
            generated_at=now.isoformat(),
            confidence=0.9,
        )
