"""AnomalyDetectorTool — IsolationForest + robust Z-score ensemble.

Method
------
1. Load the asset's sensor window (default 7 days, 10-min cadence).
2. **Robust Z-score**: median/MAD-based z per channel for the *recent*
   window (default last 6 h) against the full-window baseline — immune to
   the anomaly itself corrupting mean/std.
3. **IsolationForest**: fitted on the full standardized channel matrix
   (contamination 5 %); score = fraction of recent points flagged anomalous.
4. Ensemble ``score = 0.6·iforest + 0.4·z_norm`` with ``z_norm = clip(z_max/12)``.

Severity policy
---------------
Bands: < 0.35 LOW · < 0.60 MED · < 0.85 HIGH · else CRITICAL — with a
corroboration rule: CRITICAL requires ``max(recent/baseline ratio) ≥ 4`` OR
breach of an absolute safety limit (bearing temp 80 °C, transformer oil
75 °C, …). A textbook 3×-vibration bearing wear therefore lands at HIGH —
actionable, but not a trip. Detected events are persisted to
``anomaly_events`` (best effort; persistence failure does not fail the tool).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import ClassVar

import numpy as np
import polars as pl
from pydantic import Field

from memory.db import AnomalyEvent, session_scope
from tools.base import BaseToolInput, BaseToolOutput, EnergyForgeTool
from tools.sensor_query import fetch_sensor_frame

# Absolute safety limits from equipment manuals (memory/knowledge_corpus.py).
_ABSOLUTE_LIMITS: dict[str, tuple[str, float]] = {
    "bearing_temp_c": ("above", 80.0),
    "oil_temp_c": ("above", 75.0),
    "exhaust_temp_c": ("above", 565.0),
    "vibration_x": ("above", 4.5),
    "vibration_y": ("above", 4.5),
    "vibration_z": ("above", 4.5),
    "efficiency": ("below", 0.90),
}


class AnomalyDetectorInput(BaseToolInput):
    """Input for AnomalyDetectorTool."""

    asset_id: str = Field(description="Fleet asset id, e.g. WT-07")
    hours_back: float = Field(default=168.0, gt=1, le=24 * 90,
                              description="Baseline window in hours")
    recent_hours: float = Field(default=6.0, gt=0.25, le=72,
                                description="Recent window to evaluate")


class AnomalyDetectionOutput(BaseToolOutput):
    """Anomaly ensemble result."""

    asset_id: str = ""
    is_anomalous: bool = False
    score: float = 0.0  # ensemble anomaly score 0–1
    severity: str = "LOW"  # LOW / MED / HIGH / CRITICAL
    affected_channels: list[str] = Field(default_factory=list)
    per_channel: dict[str, dict[str, float]] = Field(default_factory=dict)
    absolute_breach: bool = False
    explanation: str = ""
    event_id: str = ""


class AnomalyDetectorTool(EnergyForgeTool[AnomalyDetectorInput, AnomalyDetectionOutput]):
    """Detect anomalies in an asset's recent sensor window.

    Combines an IsolationForest with robust (median/MAD) Z-scores. Returns a
    0–1 anomaly score, the affected channels, and a LOW/MED/HIGH/CRITICAL
    severity that respects absolute equipment safety limits.
    """

    name: ClassVar[str] = "anomaly_detector"
    description: ClassVar[str] = (
        "Run an IsolationForest + robust Z-score ensemble over an asset's recent "
        "sensor window. Returns anomaly score (0–1), affected channels, severity "
        "(LOW/MED/HIGH/CRITICAL) and per-channel evidence (z-score, recent vs "
        "baseline means). Input: asset_id, hours_back (baseline), recent_hours."
    )
    input_model: ClassVar[type[BaseToolInput]] = AnomalyDetectorInput
    output_model: ClassVar[type[BaseToolOutput]] = AnomalyDetectionOutput

    async def _arun(self, tool_input: AnomalyDetectorInput) -> AnomalyDetectionOutput:
        result = await fetch_sensor_frame(
            tool_input.asset_id.strip().upper(),
            hours_back=tool_input.hours_back,
            settings=self._settings,
        )
        frame = result.frame
        channels = [c for c in frame.columns if c != "time"]
        if frame.height < 50:
            return AnomalyDetectionOutput(
                asset_id=tool_input.asset_id,
                error=f"insufficient data: {frame.height} rows (need ≥ 50)",
            )

        matrix = np.column_stack([frame[c].to_numpy() for c in channels])
        n_recent = max(5, int(frame.height * tool_input.recent_hours / tool_input.hours_back))
        recent, baseline = matrix[-n_recent:], matrix[:-n_recent]

        # ── Robust Z-score (median/MAD) per channel ──────────────────────
        median = np.median(baseline, axis=0)
        mad = np.median(np.abs(baseline - median), axis=0)
        mad = np.where(mad < 1e-9, 1e-9, mad)
        z_recent = np.abs(0.6745 * (recent - median) / mad)
        max_z_per_channel = z_recent.max(axis=0)

        baseline_mean = np.abs(baseline.mean(axis=0)) + 1e-9
        ratio = np.abs(recent.mean(axis=0)) / baseline_mean

        # ── IsolationForest on the full standardized window ──────────────
        from sklearn.ensemble import IsolationForest

        std = baseline.std(axis=0)
        std = np.where(std < 1e-9, 1e-9, std)
        scaled = (matrix - baseline.mean(axis=0)) / std
        forest = IsolationForest(
            n_estimators=200, contamination=0.05,
            random_state=self._settings.random_seed, n_jobs=1,
        ).fit(scaled)
        iforest_frac = float(np.mean(forest.predict(scaled[-n_recent:]) == -1))

        # ── Ensemble ─────────────────────────────────────────────────────
        z_max = float(max_z_per_channel.max())
        z_norm = min(1.0, z_max / 12.0)
        score = round(0.6 * iforest_frac + 0.4 * z_norm, 4)

        # ── Evidence + severity ──────────────────────────────────────────
        affected_idx = np.where(max_z_per_channel > 4.0)[0]
        if iforest_frac > 0.5 and not len(affected_idx):
            affected_idx = np.argsort(max_z_per_channel)[::-1][:2]  # largest movers
        affected = sorted({channels[i] for i in affected_idx})

        per_channel: dict[str, dict[str, float]] = {}
        for i, channel in enumerate(channels):
            per_channel[channel] = {
                "z": round(float(max_z_per_channel[i]), 2),
                "recent_mean": round(float(recent[:, i].mean()), 4),
                "baseline_mean": round(float(baseline[:, i].mean()), 4),
                "ratio": round(float(ratio[i]), 3),
            }

        breach = self._check_absolute_breach(recent, channels)
        max_ratio = float(ratio.max())
        severity = self._severity(score, max_ratio, breach)

        is_anomalous = score >= 0.35
        explanation = self._explain(score, severity, affected, z_max, iforest_frac, breach)
        event_id = ""
        if is_anomalous:
            event_id = await self._persist_event(
                tool_input, score, severity, affected, explanation
            )

        confidence = min(0.97, 0.55 + 0.25 * iforest_frac + 0.20 * z_norm)
        return AnomalyDetectionOutput(
            asset_id=tool_input.asset_id.strip().upper(),
            is_anomalous=is_anomalous,
            score=score,
            severity=severity,
            affected_channels=affected,
            per_channel=per_channel,
            absolute_breach=breach,
            explanation=explanation,
            event_id=event_id,
            confidence=round(confidence, 3),
        )

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _check_absolute_breach(recent: np.ndarray, channels: list[str]) -> bool:
        for i, channel in enumerate(channels):
            limit = _ABSOLUTE_LIMITS.get(channel)
            if limit is None:
                continue
            direction, value = limit
            series = recent[:, i]
            if direction == "above" and bool((series > value).any()):
                return True
            if direction == "below" and bool((series < value).any()):
                return True
        return False

    @staticmethod
    def _severity(score: float, max_ratio: float, breach: bool) -> str:
        if score < 0.35:
            return "LOW"
        if score < 0.60:
            return "MED"
        if score < 0.85:
            return "HIGH"
        # CRITICAL band — corroborated by ≥4× excursion or absolute safety breach
        if max_ratio >= 4.0 or breach:
            return "CRITICAL"
        return "HIGH"

    @staticmethod
    def _explain(
        score: float,
        severity: str,
        affected: list[str],
        z_max: float,
        iforest_frac: float,
        breach: bool,
    ) -> str:
        if severity in ("LOW", "MED") and score < 0.35:
            return f"No significant anomaly (score={score:.2f}); channels within normal envelope."
        parts = [
            f"Anomaly score {score:.2f} ({severity}) on {', '.join(affected) or 'n/a'}",
            f"max robust z={z_max:.1f}",
            f"{iforest_frac:.0%} of recent points flagged by IsolationForest",
        ]
        if breach:
            parts.append("ABSOLUTE SAFETY LIMIT BREACHED")
        return "; ".join(parts) + "."

    async def _persist_event(
        self,
        tool_input: AnomalyDetectorInput,
        score: float,
        severity: str,
        affected: list[str],
        explanation: str,
    ) -> str:
        try:
            event = AnomalyEvent(
                time=datetime.now(UTC),
                asset_id=tool_input.asset_id.strip().upper(),
                detector=self.name,
                score=score,
                severity=severity,
                channels=list(affected),
                description=explanation,
            )
            async with session_scope() as session:
                session.add(event)
                await session.flush()
                return str(event.id)
        except Exception as exc:  # persistence is best-effort — detection already succeeded
            from logging_config import get_logger

            get_logger("tools.anomaly_detector").warning(
                "anomaly_event.persist_failed", error=str(exc)[:200]
            )
            return ""


def frame_recent_window(frame: pl.DataFrame, hours: float, total_hours: float) -> pl.DataFrame:
    """Slice the trailing ``hours`` of a wide frame — exported for reuse."""
    fraction = min(1.0, hours / max(total_hours, 1e-9))
    n = max(1, int(frame.height * fraction))
    return frame.tail(n)
