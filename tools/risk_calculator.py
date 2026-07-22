"""RiskCalculatorTool — bow-tie risk model.

Bow-tie structure per hazard::

    threats ──► [preventive barriers] ──► TOP EVENT ──► [mitigative barriers] ──► consequences

Computation
-----------
* inherent likelihood: mapped from the input severity
  (LOW 0.05 / MED 0.20 / HIGH 0.45 / CRITICAL 0.80)
* preventive barriers reduce likelihood; mitigative barriers reduce the
  consequence severity class (1–10)
* ``residual_likelihood = L0 · Π(1 − eff_i·health_i)``
* ``residual_consequence = C0 · Π(1 − 0.5·eff_j·health_j)``
* ``risk_score = residual_likelihood · residual_consequence · 10`` (0–100)

``top_3_barriers`` are the barriers with the largest *leverage gap*
``eff_i·(1 − health_i)`` — i.e. where restoring health buys the most risk
reduction. Bow-tie libraries live in-code per asset type/hazard class; in
production they would be curated in a safety-knowledge base.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from data.generators import AssetType, get_asset
from tools.base import BaseToolInput, BaseToolOutput, EnergyForgeTool

_SEVERITY_LIKELIHOOD = {"LOW": 0.05, "MED": 0.20, "HIGH": 0.45, "CRITICAL": 0.80}


class Barrier(BaseModel):
    """One bow-tie barrier."""

    name: str
    side: str  # "preventive" | "mitigative"
    effectiveness: float = Field(ge=0.0, le=1.0)
    health: float = Field(default=1.0, ge=0.0, le=1.0)


class BowTie(BaseModel):
    """Hazard definition: top event, consequence class, barriers."""

    hazard: str
    top_event: str
    consequence_class: float = Field(gt=0, le=10)
    barriers: list[Barrier]


# ── Bow-tie libraries per asset type ──────────────────────────────────────────
def _wt_bowtie() -> BowTie:
    return BowTie(
        hazard="bearing_failure",
        top_event="Main bearing seizure",
        consequence_class=6.0,
        barriers=[
            Barrier(name="Vibration CMS trending", side="preventive", effectiveness=0.80),
            Barrier(name="Bearing temperature alarm (80°C)", side="preventive", effectiveness=0.75),
            Barrier(name="Grease/oil analysis programme", side="preventive", effectiveness=0.60),
            Barrier(name="Planned replacement before RUL exhaustion", side="mitigative", effectiveness=0.70),
            Barrier(name="Automatic trip at 90°C", side="mitigative", effectiveness=0.85),
            Barrier(name="Low-wind weather window scheduling", side="mitigative", effectiveness=0.40),
        ],
    )


def _gt_bowtie() -> BowTie:
    return BowTie(
        hazard="compressor_degradation",
        top_event="Compressor surge",
        consequence_class=7.0,
        barriers=[
            Barrier(name="Pressure-ratio trend monitoring", side="preventive", effectiveness=0.75),
            Barrier(name="Inlet filter ΔP watch", side="preventive", effectiveness=0.55),
            Barrier(name="Condition-based water wash", side="preventive", effectiveness=0.65),
            Barrier(name="Surge protection valves", side="mitigative", effectiveness=0.90),
            Barrier(name="Runback to reduced load", side="mitigative", effectiveness=0.60),
        ],
    )


def _pv_bowtie() -> BowTie:
    return BowTie(
        hazard="string_underperformance",
        top_event="Sustained energy yield loss",
        consequence_class=3.0,
        barriers=[
            Barrier(name="Efficiency-normalised irradiance tracking", side="preventive", effectiveness=0.70),
            Barrier(name="Soiling monitoring", side="preventive", effectiveness=0.55),
            Barrier(name="Robotic cleaning schedule", side="preventive", effectiveness=0.65),
            Barrier(name="MPPT firmware updates", side="mitigative", effectiveness=0.40),
        ],
    )


def _tr_bowtie() -> BowTie:
    return BowTie(
        hazard="overheating",
        top_event="Winding insulation damage",
        consequence_class=8.0,
        barriers=[
            Barrier(name="Top-oil temperature alarm (75°C)", side="preventive", effectiveness=0.80),
            Barrier(name="Cooling fan/pump status monitoring", side="preventive", effectiveness=0.70),
            Barrier(name="Dissolved gas analysis (DGA)", side="preventive", effectiveness=0.60),
            Barrier(name="Automatic trip at 90°C", side="mitigative", effectiveness=0.90),
            Barrier(name="Load transfer to parallel transformer", side="mitigative", effectiveness=0.65),
        ],
    )


_BOWTIES: dict[AssetType, BowTie] = {
    AssetType.WIND_TURBINE: _wt_bowtie(),
    AssetType.SOLAR_INVERTER: _pv_bowtie(),
    AssetType.GAS_TURBINE: _gt_bowtie(),
    AssetType.HV_TRANSFORMER: _tr_bowtie(),
}

_MITIGATION_LIBRARY: dict[str, list[str]] = {
    "bearing_failure": [
        "Confirm grease/oil sample for ferrography within 24h",
        "Schedule bearing replacement inside the P50 RUL window (low-wind period)",
        "Increase CMS vibration sampling to 1/min until intervention",
        "Pre-stage crane + bearing kit to cut outage duration",
    ],
    "compressor_degradation": [
        "Schedule offline crank water wash at next planned stop",
        "Inspect inlet filters and vent-line oil mist source",
        "Trend corrected fuel flow weekly until wash",
    ],
    "string_underperformance": [
        "Dispatch robotic/soiling cleaning crew within 2 weeks",
        "Verify MPPT tracker firmware against fleet baseline",
        "Correlate yield loss with soiling station measurements",
    ],
    "overheating": [
        "Verify all radiator fans/pumps running; reset tripped stages",
        "Take DGA sample to rule out winding involvement",
        "Prepare load-transfer plan if top-oil exceeds 75°C",
    ],
}


class RiskCalculatorInput(BaseToolInput):
    """Input for RiskCalculatorTool."""

    asset_id: str = Field(description="Fleet asset id")
    severity: str = Field(default="MED", description="LOW | MED | HIGH | CRITICAL")
    degraded_barriers: list[str] = Field(
        default_factory=list,
        description="Barrier names known/assumed degraded (health 0.5); e.g. from inspection",
    )
    failed_barriers: list[str] = Field(
        default_factory=list,
        description="Barrier names known failed (health 0.0)",
    )


class RiskCalculatorOutput(BaseToolOutput):
    """Bow-tie risk evaluation."""

    asset_id: str = ""
    hazard: str = ""
    top_event: str = ""
    risk_score: float = 0.0
    risk_band: str = "LOW"
    top_3_barriers: list[str] = Field(default_factory=list)
    recommended_mitigations: list[str] = Field(default_factory=list)
    residual_likelihood: float = 0.0


class RiskCalculatorTool(EnergyForgeTool[RiskCalculatorInput, RiskCalculatorOutput]):
    """Evaluate bow-tie risk for an asset's dominant hazard.

    Combines severity-derived likelihood with barrier effectiveness/health to
    a 0–100 residual risk score, ranks the top-3 barriers by leverage gap,
    and returns hazard-specific recommended mitigations.
    """

    name: ClassVar[str] = "risk_calculator"
    description: ClassVar[str] = (
        "Bow-tie risk model for an asset's dominant hazard. Inputs: asset_id, "
        "severity (LOW/MED/HIGH/CRITICAL), optional degraded/failed barrier "
        "names. Returns risk_score (0–100), risk band, top_3_barriers by "
        "leverage, and recommended_mitigations."
    )
    input_model: ClassVar[type[BaseToolInput]] = RiskCalculatorInput
    output_model: ClassVar[type[BaseToolOutput]] = RiskCalculatorOutput

    async def _arun(self, tool_input: RiskCalculatorInput) -> RiskCalculatorOutput:
        asset = get_asset(tool_input.asset_id.strip().upper())
        bowtie = _BOWTIES[asset.asset_type]
        severity = tool_input.severity.strip().upper()
        if severity not in _SEVERITY_LIKELIHOOD:
            severity = "MED"

        degraded = {b.lower() for b in tool_input.degraded_barriers}
        failed = {b.lower() for b in tool_input.failed_barriers}
        barriers = [b.model_copy() for b in bowtie.barriers]
        for barrier in barriers:
            name = barrier.name.lower()
            if any(f in name for f in failed):
                barrier.health = 0.0
            elif any(d in name for d in degraded):
                barrier.health = 0.5

        likelihood = _SEVERITY_LIKELIHOOD[severity]
        consequence = bowtie.consequence_class
        for barrier in barriers:
            effective = barrier.effectiveness * barrier.health
            if barrier.side == "preventive":
                likelihood *= 1.0 - effective
            else:
                consequence *= 1.0 - 0.5 * effective

        risk_score = round(min(100.0, likelihood * consequence * 10.0), 2)
        band = (
            "SEVERE" if risk_score >= 40 else
            "HIGH" if risk_score >= 20 else
            "MODERATE" if risk_score >= 8 else "LOW"
        )
        leverage = sorted(
            barriers, key=lambda b: b.effectiveness * (1.0 - b.health), reverse=True
        )
        return RiskCalculatorOutput(
            asset_id=asset.asset_id,
            hazard=bowtie.hazard,
            top_event=bowtie.top_event,
            risk_score=risk_score,
            risk_band=band,
            top_3_barriers=[f"{b.name} (health {b.health:.1f}, eff {b.effectiveness:.2f})"
                            for b in leverage[:3]],
            recommended_mitigations=_MITIGATION_LIBRARY[bowtie.hazard],
            residual_likelihood=round(likelihood, 4),
            confidence=0.85,
        )
