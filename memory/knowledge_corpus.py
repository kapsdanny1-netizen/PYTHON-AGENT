"""Built-in seed corpus for the Chroma long-term memory.

Two knowledge classes, tagged via metadata so agents can filter:

* ``type=manual`` — equipment manuals: operating envelopes & alarm thresholds
* ``type=rca``    — historical root-cause analyses of past incidents

RCA-WT-001 deliberately mirrors the Phase 6 integration scenario (wind-turbine
main-bearing wear) so retrieval-grounded diagnostics have something true to
find. In production this corpus would be loaded from the document management
system; here it ships in-code for a self-contained demo.
"""

from __future__ import annotations

from memory.vector_store import DocumentRecord

EQUIPMENT_MANUALS: list[DocumentRecord] = [
    DocumentRecord(
        id="MANUAL-WT-BRG-001",
        text=(
            "Wind turbine main shaft bearing — operating envelope. Vibration (ISO 10816-21, "
            "rms velocity at bearing housing): normal < 1.6 mm/s; alert 1.6-4.5 mm/s; alarm "
            "> 4.5 mm/s. A sustained reading at 3x the fleet baseline with bearing temperature "
            "rising faster than 1.5 C/hr indicates developing raceway damage. Bearing oil "
            "temperature limits: normal < 70 C, alarm 80 C, automatic trip 90 C. Do not wait "
            "for the trip threshold: schedule inspection at the alert band to avoid collateral "
            "gearbox damage."
        ),
        metadata={"type": "manual", "asset_type": "wind_turbine", "equipment": "main_bearing"},
    ),
    DocumentRecord(
        id="MANUAL-WT-CTRL-001",
        text=(
            "Wind turbine supervisory controls. Rotor speed envelope 8-18.5 rpm; pitch system "
            "stalls blades above rated wind to hold rated power. Pitch faults manifest as "
            "oscillation in rotor rpm with normal vibration signatures — distinguish from "
            "bearing faults (vibration rises across X/Y/Z simultaneously with temperature)."
        ),
        metadata={"type": "manual", "asset_type": "wind_turbine", "equipment": "controls"},
    ),
    DocumentRecord(
        id="MANUAL-PV-INV-001",
        text=(
            "Solar string inverter — datasheet and limits. MPPT window 600-820 V DC; peak "
            "efficiency 98.5%; string current limit 13 A continuous. Gradual efficiency decay "
            "below 96% under constant irradiance points to soiling/fouling or MPPT tracker "
            "aging; a sudden string-current step to zero indicates fuse or connector failure. "
            "Clean modules when soiling loss exceeds 3%."
        ),
        metadata={"type": "manual", "asset_type": "solar_inverter", "equipment": "inverter"},
    ),
    DocumentRecord(
        id="MANUAL-GT-CMP-001",
        text=(
            "Gas turbine compressor section — performance monitoring. Compressor fouling "
            "signature: pressure ratio decay > 2% over weeks, corrected fuel flow rising for "
            "constant load, exhaust gas temperature trending up. Recovery: offline water wash "
            "restores 80-95% of lost performance. Monitor and schedule washes on condition, "
            "not calendar."
        ),
        metadata={"type": "manual", "asset_type": "gas_turbine", "equipment": "compressor"},
    ),
    DocumentRecord(
        id="MANUAL-TR-OIL-001",
        text=(
            "HV power transformer — thermal limits (ONAN cooling). Top-oil temperature: normal "
            "< 65 C at rated load, alarm 75 C, trip 90 C. Load cycling 30-95% is within design. "
            "A step increase in oil temperature at constant load indicates cooling-stage "
            "failure (radiator fans/pumps) rather than winding degradation; check cooler "
            "status before derating."
        ),
        metadata={"type": "manual", "asset_type": "hv_transformer", "equipment": "cooling"},
    ),
]

HISTORICAL_RCAS: list[DocumentRecord] = [
    DocumentRecord(
        id="RCA-WT-001",
        text=(
            "RCA: main bearing inner-race spall, wind turbine WT-04. Signature: vibration on "
            "all three axes climbed to 3x baseline over ~6 hours while bearing temperature "
            "rose 2 C/hr (52 C to 67 C). Root cause: lubrication film breakdown from water "
            "contamination; confirmed by grease ferrography. Corrective action: bearing "
            "replaced during the next low-wind window, six days after detection, at 40% of "
            "the cost of a forced outage. No emergency shutdown required — temperatures "
            "stayed below the 80 C alarm. Detection-to-repair lead time is the value driver."
        ),
        metadata={"type": "rca", "asset_type": "wind_turbine", "equipment": "main_bearing"},
    ),
    DocumentRecord(
        id="RCA-WT-002",
        text=(
            "RCA: gearbox high-speed-stage tooth crack, wind turbine WT-02. Signature: "
            "sideband energy around gear-mesh frequency with vibration concentrated on the X "
            "axis only, temperature stable. Distinguished from main-bearing wear by the "
            "single-axis pattern and absence of thermal rise. Corrective action: up-tower "
            "gearbox repair within 14 days."
        ),
        metadata={"type": "rca", "asset_type": "wind_turbine", "equipment": "gearbox"},
    ),
    DocumentRecord(
        id="RCA-PV-001",
        text=(
            "RCA: string underperformance at PV plant after harmattan dust season. Signature: "
            "efficiency decaying ~0.5% per day under constant irradiance, DC voltage stable, "
            "string current proportionally reduced. Root cause: cement-like soiling layer. "
            "Corrective action: robotic dry cleaning; efficiency recovered to 98.1%. Soiling "
            "recurrence window: 6-9 weeks in dry season."
        ),
        metadata={"type": "rca", "asset_type": "solar_inverter", "equipment": "modules"},
    ),
    DocumentRecord(
        id="RCA-GT-001",
        text=(
            "RCA: gas turbine compressor fouling at GT-01. Signature: pressure ratio decayed "
            "3.1% over five weeks, fuel flow +1.9% at constant load, exhaust temperature +18 C. "
            "Root cause: ingested dust plus oil mist from adjacent vent line. Corrective "
            "action: offline crank wash; 92% of performance recovered. Preventive: inlet "
            "filter ΔP monitoring added to weekly rounds."
        ),
        metadata={"type": "rca", "asset_type": "gas_turbine", "equipment": "compressor"},
    ),
    DocumentRecord(
        id="RCA-TR-001",
        text=(
            "RCA: transformer cooling-stage failure, TR-01. Signature: step change of +12 C "
            "in top-oil temperature at constant 62% load, holding steady thereafter. Root "
            "cause: two of four radiator fans tripped on a seized bearing contactor. "
            "Corrective action: fans reset and contactor replaced within one shift; no "
            "derating required as temperature peaked at 71 C, below the 75 C alarm. Lesson: "
            "step changes at constant load point at cooling, gradual rises point at loading "
            "or winding issues."
        ),
        metadata={"type": "rca", "asset_type": "hv_transformer", "equipment": "cooling"},
    ),
]
