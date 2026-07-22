"""Orchestrator graph nodes.

Flow:  intent_classifier → task_planner → dispatch ⇆ agent_* → aggregator
        → hitl_gate → response_formatter → END

Cross-cutting guarantees implemented here:

* **Every node writes to ``audit_log`` before returning** (the
  :func:`_audited` decorator; DB failures degrade to a log warning in dev,
  never crash the turn — but are real rows when a database is attached).
* **LLM budget**: orchestrator-level LLM calls (only the intent-classifier
  fallback) are charged against ``settings.max_llm_calls_per_turn`` (=3);
  exceeding it raises :class:`~exceptions.OrchestrationError`. Agent-crew
  iterations are bounded independently (crew ``max_iter``); tool calls are
  unlimited per policy.
* Agent failures are contained: a crashed agent lands in ``agent_outputs``
  as ``{"error": ...}`` and the pipeline continues (aggregator discounts it).
"""

from __future__ import annotations

import functools
import json
import re
from collections.abc import Awaitable, Callable
from typing import TypeVar

from agents import (
    AgentRunResult,
    llm_complete,
    run_diagnostics,
    run_maintenance,
    run_optimization,
    run_reporting,
    run_safety,
)
from agents.base import AgentOutputBase
from config.settings import Settings, get_settings
from data.generators import FLEET
from exceptions import AgentExecutionError, EnergyForgeError, OrchestrationError
from logging_config import bind_log_context, clear_log_context, get_logger
from memory.db import record_audit, session_scope
from orchestrator.hitl import ApprovalDecision, make_hitl_item, request_approval
from orchestrator.state import (
    CANONICAL_AGENTS,
    SEVERITY_ORDER,
    EnergyForgeState,
)

logger = get_logger(__name__)

StateUpdate = dict[str, object]
NodeFn = Callable[[EnergyForgeState], Awaitable[StateUpdate]]

# ─────────────────────────────────────────────────────────────────────────────
# Audit + context plumbing
# ─────────────────────────────────────────────────────────────────────────────

F = TypeVar("F", bound=NodeFn)


async def _audit(state: EnergyForgeState, node: str, payload: dict[str, object]) -> None:
    """Append one audit_log row for the node (best effort)."""
    asset = state["active_assets"][0] if state["active_assets"] else None
    try:
        async with session_scope() as session:
            await record_audit(
                session,
                trace_id=state["trace_id"],
                actor="orchestrator",
                action=f"node:{node}",
                asset_id=asset,
                payload=payload,
            )
    except EnergyForgeError as exc:
        logger.warning("audit.unavailable", node=node, error=str(exc)[:200])


def _audited(node_name: str) -> Callable[[F], F]:
    """Decorator: write the audit row AFTER the node body, BEFORE returning."""

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        async def wrapper(state: EnergyForgeState) -> StateUpdate:
            bind_log_context(trace_id=state["trace_id"])
            try:
                update = await fn(state)
            finally:
                clear_log_context()
            await _audit(state, node_name, {"result_keys": sorted(update)})
            return update

        return wrapper  # type: ignore[return-value]

    return decorator


def _event(state: EnergyForgeState, message: str) -> list[str]:
    return [*state["events"], message]


# ─────────────────────────────────────────────────────────────────────────────
# intent_classifier
# ─────────────────────────────────────────────────────────────────────────────

_INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "diagnose": ("vibrat", "anomal", "fault", "degrad", "elevated", "abnormal",
                 "bearing", "wear", "temperature ris", "why"),
    "optimize": ("optimi", "maximi", "setpoint", "set-point", "set point",
                 "dispatch", "output mw", "increase generation"),
    "maintain": ("maintenance", "work order", "repair", "replace", "rul",
                 "service"),
    "safety": ("safety", "shutdown", "shut down", "hazard", "risk of", "trip"),
    "report": ("report", "pdf", "document", "write-up", "writeup"),
}

_INTENT_ORDER = ("diagnose", "maintain", "safety", "optimize", "report")
_ASSET_PATTERN = re.compile(r"\b([A-Z]{2,3}-\d{1,3})\b")


def _extract_assets(text: str) -> list[str]:
    found = {m for m in _ASSET_PATTERN.findall(text.upper()) if m in FLEET}
    return sorted(found)


def _classify_rules(query: str) -> list[str]:
    lowered = query.lower()
    intents = [
        intent for intent in _INTENT_ORDER
        if any(kw in lowered for kw in _INTENT_KEYWORDS[intent])
    ]
    return intents


async def _classify_llm(query: str, llm_calls_used: int) -> list[str]:
    """LLM fallback for ambiguous queries — counts against the turn budget."""
    intents = await llm_complete(
        "Classify the operator request into one or more of: diagnose, maintain, "
        "safety, optimize, report, chat. Reply with a comma-separated list, "
        "nothing else.\n\nRequest: " + query,
    )
    chosen = [i.strip() for i in intents.lower().split(",") if i.strip() in (*_INTENT_ORDER, "chat")]
    if not chosen or chosen == ["chat"]:
        return ["chat"]
    return [i for i in _INTENT_ORDER if i in chosen]


@_audited("intent_classifier")
async def intent_classifier_node(state: EnergyForgeState) -> StateUpdate:
    """Determine intents + active assets (rules first, LLM fallback, budgeted)."""
    settings = get_settings()
    query = state["user_query"]
    assets = state["active_assets"] or _extract_assets(query)
    intents = _classify_rules(query)

    llm_used = state["llm_calls_used"]
    if not intents:
        if llm_used + 1 > settings.max_llm_calls_per_turn:
            raise OrchestrationError(
                "per-turn LLM budget exhausted before planning",
                context={"used": llm_used, "cap": settings.max_llm_calls_per_turn},
            )
        llm_used += 1
        intents = await _classify_llm(query, llm_used)
        logger.info("orchestrator.intent_llm_fallback", intents=intents)

    if not assets:
        assets = _extract_assets(query)  # retry post-LLM for dominantly-chat turns
    message = f"intent={intents or ['chat']} assets={assets or ['none']}"
    logger.info("orchestrator.intent", detail=message)
    return {
        "intents": intents or ["chat"],
        "active_assets": assets,
        "llm_calls_used": llm_used,
        "events": _event(state, f"classified: {message}"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# task_planner
# ─────────────────────────────────────────────────────────────────────────────

# Canonical execution order is CANONICAL_AGENTS; diagnose auto-attaches a
# report so every anomaly turn leaves an auditable artifact.
_INTENT_TO_AGENTS: dict[str, tuple[str, ...]] = {
    "diagnose": ("diagnostics", "maintenance", "safety", "reporting"),
    "maintain": ("maintenance",),
    "safety": ("safety",),
    "optimize": ("optimization",),
    "report": ("reporting",),
    "chat": (),
}


@_audited("task_planner")
async def task_planner_node(state: EnergyForgeState) -> StateUpdate:
    """Map intents → ordered agent plan (deterministic, canonical order)."""
    wanted: set[str] = set()
    for intent in state["intents"]:
        wanted.update(_INTENT_TO_AGENTS.get(intent, ()))
    plan = [agent for agent in CANONICAL_AGENTS if agent in wanted]
    logger.info("orchestrator.plan", plan=plan)
    return {
        "pending_tasks": plan,
        "events": _event(state, f"plan: {plan or ['no agents — direct answer']}"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Agent nodes
# ─────────────────────────────────────────────────────────────────────────────


def _agent_of(state: EnergyForgeState) -> str:
    """Asset under investigation (first active asset, else first wind turbine)."""
    return state["active_assets"][0] if state["active_assets"] else "WT-01"


def _store_agent_result(
    state: EnergyForgeState,
    agent_key: str,
    result: AgentRunResult[AgentOutputBase] | None,
    error: str | None,
) -> StateUpdate:
    outputs = dict(state["agent_outputs"])
    traces = dict(state["agent_traces"])
    events = list(state["events"])
    if result is not None:
        outputs[agent_key] = result.output.model_dump(mode="json")
        traces[agent_key] = [r.model_dump() for r in result.trace]
        events.append(
            f"{agent_key}: done (confidence={result.output.confidence:.2f}, "
            f"tools={len(result.trace)})"
        )
    else:
        outputs[agent_key] = {"error": error or "unknown failure", "confidence": 0.0}
        traces[agent_key] = []
        events.append(f"{agent_key}: FAILED — {error}")
    pending = state["pending_tasks"][1:]
    return {
        "agent_outputs": outputs,
        "agent_traces": traces,
        "pending_tasks": pending,
        "events": events,
    }


def _diagnosis_context(state: EnergyForgeState) -> tuple[str, str]:
    diag = state["agent_outputs"].get("diagnostics", {})
    return (
        f"root_cause={diag.get('root_cause', 'n/a')} | evidence={diag.get('evidence', [])}",
        str(diag.get("severity", "MED")).upper(),
    )


async def _run_agent_guarded(
    agent_key: str,
    settings: Settings,
    call: Callable[[], Awaitable[AgentRunResult[AgentOutputBase]]],
) -> tuple[AgentRunResult[AgentOutputBase] | None, str | None]:
    try:
        return await call(), None
    except AgentExecutionError as exc:
        return None, str(exc)
    except EnergyForgeError as exc:
        return None, f"{exc.code}: {exc.message}"
    except Exception as exc:  # broad: one agent must not sink the turn
        return None, f"unexpected {type(exc).__name__}: {str(exc)[:200]}"


@_audited("agent_diagnostics")
async def diagnostics_node(state: EnergyForgeState) -> StateUpdate:
    """Run the DiagnosticsAgent for the active asset."""
    settings = get_settings()
    result, error = await _run_agent_guarded(
        "diagnostics", settings,
        lambda: run_diagnostics(
            asset_id=_agent_of(state), trigger=state["user_query"], settings=settings,
        ),
    )
    return _store_agent_result(state, "diagnostics", result, error)


@_audited("agent_maintenance")
async def maintenance_node(state: EnergyForgeState) -> StateUpdate:
    """Run the MaintenanceAgent (feeds on the diagnosis)."""
    settings = get_settings()
    diagnosis, _ = _diagnosis_context(state)
    result, error = await _run_agent_guarded(
        "maintenance", settings,
        lambda: run_maintenance(
            asset_id=_agent_of(state), trigger=state["user_query"],
            diagnosis=diagnosis, settings=settings,
        ),
    )
    return _store_agent_result(state, "maintenance", result, error)


@_audited("agent_safety")
async def safety_node(state: EnergyForgeState) -> StateUpdate:
    """Run the SafetyAgent (independent review of the diagnosis)."""
    settings = get_settings()
    diagnosis, severity = _diagnosis_context(state)
    result, error = await _run_agent_guarded(
        "safety", settings,
        lambda: run_safety(
            asset_id=_agent_of(state), trigger=state["user_query"],
            diagnosis=diagnosis, severity=severity, settings=settings,
        ),
    )
    return _store_agent_result(state, "safety", result, error)


@_audited("agent_optimization")
async def optimization_node(state: EnergyForgeState) -> StateUpdate:
    """Run the OptimizationAgent."""
    settings = get_settings()
    result, error = await _run_agent_guarded(
        "optimization", settings,
        lambda: run_optimization(
            asset_id=_agent_of(state), trigger=state["user_query"], settings=settings,
        ),
    )
    return _store_agent_result(state, "optimization", result, error)


@_audited("agent_reporting")
async def reporting_node(state: EnergyForgeState) -> StateUpdate:
    """Run the ReportingAgent over the complete findings bundle."""
    settings = get_settings()
    bundle = json.dumps(state["agent_outputs"], indent=1, default=str)[:3500]
    result, error = await _run_agent_guarded(
        "reporting", settings,
        lambda: run_reporting(
            asset_id=_agent_of(state), trigger=state["user_query"],
            findings_bundle=bundle, settings=settings,
        ),
    )
    return _store_agent_result(state, "reporting", result, error)


# ─────────────────────────────────────────────────────────────────────────────
# aggregator → hitl_gate → response_formatter
# ─────────────────────────────────────────────────────────────────────────────


@_audited("aggregator")
async def aggregator_node(state: EnergyForgeState) -> StateUpdate:
    """Fold agent outputs into severity/confidence extrema + HITL decision."""
    severities = ["LOW"]
    confidences: list[float] = []
    for key, output in state["agent_outputs"].items():
        if "error" in output:
            continue
        if key == "diagnostics" and str(output.get("severity", "")).upper() in SEVERITY_ORDER:
            severities.append(str(output["severity"]).upper())
        try:
            confidences.append(float(output.get("confidence", 0.0)))
        except (TypeError, ValueError):
            continue
    max_severity = max(severities, key=lambda s: SEVERITY_ORDER[s])
    min_confidence = min(confidences, default=1.0)
    requires_hitl = (
        min_confidence < state["confidence_threshold"] or max_severity == "CRITICAL"
    )
    logger.info(
        "orchestrator.aggregate",
        max_severity=max_severity, min_confidence=min_confidence, hitl=requires_hitl,
    )
    return {
        "max_severity": max_severity,
        "min_confidence": min_confidence,
        "requires_hitl": requires_hitl,
        "events": _event(
            state,
            f"aggregated: severity={max_severity} min_confidence={min_confidence:.2f} "
            f"hitl={'yes' if requires_hitl else 'no'}",
        ),
    }


@_audited("hitl_gate")
async def hitl_gate_node(state: EnergyForgeState) -> StateUpdate:
    """Pause for human approval on low confidence or CRITICAL severity.

    Conditional routing policy: CRITICAL severity or any output below the
    confidence threshold triggers approval. Dev → stdin; prod → Slack+Redis;
    test/non-interactive → recorded auto-approve. Pre-approval (dashboard
    widget) short-circuits the gate on resume.
    """
    if not state["requires_hitl"]:
        return {"events": _event(state, "hitl_gate: pass (no approval needed)")}
    if state["hitl_preapproved"]:
        return {"events": _event(state, "hitl_gate: pre-approved by operator")}

    reasons: list[str] = []
    if state["min_confidence"] < state["confidence_threshold"]:
        reasons.append(
            f"min_confidence {state['min_confidence']:.2f} < {state['confidence_threshold']}"
        )
    if state["max_severity"] == "CRITICAL":
        reasons.append("severity CRITICAL present")
    item = make_hitl_item(
        reason="; ".join(reasons), agent="aggregator",
        detail=json.dumps(state["agent_outputs"], default=str)[:600],
    )
    decision = await request_approval(item, trace_id=state["trace_id"])
    item["status"] = decision.value
    queue = [*state["hitl_queue"], item]
    halted = decision is ApprovalDecision.REJECTED
    logger.info("hitl_gate.decision", decision=decision.value, halted=halted)
    return {
        "hitl_queue": queue,
        "events": _event(
            state,
            f"hitl_gate: {decision.value}" + (" — pipeline HALTED" if halted else ""),
        ),
        "final_response": (
            "Action halted pending human review (HITL rejected)." if halted else ""
        ),
    }


@_audited("response_formatter")
async def response_formatter_node(state: EnergyForgeState) -> StateUpdate:
    """Deterministically render the final operator-facing answer (no LLM)."""
    if state["final_response"]:  # HITL halt already composed the message
        return {}

    lines = [f"### EnergyForge turn `{state['trace_id'][:8]}`", ""]
    if state["active_assets"]:
        lines.append(f"**Assets:** {', '.join(state['active_assets'])}")
    lines.append(f"**Severity:** {state['max_severity']} · "
                 f"**Min confidence:** {state['min_confidence']:.2f} · "
                 f"**HITL:** {'required' if state['requires_hitl'] else 'not required'}")
    lines.append("")

    outputs = state["agent_outputs"]
    if "diagnostics" in outputs:
        diag = outputs["diagnostics"]
        lines.append("#### 🔍 Diagnostics")
        if "error" in diag:
            lines.append(f"_failed: {diag['error']}_")
        else:
            lines.append(f"- **Root cause:** {diag.get('root_cause', 'n/a')}")
            lines.append(f"- **Severity:** {diag.get('severity')} "
                         f"(confidence {float(diag.get('confidence', 0)):.2f})")
            for item in list(diag.get("evidence", []))[:5]:
                lines.append(f"  - {item}")
        lines.append("")
    if "maintenance" in outputs:
        maint = outputs["maintenance"]
        lines.append("#### 🛠 Maintenance")
        if "error" in maint:
            lines.append(f"_failed: {maint['error']}_")
        else:
            lines.append(
                f"- **{maint.get('priority')}** — {maint.get('action', 'n/a')}"
            )
            lines.append(f"- Work order: `{maint.get('wo_id', 'PENDING')}` · "
                         f"action within {maint.get('tta_days', '?')} day(s)")
        lines.append("")
    if "safety" in outputs:
        safety = outputs["safety"]
        lines.append("#### 🦺 Safety")
        if "error" in safety:
            lines.append(f"_failed: {safety['error']}_")
        else:
            lines.append(f"- Risk score: **{safety.get('risk_score')}**/100 · "
                         f"shutdown required: **{safety.get('requires_shutdown')}**")
            for action in list(safety.get("immediate_actions", []))[:4]:
                lines.append(f"  - {action}")
        lines.append("")
    if "optimization" in outputs:
        plan = outputs["optimization"]
        lines.append("#### ⚡ Optimization")
        if "error" in plan:
            lines.append(f"_failed: {plan['error']}_")
        else:
            lines.append(f"- Set-points: `{plan.get('set_points', {})}` · "
                         f"Δ {plan.get('delta_mw', 0)} MW")
        lines.append("")
    if "reporting" in outputs:
        report = outputs["reporting"]
        lines.append("#### 📄 Report")
        if "error" in report:
            lines.append(f"_failed: {report['error']}_")
        else:
            lines.append(f"- `{report.get('report_type', 'incident')}` → "
                         f"{report.get('file_path', 'n/a')}")
            lines.append(f"- {report.get('summary', '')[:400]}")
        lines.append("")
    if state["hitl_queue"]:
        lines.append("#### 👤 HITL decisions")
        for item in state["hitl_queue"]:
            lines.append(f"- `{item['item_id']}` {item['reason']} → **{item['status']}**")

    response = "\n".join(lines).strip()
    conversation = [
        *state["conversation"],
        {"role": "user", "content": state["user_query"], "ts": ""},
        {"role": "assistant", "content": response, "ts": ""},
    ]
    return {"final_response": response, "conversation": conversation}
