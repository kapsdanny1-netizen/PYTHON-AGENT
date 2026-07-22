"""Orchestrator state definitions (LangGraph ``StateGraph`` state schema).

``EnergyForgeState`` threads one operator turn through the whole graph:
what the user asked, which assets and agents are in play, what each agent
produced, the HITL queue, and governance counters (LLM budget, HITL flag).

Nodes return partial dicts; LangGraph replaces the listed keys (no reducers
needed — every field is fully rewritten by the node that owns it).
"""

from __future__ import annotations

from typing import TypedDict

from config.settings import Settings, get_settings
from logging_config import new_trace_id

Intent = str  # "diagnose" | "maintain" | "safety" | "optimize" | "report" | "chat"

CANONICAL_AGENTS: tuple[str, ...] = (
    "diagnostics", "maintenance", "safety", "optimization", "reporting",
)
SEVERITY_ORDER: dict[str, int] = {"LOW": 0, "MED": 1, "HIGH": 2, "CRITICAL": 3}


class Message(TypedDict):
    """One conversation message."""

    role: str  # "user" | "assistant" | "system"
    content: str
    ts: str


class HitlItem(TypedDict):
    """One human-in-the-loop approval request."""

    item_id: str
    reason: str
    agent: str
    detail: str
    status: str  # "pending" | "approved" | "rejected" | "auto_approved"


class StreamChunk(TypedDict):
    """Yielded by ``stream_turn`` — partial progress or the final view."""

    node: str
    delta_keys: list[str]
    done: bool


class EnergyForgeState(TypedDict):
    """Full orchestrator turn state."""

    user_query: str
    conversation: list[Message]
    active_assets: list[str]
    intents: list[Intent]
    pending_tasks: list[str]  # agent names still to execute (canonical order)
    agent_outputs: dict[str, dict[str, object]]  # agent -> output model_dump (json mode)
    agent_traces: dict[str, list[dict[str, object]]]  # agent -> ToolCallRecord dumps
    hitl_queue: list[HitlItem]
    hitl_preapproved: bool  # dashboard approval widget sets this on resume
    requires_hitl: bool
    max_severity: str
    min_confidence: float
    confidence_threshold: float
    llm_calls_used: int  # orchestrator-level LLM calls only; hard cap 3
    trace_id: str
    final_response: str
    events: list[str]  # progress events (also the streaming payload)


def initial_state(
    user_query: str,
    *,
    assets: list[str] | None = None,
    settings: Settings | None = None,
    preapproved: bool = False,
) -> EnergyForgeState:
    """Build the entry state for one orchestrator turn."""
    cfg = settings or get_settings()
    return EnergyForgeState(
        user_query=user_query,
        conversation=[],
        active_assets=[a.strip().upper() for a in (assets or [])],
        intents=[],
        pending_tasks=[],
        agent_outputs={},
        agent_traces={},
        hitl_queue=[],
        hitl_preapproved=preapproved,
        requires_hitl=False,
        max_severity="LOW",
        min_confidence=1.0,
        confidence_threshold=cfg.hitl_confidence_threshold,
        llm_calls_used=0,
        trace_id=new_trace_id(),
        final_response="",
        events=[],
    )
