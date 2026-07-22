"""LangGraph StateGraph assembly + turn entry points.

Topology::

    intent_classifier ─► task_planner ─► dispatch ⇄ agent_{diagnostics,
        maintenance, safety, optimization, reporting} ─► aggregator
        ─► hitl_gate ─► response_formatter ─► END

``dispatch`` re-routes after every agent node until ``pending_tasks`` is
empty, which keeps the audit log linear and lets the plan shrink as agents
complete. Streaming: :func:`stream_turn` yields ``{"node", "delta_keys",
"done"}`` chunks via ``stream_mode="updates"`` as each node finishes, and a
final accumulated full-state chunk — the dashboard consumes this for
partial-progress rendering.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import cast

from config.settings import Settings
from logging_config import bind_log_context, clear_log_context, get_logger
from orchestrator.nodes import (
    aggregator_node,
    diagnostics_node,
    hitl_gate_node,
    intent_classifier_node,
    maintenance_node,
    optimization_node,
    reporting_node,
    response_formatter_node,
    safety_node,
    task_planner_node,
)
from orchestrator.state import EnergyForgeState, StreamChunk, initial_state

logger = get_logger(__name__)

_AGENT_NODES = {
    "diagnostics": diagnostics_node,
    "maintenance": maintenance_node,
    "safety": safety_node,
    "optimization": optimization_node,
    "reporting": reporting_node,
}

_compiled_graph: object | None = None


def _next_node(state: EnergyForgeState) -> str:
    """Dispatch routing: next planned agent, or the aggregator when drained."""
    if state["pending_tasks"]:
        return state["pending_tasks"][0]
    return "aggregator"


def get_graph(settings: Settings | None = None) -> object:
    """Compile (and cache) the orchestrator StateGraph."""
    global _compiled_graph
    if _compiled_graph is not None:
        return _compiled_graph

    from langgraph.graph import END, StateGraph

    builder = StateGraph(EnergyForgeState)
    builder.add_node("intent_classifier", intent_classifier_node)
    builder.add_node("task_planner", task_planner_node)
    builder.add_node("dispatch", lambda state: {})  # routing junction
    for key, fn in _AGENT_NODES.items():
        builder.add_node(f"agent_{key}", fn)
    builder.add_node("aggregator", aggregator_node)
    builder.add_node("hitl_gate", hitl_gate_node)
    builder.add_node("response_formatter", response_formatter_node)

    builder.set_entry_point("intent_classifier")
    builder.add_edge("intent_classifier", "task_planner")
    builder.add_edge("task_planner", "dispatch")
    builder.add_conditional_edges(
        "dispatch",
        _next_node,
        {**{key: f"agent_{key}" for key in _AGENT_NODES}, "aggregator": "aggregator"},
    )
    for key in _AGENT_NODES:
        builder.add_edge(f"agent_{key}", "dispatch")
    builder.add_edge("aggregator", "hitl_gate")
    builder.add_edge("hitl_gate", "response_formatter")
    builder.add_edge("response_formatter", END)

    _compiled_graph = builder.compile().with_config(recursion_limit=50)
    logger.info("orchestrator.graph_compiled", nodes=8 + len(_AGENT_NODES))
    return _compiled_graph


def reset_graph_cache() -> None:
    """Drop the cached compiled graph (after settings changes)."""
    global _compiled_graph
    _compiled_graph = None


async def run_turn(
    query: str,
    *,
    assets: list[str] | None = None,
    settings: Settings | None = None,
    preapproved: bool = False,
) -> EnergyForgeState:
    """Execute one complete orchestrator turn and return the final state."""
    state = initial_state(query, assets=assets, settings=settings, preapproved=preapproved)
    bind_log_context(trace_id=state["trace_id"])
    logger.info("orchestrator.turn_start", query=query[:120], plan_hint=assets)
    graph = get_graph(settings)
    final = cast(
        EnergyForgeState,
        await graph.ainvoke(state),  # type: ignore[attr-defined]  # langgraph untyped
    )
    logger.info(
        "orchestrator.turn_end",
        severity=final["max_severity"], hitl=final["requires_hitl"],
        agents=list(final["agent_outputs"]),
    )
    clear_log_context()
    return final


async def stream_turn(
    query: str,
    *,
    assets: list[str] | None = None,
    settings: Settings | None = None,
    preapproved: bool = False,
) -> AsyncIterator[dict[str, object]]:
    """Stream one turn: partial chunk per completed node + final full state.

    Yields:
        ``{"node": <name>, "delta_keys": [...], "done": False}`` as nodes
        complete, then one ``{"node": "__end__", "state": <full>, "done": True}``.
    """
    state = initial_state(query, assets=assets, settings=settings, preapproved=preapproved)
    bind_log_context(trace_id=state["trace_id"])
    graph = get_graph(settings)
    view = dict(state)
    async for chunk in graph.astream(state, stream_mode="updates"):  # type: ignore[attr-defined]
        for node_name, delta in chunk.items():
            if isinstance(delta, dict):
                view.update(delta)
            yield StreamChunk(node=node_name, delta_keys=sorted(delta or {}), done=False)
    yield {"node": "__end__", "state": view, "done": True}
    clear_log_context()
