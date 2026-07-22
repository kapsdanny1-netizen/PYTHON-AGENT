"""EnergyForge Agent — Streamlit operations dashboard.

Three pages (sidebar radio):

* **Live Monitor** — per-asset Plotly sensor charts with anomaly markers,
  KPI cards (availability %, MTBF, active WOs)
* **Chat with EnergyForge** — st.chat interface streaming orchestrator
  progress; expandable agent traces; HITL approval widget
* **Reports** — generated PDF artifacts + trend charts

State lives in ``st.session_state`` (conversation, selected asset/window,
pending HITL) so chat interactions never wipe context. All async work runs
on the dashboard bridge loop (dashboard/bridge.py); reruns never re-create
the DB pool.

Run:  ``streamlit run dashboard/app.py``  (or the compose ``app`` service)
"""

from __future__ import annotations

import queue
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:  # annotations only — keep import-time light
    import pandas as pd

st.set_page_config(
    page_title="EnergyForge Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — asset / window / LLM provider / navigation
# ─────────────────────────────────────────────────────────────────────────────


def _sidebar() -> tuple[str, float, str]:
    from config.settings import LLMProvider, get_settings
    from data.generators import FLEET

    settings = get_settings()
    st.sidebar.title("⚡ EnergyForge")
    st.sidebar.caption("Industrial energy operations copilot")

    assets = sorted(FLEET)
    default_idx = assets.index("WT-07") if "WT-07" in assets else 0
    asset = st.sidebar.selectbox("Asset", assets, index=default_idx, key="asset")
    window_hours = float(
        st.sidebar.select_slider(
            "Time window (hours)", options=[6, 12, 24, 48, 72, 168], value=24, key="window"
        )
    )

    providers = [p.value for p in LLMProvider]
    current = settings.llm_provider.value
    provider = st.sidebar.selectbox(
        "LLM provider", providers, index=providers.index(current), key="llm_provider",
        help="Maps to LLM_PROVIDER. Changing takes effect on the next turn.",
    )
    if provider != current:
        _switch_llm_provider(provider)
    st.sidebar.caption(f"model: `{settings.resolved_llm_model}`")

    st.sidebar.divider()
    page = st.sidebar.radio(
        "Page", ["📈 Live Monitor", "💬 Chat with EnergyForge", "📄 Reports"], key="page"
    )
    return asset, window_hours, page


def _switch_llm_provider(provider: str) -> None:
    """Hot-switch LLM_PROVIDER: env override + settings/graph cache reset."""
    import os

    from config.settings import get_settings
    from orchestrator import reset_graph_cache

    os.environ["LLM_PROVIDER"] = provider
    get_settings.cache_clear()
    reset_graph_cache()
    st.sidebar.toast(f"LLM provider → {provider} (next turn)", icon="🔁")


# ─────────────────────────────────────────────────────────────────────────────
# Page 1 — Live Monitor
# ─────────────────────────────────────────────────────────────────────────────


def _kpi_cards(asset: str, window_hours: float) -> None:
    from dashboard.queries import kpi_bundle_sync

    bundle = kpi_bundle_sync(asset, window_hours)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Availability", f"{bundle['availability_pct']}%")
    col2.metric("MTBF", f"{bundle['mtbf_hours']} h")
    col3.metric("Active WOs", bundle["active_wos"])
    col4.metric("Anomaly events (window)", bundle["events_in_window"])


def _monitor_chart(asset: str, window_hours: float, source_out: "st.container") -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from dashboard.queries import fetch_anomaly_events_sync, fetch_frame_sync

    frame, source = fetch_frame_sync(asset, window_hours)
    if source != "timescaledb":
        source_out.caption(f"⚠️ data source: {source} (seed the DB for live data)")
    channels = [c for c in frame.columns if c != "time"]
    events = fetch_anomaly_events_sync(asset, window_hours)

    pdf = frame.to_pandas()
    fig = make_subplots(
        rows=len(channels), cols=1, shared_xaxes=True,
        subplot_titles=channels, vertical_spacing=0.05,
    )
    for i, channel in enumerate(channels, start=1):
        fig.add_trace(
            go.Scatter(x=pdf["time"], y=pdf[channel], mode="lines",
                       name=channel, line={"width": 1.4}),
            row=i, col=1,
        )
    for event in events[:10]:  # anomaly markers overlaid (red vertical lines)
        try:
            fig.add_vline(x=datetime.fromisoformat(str(event["time"])),
                          line_color="red", line_dash="dot", line_width=1,
                          opacity=0.7, row="all", col=1)
        except Exception:  # malformed event timestamps never break the chart
            continue
    fig.update_layout(
        height=180 * len(channels) + 80, showlegend=False,
        margin={"t": 40, "b": 20, "l": 40, "r": 20},
        title=f"{asset} — live sensor channels",
    )
    st.plotly_chart(fig, use_container_width=True)

    if events:
        with st.expander(f"⚠️ {len(events)} anomaly event(s) in window"):
            st.dataframe(events, use_container_width=True)


def page_live_monitor(asset: str, window_hours: float) -> None:
    st.header("📈 Live Monitor")
    source_container = st.container()
    _kpi_cards(asset, window_hours)
    try:
        _monitor_chart(asset, window_hours, source_container)
    except Exception as exc:  # infrastructure down ⇒ explain, don't crash
        st.error(
            "Sensor data unavailable — is TimescaleDB up and seeded? "
            f"`docker compose up -d db && python main.py seed-demo`\n\n`{exc}`"
        )

    with st.expander("🧾 Active work orders"):
        from dashboard.queries import fetch_work_orders_sync

        orders = fetch_work_orders_sync(open_only=True)
        if orders:
            st.dataframe(orders, use_container_width=True)
        else:
            st.info("No open work orders.")


# ─────────────────────────────────────────────────────────────────────────────
# Page 2 — Chat with EnergyForge
# ─────────────────────────────────────────────────────────────────────────────

_CHAT_KEY = "ef_messages"


def _messages() -> list[dict[str, object]]:
    if _CHAT_KEY not in st.session_state:
        st.session_state[_CHAT_KEY] = []
    return st.session_state[_CHAT_KEY]  # type: ignore[return-value]


def _render_trace(traces: dict[str, list[dict[str, object]]]) -> None:
    for agent, records in traces.items():
        icon = "✅" if records else "⚠️"
        with st.expander(f"{icon} {agent} — {len(records)} tool call(s)"):
            for record in records:
                ok_icon = "✅" if record.get("ok") else "❌"
                st.markdown(f"- {ok_icon} `{record.get('tool')}` — {record.get('summary', '')}")


def _render_hitl_widget(hitl_queue: list[dict[str, object]], query: str, asset: str) -> None:
    if not hitl_queue:
        return
    st.warning("Human-in-the-loop gate engaged for this turn.")
    for item in hitl_queue:
        st.markdown(
            f"- `{item.get('item_id')}` — **{item.get('reason')}** → "
            f"status: `{item.get('status')}`"
        )
    if st.button("✅ Approve & re-run turn", key=f"approve_{len(_messages())}"):
        st.session_state["ef_preapprove"] = True
        st.session_state["ef_pending_query"] = query
        st.session_state["ef_pending_asset"] = asset
        st.rerun()


def _run_turn_streamed(query: str, asset: str) -> dict[str, object] | None:
    """Stream one orchestrator turn, rendering progress as nodes complete."""
    from orchestrator import stream_turn

    bridge_chunks = None
    preapproved = bool(st.session_state.pop("ef_preapprove", False))
    with st.status("EnergyForge is working…", expanded=True) as status:
        try:
            from dashboard.bridge import get_bridge

            bridge_chunks = get_bridge().stream(
                lambda: stream_turn(query, assets=[asset], preapproved=preapproved)
            )
            while True:
                try:
                    chunk = bridge_chunks.get(timeout=600)
                except queue.Empty:
                    st.error("Orchestrator stream stalled (600 s without output).")
                    return None
                if chunk is None:
                    break
                if isinstance(chunk, BaseException):
                    st.error(f"turn failed: {chunk}")
                    return None
                if chunk.get("done"):
                    status.update(label="Turn complete", state="complete")
                    return chunk["state"]  # type: ignore[return-value]
                node = chunk.get("node", "?")
                status.write(f"✔ `{node}` finished — keys: {chunk.get('delta_keys')}")
            status.update(label="Turn complete", state="complete")
        except Exception as exc:  # surface infra failures cleanly in chat
            st.error(f"orchestrator unavailable: {exc}")
            return None
    return None


def page_chat(asset: str) -> None:
    st.header("💬 Chat with EnergyForge")
    st.caption("Ask about asset health, anomalies, maintenance, safety or reports.")

    history = _messages()
    for message in history:
        with st.chat_message(str(message["role"])):
            st.markdown(str(message["content"]))
            if message.get("agent_traces"):
                _render_trace(message["agent_traces"])  # type: ignore[arg-type]

    # HITL re-run request (set by the approval widget on the previous run)
    pending = st.session_state.pop("ef_pending_query", None)
    pending_asset = st.session_state.pop("ef_pending_asset", asset)
    prompt = st.chat_input(f"Ask about {asset} or the fleet…")
    query = pending or prompt

    if query:
        used_asset = pending_asset if pending else asset
        with st.chat_message("user"):
            st.markdown(str(query))
            history.append({"role": "user", "content": str(query)})

        with st.chat_message("assistant"):
            final_state = _run_turn_streamed(str(query), str(used_asset))
            if final_state is None:
                failure = "The turn failed — check infrastructure and provider keys."
                st.markdown(failure)
                history.append({"role": "assistant", "content": failure})
            else:
                response = final_state.get("final_response") or "(no response)"
                st.markdown(str(response))
                traces = final_state.get("agent_traces") or {}
                if traces:
                    _render_trace(traces)  # type: ignore[arg-type]
                hitl_queue = final_state.get("hitl_queue") or []
                _render_hitl_widget(hitl_queue, str(query), str(used_asset))  # type: ignore[arg-type]
                history.append({
                    "role": "assistant",
                    "content": str(response),
                    "agent_traces": traces,
                    "trace_id": final_state.get("trace_id", ""),
                })
    else:
        st.info(
            f"Chatting in the context of **{asset}**. Try: "
            f"*“{asset} shows elevated bearing vibration at 3× normal for 6 hours. "
            "Bearing temperature rising 2°C/hr.”*"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Page 3 — Reports
# ─────────────────────────────────────────────────────────────────────────────


def _trend_chart(asset: str) -> None:
    import plotly.graph_objects as go
    import polars as pl

    from dashboard.queries import fetch_frame_sync

    frame, _ = fetch_frame_sync(asset, 168.0)
    channels = [c for c in frame.columns if c != "time"]
    daily = frame.group_by_dynamic("time", every="1d").agg(
        [pl.col(c).mean().alias(c) for c in channels]
    ).to_pandas()
    fig = go.Figure()
    for channel in channels:
        fig.add_trace(go.Scatter(x=daily["time"], y=daily[channel],
                                 mode="lines+markers", name=channel))
    fig.update_layout(title=f"{asset} — 7-day daily-mean trends",
                      height=380, margin={"t": 50, "b": 30})
    st.plotly_chart(fig, use_container_width=True)


def page_reports(asset: str) -> None:
    from dashboard.queries import list_reports_sync

    st.header("📄 Reports")
    reports = list_reports_sync()
    if not reports:
        st.info("No reports yet — ask the chat for an incident report on an anomaly.")
    for path in reports:
        col_name, col_size, col_dl = st.columns([6, 2, 2])
        col_name.markdown(f"📄 `{path.name}`")
        col_size.caption(f"{path.stat().st_size / 1024:.1f} KB")
        col_dl.download_button(
            "⬇️ Download", data=path.read_bytes(), file_name=path.name,
            mime="application/pdf", key=f"dl_{path.name}",
        )

    st.divider()
    st.subheader("Trend charts (7 days)")
    try:
        _trend_chart(asset)
    except Exception as exc:
        st.warning(f"Trend data unavailable: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    asset, window_hours, page = _sidebar()
    if page.startswith("📈"):
        page_live_monitor(asset, window_hours)
    elif page.startswith("💬"):
        page_chat(asset)
    else:
        page_reports(asset)


main()
