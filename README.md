# ⚡ EnergyForge Agent

Production-grade autonomous **multi-agent system for industrial energy operations**.
It watches a synthetic fleet (wind turbines, solar inverters, gas turbines, HV
transformers), detects anomalies, diagnoses root causes against a semantic memory
of manuals + historical RCAs, forecasts remaining useful life, raises work orders,
runs a bow-tie safety review, optimizes set-points, and renders PDF incident
reports — orchestrated by a LangGraph state machine with a human-in-the-loop gate.

## Quick-start (3 commands to a running demo)

```bash
# 1. configure: pick a provider key in .env (or LLM_PROVIDER=ollama, no key needed)
cp .env.example .env

# 2. start the stack (TimescaleDB + Chroma + Redis + Streamlit dashboard)
docker compose up -d --build

# 3. provision the demo: schema + synthetic fleet incl. the WT-07 anomaly + memory
docker compose exec app python main.py bootstrap
```

Then open **http://localhost:8501** — Live Monitor shows all assets; the Chat page
answers *“WT-07 shows elevated bearing vibration at 3× normal for 6 hours. Bearing
temperature rising 2°C/hr.”* end-to-end.

Bare-metal alternative: `pip install -e ".[dev]"`, point `.env` hosts at the
compose services, then `python main.py bootstrap` and
`streamlit run dashboard/app.py`. CLI: `python main.py ask "your question"`.

## Architecture

```
                         ┌──────────────────────── OPERATOR ─────────────────────────┐
                         │  Streamlit dashboard (8501)  │  energyforge CLI (typer)   │
                         └──────────────┬──────────────────────────▲─────────────────┘
                                        │ stream_turn (async)      │ final_response
              ┌─────────────────────────▼──────────────────────────┴────────────────┐
              │           LangGraph ORCHESTRATOR  (orchestrator/)                    │
              │  intent → plan → dispatch┐      ┌→ aggregator → HITL gate → format   │
              │      (rules+LLM ≤3/turn) │      │      (audit_log row per node)      │
              └──────────────────────────┼──────┼────────────────────────────────────┘
                 CrewAI agents (agents/) │      │      HITL: dev=stdin prod=Slack+Redis
   ┌─────────────┬──────────────┬────────┴─┬────┴───────┬────────────────┐
   ▼             ▼              ▼          ▼            ▼                ▼
┌────────┐ ┌───────────┐ ┌───────────┐ ┌─────────┐ ┌───────────┐  (each returns
│Diagnos-│ │Maintenance│ │  Safety   │ │Optimiza-│ │ Reporting │   Pydantic output
│ tics   │ │           │ │           │ │  tion   │ │           │   + confidence)
└───┬────┘ └────┬──────┘ └────┬──────┘ └────┬────┘ └────┬──────┘
    │ tools (tools/) — never raise, output confidence+error, owner-loop scheduled
    ▼
┌─────────────┐ ┌──────────────┐ ┌───────────┐ ┌──────────┐ ┌────────────┐
│sensor_query │ │anomaly_det.  │ │prognostics│ │weather   │ │setpoint_opt│
│  (SQL)      │ │IsoForest+MADz│ │Prophet+XGB│ │Open-Meteo│ │scipy LP    │
└──────┬──────┘ └──────┬───────┘ └─────┬─────┘ └──────────┘ └────────────┘
       │               │               │        ┌──────────┐ ┌────────────┐
       │               │               │        │work_order│ │document_gen│
       │               │               │        │(PG seq)  │ │MD→WeasyPDF │
       ▼               ▼               ▼        └────┬─────┘ └─────┬──────┘
┌──────────────────────────────┐  ┌─────────────────────────┐  reports/*.pdf
│ TimescaleDB (5432)           │  │ ChromaDB (8000)         │  ┌──────────┐
│ sensor_readings (hypertable) │  │ manuals + RCAs corpus   │  │ Redis    │
│ anomaly_events (hypertable)  │  │ (ONNX embeddings)       │  │ HITL bus │
│ work_orders · audit_log      │  └─────────────────────────┘  └──────────┘
└─────────────▲────────────────┘
              │ data/generators.py — 16-asset synthetic fleet, AR(1)+diurnal
                signals, bearing-wear / fouling / step-change injection,
                SCENARIO_WT07_BEARING anchors the integration test
```

**Chroma** (`chromadb/chroma:1.0.15`, vector memory) and **Redis** are part of the
compose stack; the dashboard container is built from this repo.

## Repository layout

```
config/settings.py        pydantic-settings, single LLM_PROVIDER switch
memory/db.py              async SQLAlchemy + ORM (4 tables, per-loop engines)
memory/alembic/           migration 0001_initial (2 hypertables + WO sequence)
memory/vector_store.py    Chroma async wrapper (+ knowledge_corpus.py)
memory/knowledge_corpus.py 5 manuals + 5 RCAs (RCA-WT-001 mirrors WT-07)
data/generators.py        fleet registry, signal builders, anomaly injection
tools/                    9 tools on the EnergyForgeTool contract
agents/                   5 CrewAI agents (scoped tools, Pydantic outputs)
orchestrator/             EnergyForgeState, nodes, HITL, LangGraph assembly
dashboard/                Streamlit (bridge/queries/app)
tests/                    WT-07 end-to-end scenario (a–f assertions)
main.py                   typer CLI: check/migrate/seed-demo/seed-memory/ask/bootstrap
```

## Environment variable reference

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `ENVIRONMENT` | — | `dev` | `dev`/`test`/`prod` — toggles console→Slack notifications, stdin→webhook HITL |
| `LOG_LEVEL` | — | `INFO` | structlog JSON level |
| `RANDOM_SEED` | — | `42` | synthetic generator + ML seeds |
| `LLM_PROVIDER` | — | `openai` | **`openai` / `anthropic` / `grok` / `ollama` — the single switch** |
| `LLM_MODEL` | — | provider default | override (e.g. `gpt-4o-mini` for fast tests) |
| `LLM_TEMPERATURE` / `LLM_MAX_TOKENS` / `LLM_REQUEST_TIMEOUT_S` | — | `0.1` / `2048` / `60` | LLM call shaping |
| `OPENAI_API_KEY` | ✅ if provider=openai | — | OpenAI auth (`OPENAI_BASE_URL` optional proxy) |
| `ANTHROPIC_API_KEY` | ✅ if provider=anthropic | — | Anthropic auth |
| `GROK_API_KEY` | ✅ if provider=grok | — | xAI auth (`GROK_BASE_URL=https://api.x.ai/v1`) |
| `OLLAMA_BASE_URL` | — | `http://localhost:11434` | local Ollama (no key; `ollama pull llama3.1:8b`) |
| `POSTGRES_HOST/PORT/USER/DB` | ✅ | localhost values | TimescaleDB connection (compose overrides host→`db`) |
| `POSTGRES_PASSWORD` | ✅ | `energyforge` | DB auth — change beyond local dev |
| `DB_POOL_SIZE` / `DB_POOL_MAX_OVERFLOW` | — | `10` / `10` | asyncpg pool |
| `CHROMA_HOST/PORT/SSL/COLLECTION` | — | localhost:8000 | vector memory endpoint |
| `REDIS_URL` | — | `redis://localhost:6379/0` | HITL bus + cache |
| `SLACK_WEBHOOK_URL` | ✅ in prod | — | NotificationTool + HITL alerts |
| `NOTIFICATION_CHANNEL` | — | `#energyforge-ops` | Slack channel |
| `HITL_CONFIDENCE_THRESHOLD` | — | `0.75` | gate pauses below this confidence |
| `MAX_LLM_CALLS_PER_TURN` | — | `3` | orchestrator LLM budget (tool calls unlimited) |
| `TOOL_TIMEOUT_S` | — | `30` | per-tool guard |

## Testing

```bash
docker compose up -d db chroma redis
pytest -m integration -v      # WT-07 scenario, real LLM calls, <90s budget
```

The test seeds `WT-01/03/07` (72 h @ 15 min, anchored 3×-vibration + 2 °C/hr
bearing wear in the last 6 h), runs the full graph and asserts: diagnosis names
bearing degradation (confidence ≥ 0.80) · work order `WO-00xxxx` due ≤ 7 days ·
**no** emergency shutdown (severity HIGH < CRITICAL) · valid multi-page PDF ·
elapsed < 90 s · ≥ 5 audit rows for the trace. Without an LLM key the suite
self-skips.

## Extending: add a new agent (5 steps)

1. **Output model** in `agents/your_agent.py` subclassing `AgentOutputBase`
   (confidence is mandatory).
2. **Tool set** — pick from `tools/` or create one subclassing
   `EnergyForgeTool` (Pydantic in/out schemas + `_arun`; `run()` already
   guarantees timeout/never-raise).
3. **`run_your_agent()`** — call `agents.base.run_agent()` with a ≤3-sentence
   role/goal/backstory and a strict, numbered task procedure.
4. **Wire it up** — add a node in `orchestrator/nodes.py` (decorate with
   `@_audited("agent_yours")` for the audit row), register it in
   `CANONICAL_AGENTS`, `_INTENT_TO_AGENTS` and graph.py's `_AGENT_NODES`.
5. **Planner + dashboard** — extend `_INTENT_KEYWORDS` so queries route to it;
   traces and outputs surface automatically.

Everything else (audit rows, HITL thresholds, streaming events, chat rendering)
is inherited from the harness.

## Design invariants (enforced throughout)

- Tools **never raise** — failures arrive as `output.error` with `confidence=0`.
- Agents return **validated Pydantic** or raise `AgentExecutionError`
  (contained by the orchestrator; the turn continues).
- Every node writes an **audit_log** row before returning; every log line is
  JSON with `asset_id`/`trace_id` when in context.
- **Shutdown doctrine** (safety) and **artifact/WO guarantees**
  (reporting/maintenance) are enforced in code, not entrusted to the LLM.
- No secrets in code; `.env` is git-ignored; all SQL is ORM/parameterised and
  tool inputs are registry-sanitised before touching queries.
