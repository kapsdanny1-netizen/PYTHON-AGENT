# EnergyForge Agent

Production-grade autonomous multi-agent system for industrial energy operations
(wind turbines, solar inverters, gas turbines, HV transformers).

> **Build status:** this project is constructed in strict phases.
> ✅ **Phase 0 — project scaffold** · ✅ **Phase 1 — shared data layer**.
>
> The full README — quick-start, architecture diagram, env-var reference
> table, and agent extension guide — ships with the final phase.

## Verify the scaffold + data layer

```bash
cp .env.example .env                  # fill in your LLM provider key
docker compose up -d db chroma redis  # infrastructure (TimescaleDB, Chroma, Redis)
pip install -e ".[dev]"               # install the package + dev tools
python main.py check                  # validate configuration (fails fast)
python main.py migrate                # apply Alembic migration 0001_initial
python main.py seed-demo              # synthetic fleet history incl. WT-07 anomaly
python main.py seed-memory            # manuals + historical RCAs → Chroma
```

## Layout

| Directory        | Phase | Contents                                            |
| ---------------- | ----- | --------------------------------------------------- |
| `config/`        | 0     | Pydantic settings, single `LLM_PROVIDER` switch      |
| `memory/`        | 1     | Async SQLAlchemy + Alembic + Chroma vector store     |
| `data/`          | 1     | Synthetic sensor generators with injected anomalies  |
| `tools/`         | 2     | 9 tool classes (Pydantic in/out, `confidence` field) |
| `agents/`        | 3     | 5 CrewAI agents with tightly scoped tool sets        |
| `orchestrator/`  | 4     | LangGraph state machine with HITL gate + audit log   |
| `dashboard/`     | 5     | Streamlit ops dashboard (monitor / chat / reports)   |
| `tests/`         | 6     | End-to-end integration scenario (WT-07 bearing wear) |
