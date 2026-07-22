"""EnergyForge Agent — CLI entry point.

Sub-commands are registered phase by phase; the full demo pipeline
(``energyforge demo``) lands with the Phase 6 integration scenario.

Available now:
    energyforge version    — print config summary
    energyforge check      — validate configuration, fail fast on missing keys
    energyforge migrate    — apply Alembic migrations against TimescaleDB
    energyforge seed-demo  — generate synthetic fleet history (incl. WT-07 scenario)
    energyforge seed-memory — load equipment manuals + RCAs into Chroma
"""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console

from config.settings import get_settings
from logging_config import configure_logging

app = typer.Typer(help="EnergyForge Agent CLI", no_args_is_help=True)
console = Console()


@app.command()
def version() -> None:
    """Print environment and active LLM configuration summary."""
    settings = get_settings()
    console.print(
        f"[bold cyan]{settings.app_name}[/bold cyan] — "
        f"env={settings.environment.value} "
        f"llm={settings.llm_provider.value}:{settings.resolved_llm_model}"
    )


@app.command()
def check() -> None:
    """Validate configuration; exit non-zero if the LLM provider lacks credentials."""
    settings = get_settings()
    runtime = settings.llm_runtime()  # raises ConfigurationError on misconfig
    console.print(
        f"[green]✓[/green] LLM provider ok: {runtime.provider.value} "
        f"(model={runtime.model})"
    )
    console.print(
        f"[green]✓[/green] database: "
        f"{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
    )
    console.print(
        f"[green]✓[/green] chroma: {settings.chroma_host}:{settings.chroma_port} "
        f"(collection={settings.chroma_collection})"
    )
    console.print(f"[green]✓[/green] redis: {settings.redis_url}")
    console.print("[bold]Configuration valid.[/bold]")


@app.command()
def migrate() -> None:
    """Apply all pending Alembic migrations against TimescaleDB."""
    from memory.db import run_migrations

    asyncio.run(run_migrations())
    console.print("[green]✓[/green] schema migrated to head (0001_initial)")


@app.command("seed-demo")
def seed_demo(
    hours: float = typer.Option(168.0, help="Lookback window of synthetic history, in hours"),
    freq: int = typer.Option(10, help="Sampling interval in minutes"),
) -> None:
    """Generate the synthetic fleet history, including the WT-07 bearing-wear scenario."""
    from data.generators import GenerateRequest, generate_and_store

    summary = asyncio.run(generate_and_store(GenerateRequest(hours=hours, freq_minutes=freq)))
    console.print(
        f"[green]✓[/green] seeded {summary.total_rows} rows across "
        f"{len(summary.per_asset_rows)} assets "
        f"(anomalies: {', '.join(summary.anomalies_applied)}) — trace={summary.trace_id}"
    )


@app.command("seed-memory")
def seed_memory() -> None:
    """Load the built-in equipment manuals + historical RCAs into Chroma."""
    from memory.vector_store import VectorStore

    added = asyncio.run(VectorStore().seed_default_corpus())
    console.print(f"[green]✓[/green] vector memory seeded with {added} documents")


def cli_main() -> None:
    """Sync wrapper — the only place where we enter asyncio land from CLI."""
    configure_logging(get_settings().log_level)
    app()


if __name__ == "__main__":
    cli_main()
