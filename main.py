"""EnergyForge Agent — CLI entry point.

Sub-commands are registered phase by phase; the full demo pipeline
(``energyforge demo``) lands with the Phase 6 integration scenario.

Available now (Phase 0):
    energyforge version   — print config summary
    energyforge check     — validate configuration, fail fast on missing keys
"""

from __future__ import annotations

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


def cli_main() -> None:
    """Sync wrapper — the only place where we enter asyncio land from CLI."""
    configure_logging(get_settings().log_level)
    app()


if __name__ == "__main__":
    cli_main()
