"""Integration-test fixtures.

Environment contract
--------------------
* ``ENVIRONMENT=test`` is forced here (before any project import) so that
  notifications render to console and HITL auto-approves — the pipeline is
  deterministic except for the real LLM calls under test.
* Tools are NOT mocked — the scenario exercises the real DB, real detectors,
  real LP, real PDF renderer. Only genuinely external APIs would be mocked;
  the diagnose→maintain→safety→report plan makes no external calls beyond
  the LLM provider, which is intentionally REAL (assertion-grade behaviour).
* Tests self-skip when no LLM provider key is configured, keeping a bare
  checkout green.
"""

from __future__ import annotations

import os

os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("RANDOM_SEED", "7")

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from config.settings import get_settings
from data.generators import GenerationSummary
from exceptions import ConfigurationError


@pytest.fixture(scope="session", autouse=True)
def _require_llm_provider() -> None:
    """Skip the whole session when no usable LLM provider is configured."""
    try:
        get_settings().llm_runtime()
    except ConfigurationError as exc:
        pytest.skip(
            f"integration tests need a configured LLM provider: {exc.message}",
            allow_module_level=True,
        )


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def db_ready() -> AsyncIterator[None]:
    """Apply Alembic migrations; dispose the loop's engine afterwards."""
    from memory.db import dispose_engine, run_migrations

    await run_migrations()
    yield
    await dispose_engine()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def seeded_fleet(db_ready: None) -> GenerationSummary:
    """Seed WT history incl. the anchored WT-07 bearing-wear scenario."""
    from data.generators import GenerateRequest, generate_and_store

    summary = await generate_and_store(
        GenerateRequest(
            asset_ids=["WT-01", "WT-03", "WT-07"],
            hours=72.0,
            freq_minutes=15,
            include_wt07_scenario=True,
        )
    )
    assert summary.total_rows > 0
    return summary


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def vector_ready() -> None:
    """Seed Chroma with manuals + RCAs (best effort — diagnostics degrades
    gracefully to 'memory unavailable' if Chroma isn't running)."""
    from memory.vector_store import VectorStore

    try:
        await VectorStore(collection_name="energyforge_test").seed_default_corpus()
    except Exception:  # noqa: BLE001 - soft dependency for the scenario
        pass
