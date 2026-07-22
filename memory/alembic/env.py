"""Alembic environment — async engine, TimescaleDB target.

The database URL is resolved from ``config.settings`` (the same
pydantic-settings object the application uses), so there is exactly one
place where credentials are composed.

* online  → asyncpg engine, ``connection.run_sync(...)``
* offline → sync URL (psycopg) rendered with literal binds
"""

from __future__ import annotations

import asyncio
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

# Make the repository root importable when alembic runs from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.settings import get_settings  # noqa: E402
from memory.db import Base  # noqa: E402

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL without connecting (uses the psycopg sync URL)."""
    context.configure(
        url=get_settings().sync_database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Connect with asyncpg and migrate inside a transaction."""
    engine = create_async_engine(get_settings().database_url, poolclass=pool.NullPool)
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
