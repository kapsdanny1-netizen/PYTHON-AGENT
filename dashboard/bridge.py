"""Sync↔async bridge for Streamlit.

Streamlit runs synchronous script code in ScriptRunner threads; the
EnergyForge runtime is async-first, and asyncpg engines are event-loop-bound.
The bridge therefore owns ONE dedicated background thread with a single
event loop for the entire Streamlit process lifetime — all `run()` calls
share it, so the DB engine pool is created once and reused. Cached with
``st.cache_resource`` so reruns and sessions reuse the same loop.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import TypeVar

import streamlit as st

T = TypeVar("T")


class AsyncBridge:
    """Owns a background event loop; exposes blocking ``run`` + ``stream``."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self._serve, name="energyforge-bridge", daemon=True
        )
        self.thread.start()

    def _serve(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coro: Coroutine[object, object, T], *, timeout: float = 300.0) -> T:
        """Run a coroutine on the bridge loop and block for its result."""
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result(timeout)

    def stream(
        self, agen_factory: Callable[[], AsyncIterator[dict[str, object]]]
    ) -> "queue.Queue[dict[str, object] | BaseException | None]":
        """Pump an async generator into a thread-safe queue.

        Sentinels: ``None`` = stream finished; ``BaseException`` = failure.
        """
        out: queue.Queue[dict[str, object] | BaseException | None] = queue.Queue()

        async def pump() -> None:
            try:
                async for chunk in agen_factory():
                    out.put(chunk)
            except BaseException as exc:  # surface to the Streamlit thread
                out.put(exc)
            finally:
                out.put(None)

        asyncio.run_coroutine_threadsafe(pump(), self.loop)
        return out


@st.cache_resource
def get_bridge() -> AsyncBridge:
    """Process-wide bridge singleton (survives Streamlit reruns/sessions)."""
    return AsyncBridge()
