"""EnergyForge tool contract.

Every tool in ``tools/`` implements the same narrow interface so that
LLM-driven agents can call them uniformly:

* ``name`` / ``description`` — surfaced to the LLM for tool-calling
* ``input_model``  — Pydantic schema; validates and sanitises every argument
* ``output_model`` — Pydantic schema; ALWAYS contains ``confidence`` (0–1)
  and ``error`` fields
* ``run()``        — async entry point that **never raises**; failures are
  returned as ``output_model(confidence=0.0, error="...")``

Design rules
------------
1. All output-model fields MUST have defaults, so the error path can always
   construct ``output_model(confidence=0.0, error=msg)``.
2. Tools raise ``EnergyForgeError`` subclasses internally; :meth:`run` catches
   everything and renders it into ``error`` (tools never raise).
3. Execution-time guard: :meth:`run` enforces ``settings.tool_timeout_s``.

CrewAI integration
------------------
:meth:`EnergyForgeTool.as_crewai_tool` adapts a tool to
``crewai.tools.BaseTool``. CrewAI executes tool calls **synchronously in a
worker thread**, while our tools are async and our asyncpg engine is bound to
the orchestrator's event loop. The adapter therefore:

* reads the owner loop from the ``_tool_owner_loop`` contextvar (populated by
  :func:`bind_tool_owner_loop` before ``asyncio.to_thread`` — `to_thread`
  propagates context), and schedules tool coroutines onto it with
  ``run_coroutine_threadsafe`` so ALL async I/O stays on one loop;
* falls back to ``asyncio.run`` when no owner loop is bound (plain scripts).

Tool invocations are recorded into a ``ToolCallRecorder`` (explicitly passed
to the adapter) so the dashboard can render an expandable agent trace.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from contextvars import ContextVar
from typing import ClassVar, Generic, TypeVar

from pydantic import BaseModel, Field

from config.settings import Settings, get_settings
from exceptions import EnergyForgeError, ToolExecutionError
from logging_config import get_logger

logger = get_logger(__name__)


class BaseToolInput(BaseModel):
    """Base class for tool inputs. Extra LLM-hallucinated keys are ignored."""


class BaseToolOutput(BaseModel):
    """Base class for tool outputs — every tool reports confidence + error."""

    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    error: str | None = None


class ToolCallRecord(BaseModel):
    """One recorded tool invocation (for the dashboard agent trace)."""

    tool: str
    ok: bool
    summary: str = ""


ToolCallRecorder = list[ToolCallRecord]

I = TypeVar("I", bound=BaseToolInput)
O = TypeVar("O", bound=BaseToolOutput)

# Owner event loop for tool coroutines — see module docstring.
_tool_owner_loop: ContextVar[asyncio.AbstractEventLoop | None] = ContextVar(
    "energyforge_tool_owner_loop", default=None
)


def bind_tool_owner_loop() -> None:
    """Capture the currently running loop as the tool-execution owner.

    Call right before ``asyncio.to_thread(crew.kickoff)`` — contextpropagation
    makes the value visible inside the worker thread's CrewAI adapters.
    """
    _tool_owner_loop.set(asyncio.get_running_loop())


class EnergyForgeTool(ABC, Generic[I, O]):
    """Abstract base for all EnergyForge tools."""

    name: ClassVar[str] = "base_tool"
    description: ClassVar[str] = "abstract tool"
    input_model: ClassVar[type[BaseToolInput]] = BaseToolInput
    output_model: ClassVar[type[BaseToolOutput]] = BaseToolOutput

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    @abstractmethod
    async def _arun(self, tool_input: I) -> O:
        """Tool implementation. May raise EnergyForgeError; run() contains it."""
        ...

    async def run(self, tool_input: I, *, recorder: ToolCallRecorder | None = None) -> O:
        """Execute with timeout + total error containment. Never raises.

        Args:
            tool_input: validated input model instance.
            recorder: optional trace sink (one :class:`ToolCallRecord` appended).
        """
        log = get_logger(f"tools.{self.name}")
        try:
            output = await asyncio.wait_for(
                self._arun(tool_input), timeout=self._settings.tool_timeout_s
            )
            if output.error:
                log.warning("tool.error", error=output.error)
            else:
                log.info("tool.ok", confidence=output.confidence)
            self._record(recorder, tool_input, ok=output.error is None,
                         detail=output.error or f"confidence={output.confidence:.2f}")
            return output
        except TimeoutError:
            return self._fail(recorder, tool_input,
                              f"timed out after {self._settings.tool_timeout_s:.0f}s")
        except EnergyForgeError as exc:
            return self._fail(recorder, tool_input, f"{exc.code}: {exc.message}")
        except Exception as exc:  # broad: the never-raise contract demands containment
            return self._fail(recorder, tool_input,
                              f"unexpected {type(exc).__name__}: {str(exc)[:300]}")

    def _fail(
        self,
        recorder: ToolCallRecorder | None,
        tool_input: I,
        message: str,
    ) -> O:
        get_logger(f"tools.{self.name}").warning("tool.failed", error=message)
        self._record(recorder, tool_input, ok=False, detail=message[:200])
        try:
            return self.output_model(confidence=0.0, error=message)  # type: ignore[return-value]
        except Exception as exc:  # pragma: no cover - construction-rule violation
            raise ToolExecutionError(
                f"output model of {self.name} cannot render an error result: {exc}",
                tool_name=self.name,
            ) from exc

    def _record(
        self,
        recorder: ToolCallRecorder | None,
        tool_input: BaseToolInput,
        *,
        ok: bool,
        detail: str,
    ) -> None:
        if recorder is not None:
            recorder.append(
                ToolCallRecord(tool=self.name, ok=ok, summary=detail[:200])
            )

    def record_input_summary(self, tool_input: I) -> str:
        """Short, log-safe summary of an input (no secrets — inputs never carry any)."""
        return ",".join(f"{k}={v}" for k, v in tool_input.model_dump().items())[:200]

    # ── CrewAI adapter ─────────────────────────────────────────────────────

    def as_crewai_tool(self, recorder: ToolCallRecorder | None = None) -> object:
        """Return a ``crewai.tools.BaseTool`` wrapping this tool.

        The sync ``_run`` executed in CrewAI's worker thread forwards the
        coroutine to the owner loop recorded via :func:`bind_tool_owner_loop`,
        or to a fresh ``asyncio.run`` when unbound.
        """
        from crewai.tools import BaseTool as CrewBaseTool

        tool = self

        class _Adapter(CrewBaseTool):  # type: ignore[misc]  # crewai is untyped
            name: str = tool.name
            description: str = tool.description
            args_schema: type[BaseModel] = tool.input_model

            def _run(self, **kwargs: object) -> str:
                parsed = tool.input_model.model_validate(kwargs)
                coro = tool.run(parsed, recorder=recorder)
                loop = _tool_owner_loop.get()
                if loop is not None and loop.is_running():
                    return (
                        asyncio.run_coroutine_threadsafe(coro, loop)
                        .result(timeout=180)
                        .model_dump_json()
                    )
                return asyncio.run(coro).model_dump_json()

        return _Adapter()
