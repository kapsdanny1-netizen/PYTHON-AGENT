"""CrewAI execution harness for EnergyForge agents.

Responsibilities
----------------
* Translate :meth:`config.settings.Settings.llm_runtime` into a
  ``crewai.LLM`` for any of the four providers (single ``LLM_PROVIDER``
  switch → LiteLLM prefixes: openai/anthropic/xai/ollama).
* :func:`run_agent` — build single-agent Crew (Agent + Task + tools), kick
  it off in a worker thread and return a validated Pydantic output plus the
  recorded tool trace. Failures raise
  :class:`~exceptions.AgentExecutionError` (agents, unlike tools, may raise —
  the orchestrator contains them).
* :func:`llm_complete` — one-shot async completion used by the orchestrator's
  intent-classifier fallback (counts against the per-turn LLM budget).

Threading note: ``asyncio.to_thread`` propagates contextvars, so
:func:`tools.bind_tool_owner_loop` must be called in the async context right
before kickoff — CrewAI's synchronous tool calls inside the thread then
forward their coroutines onto this very loop (see tools/base.py).
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

from config.settings import LLMProvider, Settings, get_settings
from exceptions import AgentExecutionError, LLMProviderError
from logging_config import get_logger
from tools.base import EnergyForgeTool, ToolCallRecorder, bind_tool_owner_loop

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

logger = get_logger(__name__)

O = TypeVar("O", bound=BaseModel)

_LITELLM_PREFIX: dict[LLMProvider, str] = {
    LLMProvider.OPENAI: "openai",
    LLMProvider.ANTHROPIC: "anthropic",
    LLMProvider.GROK: "xai",
    LLMProvider.OLLAMA: "ollama",
}


class AgentOutputBase(BaseModel):
    """Contract for every agent output: confidence is REQUIRED."""

    confidence: float = Field(ge=0.0, le=1.0)


@dataclass
class AgentRunResult(Generic[O]):
    """Structured output + execution trace of one agent run."""

    output: O
    trace: ToolCallRecorder = field(default_factory=list)
    llm_model: str = ""


def build_llm(settings: Settings | None = None) -> object:
    """Construct the crewai.LLM for the configured provider."""
    from crewai import LLM

    cfg = settings or get_settings()
    runtime = cfg.llm_runtime()  # fails fast on missing credentials
    kwargs: dict[str, object] = {
        "model": f"{_LITELLM_PREFIX[runtime.provider]}/{runtime.model}",
        "temperature": runtime.temperature,
        "max_tokens": runtime.max_tokens,
    }
    if runtime.api_key:
        kwargs["api_key"] = runtime.api_key
    if runtime.base_url:
        kwargs["base_url"] = runtime.base_url
    return LLM(**kwargs)


async def llm_complete(
    prompt: str,
    *,
    system: str = "You are a precise operations classifier.",
    settings: Settings | None = None,
) -> str:
    """One-shot async completion across all four providers (no CrewAI).

    Used for orchestrator-level calls that count against the per-turn
    ``MAX_LLM_CALLS_PER_TURN`` budget.
    """
    cfg = settings or get_settings()
    runtime = cfg.llm_runtime()
    try:
        if runtime.provider in (LLMProvider.OPENAI, LLMProvider.GROK):
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=runtime.api_key or "not-needed",
                base_url=runtime.base_url,
                timeout=runtime.request_timeout_s,
            )
            response = await client.chat.completions.create(
                model=runtime.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=runtime.temperature,
                max_tokens=runtime.max_tokens,
            )
            return (response.choices[0].message.content or "").strip()

        if runtime.provider is LLMProvider.ANTHROPIC:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic(
                api_key=runtime.api_key, timeout=runtime.request_timeout_s
            )
            message = await client.messages.create(
                model=runtime.model,
                max_tokens=runtime.max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                temperature=runtime.temperature,
            )
            block = message.content[0]
            return getattr(block, "text", "").strip()

        from ollama import AsyncClient as OllamaAsyncClient

        client = OllamaAsyncClient(host=runtime.base_url)
        reply = await client.chat(
            model=runtime.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return str(reply.message.content or "").strip()
    except Exception as exc:  # broad: any provider SDK error → hierarchy
        raise LLMProviderError(
            f"{runtime.provider.value} completion failed: {type(exc).__name__}: {str(exc)[:300]}",
            context={"provider": runtime.provider.value, "model": runtime.model},
        ) from exc


async def run_agent(
    *,
    role: str,
    goal: str,
    backstory: str,
    task_description: str,
    expected_output: str,
    output_model: type[O],
    tools: list[EnergyForgeTool],
    agent_name: str,
    max_iter: int = 6,
    settings: Settings | None = None,
) -> AgentRunResult[O]:
    """Run one CrewAI agent on one task and validate its structured output.

    Raises:
        AgentExecutionError: CrewAI failure or unparseable/missing output.
        ConfigurationError: LLM provider misconfigured (before any call).
    """
    from crewai import Agent, Crew, Process, Task

    cfg = settings or get_settings()
    llm = build_llm(cfg)
    runtime = cfg.llm_runtime()

    recorder: ToolCallRecorder = []
    crew_tools = [tool.as_crewai_tool(recorder) for tool in tools]
    agent = Agent(
        role=role, goal=goal, backstory=backstory,
        tools=crew_tools, llm=llm, verbose=False,
        max_iter=max_iter, allow_delegation=False,
    )
    task = Task(
        description=task_description,
        expected_output=expected_output,
        agent=agent,
        output_pydantic=output_model,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)

    logger.info("agent.kickoff", agent=agent_name)
    bind_tool_owner_loop()
    try:
        await asyncio.to_thread(crew.kickoff)
    except Exception as exc:  # broad: crewai raises many internal error types
        raise AgentExecutionError(
            f"crew kickoff failed: {type(exc).__name__}: {str(exc)[:400]}",
            context={"agent": agent_name},
        ) from exc

    parsed = getattr(crew.tasks[0].output, "pydantic", None)
    if parsed is None:
        raise AgentExecutionError(
            "agent produced no parseable structured output",
            context={"agent": agent_name, "trace": [r.model_dump() for r in recorder]},
        )
    if not isinstance(parsed, output_model):
        parsed = output_model.model_validate(parsed.model_dump())

    logger.info(
        "agent.completed", agent=agent_name,
        confidence=parsed.confidence if isinstance(parsed, AgentOutputBase) else None,
        tool_calls=len(recorder),
    )
    return AgentRunResult(
        output=parsed, trace=recorder,
        llm_model=f"{runtime.provider.value}/{runtime.model}",
    )
