"""EnergyForge Agent exception hierarchy.

A single base class keeps ``except EnergyForgeError`` a valid catch-all
anywhere in the system. Every error carries a machine-readable ``code`` and a
structured ``context`` mapping so logs, tool outputs and audit records stay
uniform.

Design rule (cross-cutting): **tools never raise**. Tool wrappers catch
``EnergyForgeError`` (and any unexpected exception) and translate it into the
``error`` field of their Pydantic output schema. These classes define how
errors travel *internally* with structure intact before being rendered.
"""

from __future__ import annotations


class EnergyForgeError(Exception):
    """Base class for all EnergyForge errors."""

    default_code: str = "ENERGYFORGE_ERROR"
    retryable: bool = False  # transient subclasses override to True

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.default_code
        self.context: dict[str, object] = dict(context or {})

    def to_dict(self) -> dict[str, object]:
        """Serialise to the JSON shape used in tool `error` fields and audit logs."""
        return {"code": self.code, "message": self.message, "context": self.context}

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


# ── Configuration ─────────────────────────────────────────────────────────────


class ConfigurationError(EnergyForgeError):
    """Invalid or missing configuration (env vars, .env). Fails fast at startup."""

    default_code = "CONFIGURATION_ERROR"


# ── Data layer ────────────────────────────────────────────────────────────────


class DataLayerError(EnergyForgeError):
    """Base for persistence-layer failures (SQL, vector store, cache)."""

    default_code = "DATA_LAYER_ERROR"


class DatabaseError(DataLayerError):
    """SQLAlchemy/TimescaleDB operation failed."""

    default_code = "DATABASE_ERROR"


class VectorStoreError(DataLayerError):
    """ChromaDB operation failed."""

    default_code = "VECTOR_STORE_ERROR"


# ── Tools & external services ─────────────────────────────────────────────────


class ToolExecutionError(EnergyForgeError):
    """A tool failed while executing. ``tool_name`` identifies the failing tool."""

    default_code = "TOOL_EXECUTION_ERROR"

    def __init__(
        self,
        message: str,
        *,
        tool_name: str,
        code: str | None = None,
        context: dict[str, object] | None = None,
    ) -> None:
        ctx = dict(context or {})
        ctx.setdefault("tool_name", tool_name)
        super().__init__(message, code=code, context=ctx)
        self.tool_name = tool_name


class ExternalServiceError(EnergyForgeError):
    """A third-party service (weather API, Slack, LLM provider) failed.

    Usually transient — orchestrator may retry with backoff.
    """

    default_code = "EXTERNAL_SERVICE_ERROR"
    retryable: bool = True


class LLMProviderError(ExternalServiceError):
    """The configured LLM provider returned an error or timed out."""

    default_code = "LLM_PROVIDER_ERROR"


class WeatherAPIError(ExternalServiceError):
    """The Open-Meteo weather API call failed."""

    default_code = "WEATHER_API_ERROR"


# ── Agents & orchestration ────────────────────────────────────────────────────


class AgentExecutionError(EnergyForgeError):
    """A CrewAI agent failed to produce a valid structured output."""

    default_code = "AGENT_EXECUTION_ERROR"


class OrchestrationError(EnergyForgeError):
    """The LangGraph orchestrator failed (planning, routing, aggregation)."""

    default_code = "ORCHESTRATION_ERROR"


class HumanApprovalRequired(OrchestrationError):
    """Raised internally by the HITL gate to pause the graph pending approval."""

    default_code = "HUMAN_APPROVAL_REQUIRED"
