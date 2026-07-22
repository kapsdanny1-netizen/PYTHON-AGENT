"""Central runtime configuration for EnergyForge Agent.

All configuration is validated at startup with pydantic-settings and sourced
from environment variables, conventionally provided via a local `.env` file
(see `.env.example` for the fully documented reference).

LLM routing is controlled by a *single* variable, ``LLM_PROVIDER``, which
selects one of: ``openai | anthropic | grok | ollama | custom``. Provider
credentials and model defaults are resolved centrally by
:meth:`Settings.llm_runtime`, which fails fast with
:class:`~exceptions.ConfigurationError` when the selected provider requires a
key that is not configured. The ``custom`` provider targets any
OpenAI-compatible gateway you control (Arena AI-hosted endpoint, corporate
LiteLLM proxy, local vLLM/Ollama, …) and can run **keyless**.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from exceptions import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class LLMProvider(StrEnum):
    """Supported LLM backends, switchable via ``LLM_PROVIDER``."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROK = "grok"
    OLLAMA = "ollama"
    # Any OpenAI-compatible gateway you control: an Arena AI-hosted endpoint,
    # a corporate LiteLLM proxy, local vLLM, etc. — no vendor API key needed
    # beyond whatever the gateway itself expects (often none).
    CUSTOM = "custom"


class Environment(StrEnum):
    """Deployment environment; toggles dev/prod behaviours (e.g. notifications)."""

    DEV = "dev"
    TEST = "test"
    PROD = "prod"


PROVIDER_DEFAULT_MODELS: dict[LLMProvider, str] = {
    LLMProvider.OPENAI: "gpt-4o",
    LLMProvider.ANTHROPIC: "claude-sonnet-4-20250514",
    LLMProvider.GROK: "grok-3",
    LLMProvider.OLLAMA: "llama3.1:8b",
    LLMProvider.CUSTOM: "",  # no default — LLM_MODEL is mandatory for custom gateways
}

# Providers that cannot operate without an API key (Ollama runs locally).
_PROVIDERS_REQUIRING_KEY = frozenset(
    {LLMProvider.OPENAI, LLMProvider.ANTHROPIC, LLMProvider.GROK}
)


class LLMRuntimeConfig(BaseModel):
    """Resolved, provider-agnostic description of the active LLM."""

    model_config = ConfigDict(frozen=True)

    provider: LLMProvider
    model: str
    temperature: float
    max_tokens: int
    request_timeout_s: float
    api_key: str | None
    base_url: str | None


class Settings(BaseSettings):
    """Application settings; every field is overridable via env / `.env`."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Core ────────────────────────────────────────────────────────────
    app_name: str = "EnergyForge Agent"
    environment: Environment = Environment.DEV
    log_level: str = "INFO"
    random_seed: int = 42

    # ── LLM provider selection ──────────────────────────────────────────
    llm_provider: LLMProvider = LLMProvider.OPENAI
    llm_model: str | None = None  # override; else provider default is used
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=2048, ge=16)
    llm_request_timeout_s: float = Field(default=60.0, gt=0.0)

    openai_api_key: SecretStr | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_api_key: SecretStr | None = None
    grok_api_key: SecretStr | None = None
    grok_base_url: str = "https://api.x.ai/v1"
    ollama_base_url: str = "http://localhost:11434"

    # ── Custom OpenAI-compatible gateway (LLM_PROVIDER=custom) ─────────
    # Arena AI-hosted endpoint, corporate LiteLLM proxy, local vLLM, etc.
    # Base URL + model are REQUIRED for this provider; the API key is
    # OPTIONAL (leave unset for keyless gateways).
    custom_llm_base_url: str | None = None
    custom_llm_model: str = ""
    custom_llm_api_key: SecretStr | None = None

    # ── TimescaleDB / PostgreSQL ────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = Field(default=5432, ge=1, le=65535)
    postgres_user: str = "energyforge"
    postgres_password: SecretStr = SecretStr("energyforge")
    postgres_db: str = "energyforge"
    db_pool_size: int = Field(default=10, ge=1)
    db_pool_max_overflow: int = Field(default=10, ge=0)

    # ── ChromaDB vector store ───────────────────────────────────────────
    chroma_host: str = "localhost"
    chroma_port: int = Field(default=8000, ge=1, le=65535)
    chroma_ssl: bool = False
    chroma_collection: str = "energyforge_memory"

    # ── Redis ───────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Notifications ───────────────────────────────────────────────────
    slack_webhook_url: SecretStr | None = None
    notification_channel: str = "#energyforge-ops"

    # ── Governance / orchestration limits ───────────────────────────────
    hitl_confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    max_llm_calls_per_turn: int = Field(default=3, ge=1)
    tool_timeout_s: float = Field(default=30.0, gt=0.0)

    # ── Validators ──────────────────────────────────────────────────────
    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        return value.upper()

    @field_validator(
        "openai_api_key",
        "anthropic_api_key",
        "grok_api_key",
        "custom_llm_api_key",
        "custom_llm_base_url",
        "slack_webhook_url",
        mode="before",
    )
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        """Treat empty strings in .env as 'not set' rather than a real value."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    # ── Derived values ──────────────────────────────────────────────────
    @property
    def is_dev(self) -> bool:
        return self.environment is Environment.DEV

    @property
    def is_prod(self) -> bool:
        return self.environment is Environment.PROD

    @property
    def database_url(self) -> str:
        """Async SQLAlchemy URL (asyncpg driver). Never log this value."""
        password = self.postgres_password.get_secret_value()
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def sync_database_url(self) -> str:
        """Sync URL (psycopg v3) for Alembic offline mode and simple scripts."""
        password = self.postgres_password.get_secret_value()
        return (
            f"postgresql+psycopg://{self.postgres_user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def resolved_llm_model(self) -> str:
        """Explicitly configured model, falling back to the provider default."""
        if self.llm_provider is LLMProvider.CUSTOM:
            return self.llm_model or self.custom_llm_model
        return self.llm_model or PROVIDER_DEFAULT_MODELS[self.llm_provider]

    def llm_runtime(self) -> LLMRuntimeConfig:
        """Resolve model, credentials and endpoint for the selected provider.

        Returns:
            A frozen, provider-agnostic :class:`LLMRuntimeConfig`.

        Raises:
            ConfigurationError: if the selected provider requires an API key
                that is not configured.
        """
        provider = self.llm_provider
        key: SecretStr | None
        base_url: str | None
        match provider:
            case LLMProvider.OPENAI:
                key, base_url = self.openai_api_key, self.openai_base_url
            case LLMProvider.ANTHROPIC:
                key, base_url = self.anthropic_api_key, None  # SDK default endpoint
            case LLMProvider.GROK:
                key, base_url = self.grok_api_key, self.grok_base_url
            case LLMProvider.OLLAMA:
                key, base_url = None, self.ollama_base_url
            case LLMProvider.CUSTOM:
                key, base_url = self.custom_llm_api_key, self.custom_llm_base_url
                if base_url is None:
                    raise ConfigurationError(
                        "LLM_PROVIDER='custom' requires CUSTOM_LLM_BASE_URL — "
                        "point it at your OpenAI-compatible gateway (e.g. your "
                        "Arena AI-hosted endpoint: https://<host>/v1).",
                        context={"provider": "custom", "missing_env": "CUSTOM_LLM_BASE_URL"},
                    )
                if not self.resolved_llm_model:
                    raise ConfigurationError(
                        "LLM_PROVIDER='custom' requires a model name — set "
                        "LLM_MODEL or CUSTOM_LLM_MODEL.",
                        context={"provider": "custom", "missing_env": "CUSTOM_LLM_MODEL"},
                    )

        if provider in _PROVIDERS_REQUIRING_KEY and key is None:
            env_var = f"{provider.value.upper()}_API_KEY"
            raise ConfigurationError(
                f"LLM_PROVIDER={provider.value!r} requires {env_var} to be set. "
                "Either add it to .env or switch LLM_PROVIDER (e.g. to 'ollama').",
                context={"provider": provider.value, "missing_env": env_var},
            )

        return LLMRuntimeConfig(
            provider=provider,
            model=self.resolved_llm_model,
            temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens,
            request_timeout_s=self.llm_request_timeout_s,
            api_key=key.get_secret_value() if key is not None else None,
            base_url=base_url,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide singleton accessor for :class:`Settings`."""
    return Settings()
