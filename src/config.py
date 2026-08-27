"""Local settings and secret management.

Loads all configuration strictly from environment variables / a local
``.env`` file via ``pydantic-settings``. No secrets are ever hardcoded here;
``.env`` is git-ignored so credentials never enter version control.
"""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strictly typed application configuration sourced from the environment.

    Attributes:
        gemini_api_key: Secret API key for the Gemini / Google GenAI API.
            Never logged or serialized in plaintext.
        model_fast: Model id used for high-throughput, low-latency sub-tasks
            (e.g. ``gemini-3.5-flash``).
        model_pro: Model id used for complex reasoning and synthesis tasks
            (e.g. ``gemini-3.1-pro``).
        database_path: Filesystem path to the local SQLite database used for
            persistent session state.
        log_level: Standard library logging level name (e.g. ``INFO``).
        otel_exporter: Name of the local OpenTelemetry exporter to use
            (``console``, ``memory``, or ``otlp``).
        history_window_turns: Number of most recent conversation turns kept
            verbatim before older turns are summarized.
        summarization_token_threshold: Prompt token count above which
            automatic history compaction/summarization is triggered.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="API key for Gemini / Google GenAI. Loaded from GEMINI_API_KEY.",
    )
    model_fast: str = Field(
        default="gemini-3.5-flash",
        description="Model id for fast, low-latency sub-tasks.",
    )
    model_pro: str = Field(
        default="gemini-3.1-pro",
        description="Model id for complex multi-step reasoning and synthesis.",
    )
    database_path: str = Field(
        default="data/travel_agent.db",
        description="Local SQLite database path for persistent session state.",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level for structured JSON logs.",
    )
    log_file_path: str | None = Field(
        default="data/logs/agent.jsonl",
        description=(
            "Local filesystem path structured JSON logs are appended to, in "
            "addition to stdout. Set to an empty value to disable file "
            "logging and emit to stdout only."
        ),
    )
    otel_exporter: str = Field(
        default="console",
        description="Local OpenTelemetry exporter target (console, memory, or otlp).",
    )
    history_window_turns: int = Field(
        default=20,
        description="Number of recent conversation turns retained verbatim.",
    )
    summarization_token_threshold: int = Field(
        default=4000,
        description="Token threshold that triggers automatic history summarization.",
    )

    def has_gemini_api_key(self) -> bool:
        """Return True if a non-empty Gemini API key has been configured."""
        return bool(self.gemini_api_key.get_secret_value())


@lru_cache
def get_settings() -> Settings:
    """Return a cached, process-wide `Settings` instance.

    Cached so `.env` is parsed once per process; tests that need to reload
    configuration should call `get_settings.cache_clear()` first.
    """
    return Settings()
