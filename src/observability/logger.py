"""Structured JSON logging with paired `AGENT_INTENT` / `AGENT_OUTCOME` events.

Every agent decision is logged as two correlated events sharing the same
`session_id` and `turn_id`:
    - `AGENT_INTENT`: emitted the moment an agent decides what to do next
      (call a tool, hand off to a specialist, respond to the user) and why.
    - `AGENT_OUTCOME`: emitted once that decision's execution completes,
      carrying the verified result status and latency.

Log records are rendered as single-line JSON via `structlog` and written to
stdout and, if `Settings.log_file_path` is set, appended to a local file.
Every field passes through `src.observability.pii_scrubber.scrub_value`
before rendering, so PII in tool arguments, user messages, or extracted
preferences never reaches disk or stdout in the clear.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import IO, Any

import structlog

from src.config import Settings, get_settings
from src.observability.pii_scrubber import scrub_value

_LEVEL_NAME_TO_NUMBER = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


class _MultiStream:
    """Fan out `.write` calls to several writable file-like objects.

    `structlog.PrintLoggerFactory` only accepts a single output stream; this
    lets configuration tee JSON log lines to both stdout and a local log
    file without a second structlog logger instance.
    """

    def __init__(self, streams: list[IO[str]]) -> None:
        self._streams = streams

    def write(self, message: str) -> None:
        for stream in self._streams:
            stream.write(message)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _scrub_event_processor(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """structlog processor that scrubs PII from every field before rendering."""
    return scrub_value(event_dict)


def configure_logging(
    settings: Settings | None = None,
    output_stream: IO[str] | None = None,
) -> None:
    """Configure structlog to emit scrubbed, single-line JSON log events.

    Safe to call more than once (e.g. once at process startup via
    `get_logger`, and again in tests with an in-memory `output_stream` to
    capture and assert on emitted records); each call fully replaces the
    prior global structlog configuration.

    Args:
        settings: Application settings controlling `log_level` and
            `log_file_path`. Defaults to `get_settings()`.
        output_stream: If given, log lines are written only to this stream
            (typically an `io.StringIO` in tests) instead of stdout/the
            configured log file.

    Returns:
        None. Configures process-global structlog state as a side effect.
    """
    settings = settings or get_settings()

    if output_stream is not None:
        destination: IO[str] = output_stream
    else:
        streams: list[IO[str]] = [sys.stdout]
        if settings.log_file_path:
            log_path = Path(settings.log_file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            streams.append(log_path.open("a", encoding="utf-8"))
        destination = _MultiStream(streams) if len(streams) > 1 else streams[0]

    min_level = _LEVEL_NAME_TO_NUMBER.get(settings.log_level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
            _scrub_event_processor,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(min_level),
        logger_factory=structlog.PrintLoggerFactory(file=destination),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str = "travel_agent") -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound to `name`, configuring defaults on first use.

    Args:
        name: Logger name, conventionally the emitting module or agent
            (e.g. `"travel_coordinator"`, `"weather_specialist"`).

    Returns:
        A structlog `BoundLogger` ready to emit scrubbed JSON log events.
    """
    if not structlog.is_configured():
        configure_logging()
    return structlog.get_logger(name)


def log_agent_intent(
    logger: Any,
    *,
    session_id: str,
    turn_id: str,
    agent_name: str,
    model_name: str,
    action: str,
    reasoning: str = "",
    **extra: Any,
) -> None:
    """Emit an `AGENT_INTENT` event describing what an agent decided to do and why.

    Args:
        logger: A structlog logger from `get_logger`.
        session_id: Identifier of the persistent conversation session.
        turn_id: Identifier of the current conversation turn.
        agent_name: Name of the deciding agent, e.g. `"BookingAgent"`.
        model_name: Model id used to reach this decision, e.g.
            `"gemini-3.1-pro"`.
        action: Short machine-readable description of the planned action,
            e.g. `"call_tool:fetch_destination_weather_forecast"` or
            `"handoff:BookingAgent"`.
        reasoning: Optional free-text rationale for the decision.
        **extra: Additional structured context (e.g. tool arguments); passed
            through PII scrubbing like every other field.

    Returns:
        None.
    """
    logger.info(
        "AGENT_INTENT",
        session_id=session_id,
        turn_id=turn_id,
        agent_name=agent_name,
        model_name=model_name,
        action=action,
        reasoning=reasoning,
        **extra,
    )


def log_agent_outcome(
    logger: Any,
    *,
    session_id: str,
    turn_id: str,
    agent_name: str,
    model_name: str,
    status: str,
    latency_ms: float,
    result_summary: str = "",
    token_counts: dict[str, int] | None = None,
    **extra: Any,
) -> None:
    """Emit an `AGENT_OUTCOME` event carrying the verified result of a prior intent.

    Args:
        logger: A structlog logger from `get_logger`.
        session_id: Identifier shared with the originating `AGENT_INTENT`.
        turn_id: Identifier shared with the originating `AGENT_INTENT`.
        agent_name: Name of the agent whose action completed.
        model_name: Model id used, e.g. `"gemini-3.5-flash"`.
        status: Outcome status, e.g. `"success"`, `"error"`, or
            `"awaiting_authorization"`.
        latency_ms: Wall-clock duration of the action in milliseconds.
        result_summary: Optional short human-readable summary of the result.
        token_counts: Optional dict of token usage, e.g.
            `{"prompt": 512, "completion": 128}`.
        **extra: Additional structured context (e.g. a tool's error_code);
            passed through PII scrubbing like every other field.

    Returns:
        None.
    """
    logger.info(
        "AGENT_OUTCOME",
        session_id=session_id,
        turn_id=turn_id,
        agent_name=agent_name,
        model_name=model_name,
        status=status,
        latency_ms=latency_ms,
        result_summary=result_summary,
        token_counts=token_counts or {},
        **extra,
    )
