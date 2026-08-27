"""Distributed tracing via OpenTelemetry with local-only exporters.

Provides OTel spans linking an incoming user query through coordinator ->
specialist agent handoffs, model requests, and tool invocations, using the
span-naming convention Google's Agent Development Kit itself uses so traces
read the same way whether produced locally or by a deployed ADK app:

    invoke_agent (one per agent in the chain)
      +-- call_llm (a model request)
      +-- execute_tool (a tool invocation)

Three exporters are supported locally, chosen via `Settings.otel_exporter`:
`console` (human-readable spans printed to stdout), `memory` (in-process
`InMemorySpanExporter`, used by tests to assert on finished spans without a
collector), and `otlp` (a local OTLP collector, e.g. `localhost:4317`, when
one is running). No cloud project or remote endpoint is required for any of
the three. Every span attribute is scrubbed for PII before being attached,
using the same `scrub_value` helper the structured logger uses.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Iterator

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Span, Tracer

from src.config import Settings, get_settings
from src.observability.pii_scrubber import scrub_value

_SERVICE_NAME = "smart-itinerary-agent"

_provider: TracerProvider | None = None
_exporter: SpanExporter | None = None


def _build_exporter(otel_exporter: str) -> SpanExporter:
    """Instantiate the local `SpanExporter` named by `otel_exporter`.

    Args:
        otel_exporter: One of `"console"`, `"memory"`, or `"otlp"`
            (case-insensitive). Unrecognized values fall back to `"console"`.

    Returns:
        A `ConsoleSpanExporter`, `InMemorySpanExporter`, or, for `"otlp"`, an
        `OTLPSpanExporter` targeting a local collector (endpoint resolved
        from the standard `OTEL_EXPORTER_OTLP_ENDPOINT` env var, defaulting
        to `localhost:4317`).
    """
    name = otel_exporter.strip().lower()
    if name == "memory":
        return InMemorySpanExporter()
    if name == "otlp":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )

        return OTLPSpanExporter()
    return ConsoleSpanExporter()


def build_tracer_provider(
    settings: Settings | None = None,
    exporter: SpanExporter | None = None,
) -> tuple[TracerProvider, SpanExporter]:
    """Build a standalone `TracerProvider` wired to a local exporter.

    Kept separate from `configure_tracing` so tests can build an isolated
    provider (typically backed by an explicit `InMemorySpanExporter`)
    without mutating process-global tracing state.

    Args:
        settings: Application settings; `settings.otel_exporter` selects the
            exporter when `exporter` is not given. Defaults to
            `get_settings()`.
        exporter: Explicit `SpanExporter` to use instead of building one
            from `settings.otel_exporter`.

    Returns:
        A `(TracerProvider, SpanExporter)` tuple. The provider has a single
        `SimpleSpanProcessor` wrapping the exporter, so spans are exported
        synchronously as soon as they end (appropriate for local dev/test,
        where losing buffered spans on process exit would be surprising).
    """
    settings = settings or get_settings()
    resource = Resource.create({SERVICE_NAME: _SERVICE_NAME})
    provider = TracerProvider(resource=resource)
    resolved_exporter = exporter or _build_exporter(settings.otel_exporter)
    provider.add_span_processor(SimpleSpanProcessor(resolved_exporter))
    return provider, resolved_exporter


def configure_tracing(
    settings: Settings | None = None,
    exporter: SpanExporter | None = None,
) -> SpanExporter:
    """(Re)configure the process-wide tracer provider used by `get_tracer`.

    Also attempts to register the provider as the global OpenTelemetry
    tracer provider via `trace.set_tracer_provider`, so any third-party
    auto-instrumentation in the process picks it up too. That global
    registration can only succeed once per process (a documented OTel API
    limitation); `get_tracer` is unaffected by that restriction because it
    reads the provider from this module's own reference rather than the
    global one, so tests may call `configure_tracing` repeatedly (e.g. with
    a fresh `InMemorySpanExporter` per test) and still get isolated traces.

    Args:
        settings: Application settings; defaults to `get_settings()`.
        exporter: Explicit `SpanExporter` to use instead of building one
            from `settings.otel_exporter`.

    Returns:
        The `SpanExporter` now backing the tracer provider (useful for
        passing an `InMemorySpanExporter` here and later reading
        `.get_finished_spans()` off the returned instance).
    """
    global _provider, _exporter
    _provider, _exporter = build_tracer_provider(settings, exporter)
    trace.set_tracer_provider(_provider)
    return _exporter


def get_tracer(name: str = "travel_agent") -> Tracer:
    """Return a `Tracer` bound to `name`, configuring defaults on first use.

    Args:
        name: Tracer/instrumentation name, conventionally the emitting
            module or agent (e.g. `"travel_coordinator"`).

    Returns:
        An OpenTelemetry `Tracer` sourced from this module's tracer
        provider (configuring one via `configure_tracing()` if none exists
        yet).
    """
    global _provider
    if _provider is None:
        configure_tracing()
    assert _provider is not None
    return _provider.get_tracer(name)


def get_finished_spans() -> list[ReadableSpan]:
    """Return spans captured so far, when the active exporter is in-memory.

    Returns:
        The list of `ReadableSpan` objects from `InMemorySpanExporter.get_finished_spans()`.

    Raises:
        RuntimeError: If tracing has not been configured, or the active
            exporter is not an `InMemorySpanExporter` (i.e. `otel_exporter`
            is `"console"` or `"otlp"`, which don't retain spans in-process).
    """
    if _exporter is None or not isinstance(_exporter, InMemorySpanExporter):
        raise RuntimeError(
            "get_finished_spans() requires tracing to be configured with the "
            "'memory' exporter; call configure_tracing(settings) with "
            "otel_exporter='memory' first."
        )
    return list(_exporter.get_finished_spans())


def _coerce_attribute_value(value: Any) -> str | bool | int | float:
    """Coerce a scrubbed value into a type OTel span attributes accept."""
    if isinstance(value, (str, bool, int, float)):
        return value
    if value is None:
        return ""
    return json.dumps(value, default=str)


def _build_span_attributes(raw_attributes: dict[str, Any]) -> dict[str, str | bool | int | float]:
    """Scrub PII from `raw_attributes` and coerce values to OTel-safe types."""
    scrubbed = scrub_value(raw_attributes)
    return {key: _coerce_attribute_value(value) for key, value in scrubbed.items()}


@contextmanager
def _traced_span(tracer: Tracer, span_name: str, attributes: dict[str, Any]) -> Iterator[Span]:
    with tracer.start_as_current_span(span_name) as span:
        span.set_attributes(_build_span_attributes(attributes))
        yield span


@contextmanager
def start_agent_invocation_span(
    tracer: Tracer,
    *,
    session_id: str,
    turn_id: str,
    agent_name: str,
    model_name: str,
    **extra: Any,
) -> Iterator[Span]:
    """Open an `invoke_agent` span for one agent's handling of a turn.

    Args:
        tracer: A `Tracer` from `get_tracer`.
        session_id: Identifier of the persistent conversation session.
        turn_id: Identifier of the current conversation turn.
        agent_name: Name of the agent being invoked, e.g.
            `"TravelCoordinatorAgent"`.
        model_name: Model id backing this agent, e.g. `"gemini-3.1-pro"`.
        **extra: Additional span attributes; PII-scrubbed before attaching.

    Yields:
        The active `Span`, ended automatically on context exit. Exceptions
        raised inside the block are recorded on the span and it is marked
        `ERROR` before the exception propagates.
    """
    with _traced_span(
        tracer,
        "invoke_agent",
        {
            "session_id": session_id,
            "turn_id": turn_id,
            "agent_name": agent_name,
            "model_name": model_name,
            **extra,
        },
    ) as span:
        yield span


@contextmanager
def start_llm_call_span(
    tracer: Tracer,
    *,
    session_id: str,
    turn_id: str,
    agent_name: str,
    model_name: str,
    **extra: Any,
) -> Iterator[Span]:
    """Open a `call_llm` span for a single model request.

    Args:
        tracer: A `Tracer` from `get_tracer`.
        session_id: Identifier of the persistent conversation session.
        turn_id: Identifier of the current conversation turn.
        agent_name: Name of the agent issuing the request.
        model_name: Model id being called, e.g. `"gemini-3.5-flash"`.
        **extra: Additional span attributes (e.g. `token_counts`);
            PII-scrubbed before attaching.

    Yields:
        The active `Span`, ended automatically on context exit.
    """
    with _traced_span(
        tracer,
        "call_llm",
        {
            "session_id": session_id,
            "turn_id": turn_id,
            "agent_name": agent_name,
            "model_name": model_name,
            **extra,
        },
    ) as span:
        yield span


@contextmanager
def start_tool_execution_span(
    tracer: Tracer,
    *,
    session_id: str,
    turn_id: str,
    agent_name: str,
    tool_name: str,
    **extra: Any,
) -> Iterator[Span]:
    """Open an `execute_tool` span for a single tool invocation.

    Args:
        tracer: A `Tracer` from `get_tracer`.
        session_id: Identifier of the persistent conversation session.
        turn_id: Identifier of the current conversation turn.
        agent_name: Name of the agent invoking the tool.
        tool_name: Name of the tool function being called, e.g.
            `"fetch_destination_weather_forecast"`.
        **extra: Additional span attributes (e.g. tool arguments, the
            resulting `status`); PII-scrubbed before attaching.

    Yields:
        The active `Span`, ended automatically on context exit.
    """
    with _traced_span(
        tracer,
        "execute_tool",
        {
            "session_id": session_id,
            "turn_id": turn_id,
            "agent_name": agent_name,
            "tool_name": tool_name,
            **extra,
        },
    ) as span:
        yield span
