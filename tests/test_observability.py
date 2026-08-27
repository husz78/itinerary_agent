"""Tests for src/observability: PII scrubbing, structured logging, and tracing."""

import io
import json

import pytest

from src.config import Settings
from src.observability.pii_scrubber import scrub_text, scrub_value
from src.observability.logger import (
    configure_logging,
    get_logger,
    log_agent_intent,
    log_agent_outcome,
)
from src.observability.tracer import (
    configure_tracing,
    get_finished_spans,
    get_tracer,
    start_agent_invocation_span,
    start_llm_call_span,
    start_tool_execution_span,
)


# --- pii_scrubber ------------------------------------------------------------


def test_scrub_text_masks_email_addresses():
    assert scrub_text("Contact jane@example.com for details.") == (
        "Contact [REDACTED_EMAIL] for details."
    )


def test_scrub_text_masks_valid_luhn_credit_card():
    assert scrub_text("Card: 4111 1111 1111 1111 exp 12/29") == (
        "Card: [REDACTED_CREDIT_CARD] exp 12/29"
    )


def test_scrub_text_masks_phone_numbers_in_common_formats():
    assert scrub_text("Call +1 415-555-0199 or (415) 555-0199 now.") == (
        "Call [REDACTED_PHONE_NUMBER] or [REDACTED_PHONE_NUMBER] now."
    )


def test_scrub_text_masks_passport_numbers():
    assert scrub_text("Passport AB1234567 needed for boarding.") == (
        "Passport [REDACTED_PASSPORT_NUMBER] needed for boarding."
    )


def test_scrub_text_preserves_iso_dates_and_timestamps():
    text = "Trip from 2026-09-01 to 2026-09-05, logged at 2026-08-27T09:54:03.984071Z"
    assert scrub_text(text) == text


def test_scrub_text_leaves_clean_text_unchanged():
    text = "No PII here, just a normal sentence about Paris in September."
    assert scrub_text(text) == text


def test_scrub_text_ignores_long_digit_run_failing_luhn_check():
    text = "Order id 20260901123456789 is not a credit card."
    assert scrub_text(text) == text


def test_scrub_text_handles_empty_string():
    assert scrub_text("") == ""


def test_scrub_value_redacts_string_leaves_in_nested_structures():
    scrubbed = scrub_value(
        {
            "email": "jane@example.com",
            "nested": {"phone": "415-555-0100", "note": "ok"},
            "items": ["contact bob@example.com", 42, None],
        }
    )
    assert scrubbed == {
        "email": "[REDACTED_EMAIL]",
        "nested": {"phone": "[REDACTED_PHONE_NUMBER]", "note": "ok"},
        "items": ["contact [REDACTED_EMAIL]", 42, None],
    }


@pytest.mark.parametrize(
    "key", ["api_key", "API_KEY", "password", "credit_card_number", "passport_number", "cvv"]
)
def test_scrub_value_blanket_redacts_sensitive_key_names(key):
    scrubbed = scrub_value({key: "totally-innocuous-value"})
    assert scrubbed[key] == "[REDACTED_FIELD]"


def test_scrub_value_passes_through_non_string_primitives():
    assert scrub_value(42) == 42
    assert scrub_value(3.14) == 3.14
    assert scrub_value(True) is True
    assert scrub_value(None) is None


def test_scrub_value_converts_basemodel_to_scrubbed_dict():
    from pydantic import BaseModel

    class Contact(BaseModel):
        email: str
        note: str

    scrubbed = scrub_value(Contact(email="jane@example.com", note="vip"))
    assert scrubbed == {"email": "[REDACTED_EMAIL]", "note": "vip"}


# --- logger --------------------------------------------------------------


def _read_json_lines(buf: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in buf.getvalue().splitlines() if line.strip()]


def test_configure_logging_emits_single_line_json_records():
    buf = io.StringIO()
    configure_logging(Settings(log_file_path=None), output_stream=buf)
    logger = get_logger("test_logger")

    logger.info("something_happened", detail="fine")

    records = _read_json_lines(buf)
    assert len(records) == 1
    assert records[0]["event"] == "something_happened"
    assert records[0]["detail"] == "fine"
    assert records[0]["level"] == "info"
    assert "timestamp" in records[0]


def test_configure_logging_scrubs_pii_before_rendering():
    buf = io.StringIO()
    configure_logging(Settings(log_file_path=None), output_stream=buf)
    logger = get_logger("test_logger")

    logger.info("user_message_received", email="jane@example.com", phone="415-555-0100")

    record = _read_json_lines(buf)[0]
    assert record["email"] == "[REDACTED_EMAIL]"
    assert record["phone"] == "[REDACTED_PHONE_NUMBER]"


def test_log_agent_intent_and_outcome_emit_paired_events_with_shared_ids():
    buf = io.StringIO()
    configure_logging(Settings(log_file_path=None), output_stream=buf)
    logger = get_logger("test_logger")

    log_agent_intent(
        logger,
        session_id="sess-1",
        turn_id="turn-1",
        agent_name="WeatherSpecialistAgent",
        model_name="gemini-3.5-flash",
        action="call_tool:fetch_destination_weather_forecast",
        reasoning="need forecast for outdoor activity check",
    )
    log_agent_outcome(
        logger,
        session_id="sess-1",
        turn_id="turn-1",
        agent_name="WeatherSpecialistAgent",
        model_name="gemini-3.5-flash",
        status="success",
        latency_ms=87.5,
        result_summary="3-day forecast returned",
        token_counts={"prompt": 120, "completion": 40},
    )

    intent, outcome = _read_json_lines(buf)
    assert intent["event"] == "AGENT_INTENT"
    assert outcome["event"] == "AGENT_OUTCOME"
    assert intent["session_id"] == outcome["session_id"] == "sess-1"
    assert intent["turn_id"] == outcome["turn_id"] == "turn-1"
    assert outcome["status"] == "success"
    assert outcome["token_counts"] == {"prompt": 120, "completion": 40}


def test_log_agent_intent_scrubs_pii_in_extra_fields():
    buf = io.StringIO()
    configure_logging(Settings(log_file_path=None), output_stream=buf)
    logger = get_logger("test_logger")

    log_agent_intent(
        logger,
        session_id="sess-1",
        turn_id="turn-1",
        agent_name="BookingAgent",
        model_name="gemini-3.1-pro",
        action="call_tool:stage_provisional_booking",
        traveler_email="jane@example.com",
    )

    record = _read_json_lines(buf)[0]
    assert record["traveler_email"] == "[REDACTED_EMAIL]"


# --- tracer ----------------------------------------------------------------


@pytest.fixture
def memory_tracer():
    configure_tracing(Settings(otel_exporter="memory"))
    return get_tracer("test_tracer")


def test_start_agent_invocation_span_records_expected_attributes(memory_tracer):
    with start_agent_invocation_span(
        memory_tracer,
        session_id="sess-1",
        turn_id="turn-1",
        agent_name="TravelCoordinatorAgent",
        model_name="gemini-3.1-pro",
    ):
        pass

    spans = get_finished_spans()
    span = spans[-1]
    assert span.name == "invoke_agent"
    assert span.attributes["session_id"] == "sess-1"
    assert span.attributes["agent_name"] == "TravelCoordinatorAgent"
    assert span.attributes["model_name"] == "gemini-3.1-pro"


def test_nested_llm_and_tool_spans_share_trace_with_parent_agent_span(memory_tracer):
    with start_agent_invocation_span(
        memory_tracer,
        session_id="sess-2",
        turn_id="turn-1",
        agent_name="TravelCoordinatorAgent",
        model_name="gemini-3.1-pro",
    ) as parent_span:
        with start_llm_call_span(
            memory_tracer,
            session_id="sess-2",
            turn_id="turn-1",
            agent_name="TravelCoordinatorAgent",
            model_name="gemini-3.1-pro",
        ) as llm_span:
            pass
        with start_tool_execution_span(
            memory_tracer,
            session_id="sess-2",
            turn_id="turn-1",
            agent_name="TravelCoordinatorAgent",
            tool_name="fetch_destination_weather_forecast",
        ) as tool_span:
            pass

    trace_id = parent_span.get_span_context().trace_id
    assert llm_span.get_span_context().trace_id == trace_id
    assert tool_span.get_span_context().trace_id == trace_id

    spans = get_finished_spans()
    names = [s.name for s in spans[-3:]]
    assert names == ["call_llm", "execute_tool", "invoke_agent"]


def test_tool_span_scrubs_pii_from_attributes(memory_tracer):
    with start_tool_execution_span(
        memory_tracer,
        session_id="sess-3",
        turn_id="turn-1",
        agent_name="BookingAgent",
        tool_name="stage_provisional_booking",
        traveler_email="jane@example.com",
    ):
        pass

    span = get_finished_spans()[-1]
    assert span.attributes["traveler_email"] == "[REDACTED_EMAIL]"


def test_tool_span_records_exception_and_marks_error_status(memory_tracer):
    from opentelemetry.trace import StatusCode

    with pytest.raises(ValueError):
        with start_tool_execution_span(
            memory_tracer,
            session_id="sess-4",
            turn_id="turn-1",
            agent_name="BookingAgent",
            tool_name="confirm_reservation_booking",
        ):
            raise ValueError("upstream booking API timed out")

    span = get_finished_spans()[-1]
    assert span.status.status_code == StatusCode.ERROR
    assert any(event.name == "exception" for event in span.events)


def test_get_finished_spans_raises_when_exporter_is_not_memory():
    configure_tracing(Settings(otel_exporter="console"))
    with pytest.raises(RuntimeError):
        get_finished_spans()
