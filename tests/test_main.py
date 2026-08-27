"""Tests for the local interactive CLI entrypoint (Phase 6, Task 6.3).

Exercises `TravelAgentSession` against a fake ADK `Runner`/`session_service`
double instead of a live Gemini API call, so these tests run fully offline
while still verifying the real integration points: persistence via
`SessionStore`, structured logging/tracing, and the HITL booking
authorization round trip.
"""

from __future__ import annotations

import pytest
from google.adk.events import Event
from google.adk.sessions import InMemorySessionService
from google.genai import types

from src.guardrails.hitl_manager import AUTHORIZED_BOOKING_IDS_STATE_KEY
from src.main import APP_NAME, DEFAULT_USER_ID, TravelAgentSession
from src.memory.session_store import SessionStore


def _text_event(author: str, text: str) -> Event:
    return Event(author=author, content=types.Content(role="model", parts=[types.Part(text=text)]))


def _confirm_call_event(author: str, provisional_booking_id: str) -> Event:
    return Event(
        author=author,
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="confirm_reservation_booking",
                        args={
                            "provisional_booking_id": provisional_booking_id,
                            "user_confirmation_token": "TOKEN123",
                        },
                        id="call-1",
                    )
                )
            ],
        ),
    )


def _blocked_confirm_response_event(author: str) -> Event:
    return Event(
        author=author,
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        name="confirm_reservation_booking",
                        response={
                            "status": "error",
                            "error_code": "AUTHORIZATION_REQUIRED",
                            "message": "not authorized yet",
                        },
                        id="call-1",
                    )
                )
            ],
        ),
    )


class FakeRunner:
    """Stand-in for `google.adk.runners.Runner`: replays scripted event lists."""

    def __init__(self, scripted_event_lists: list[list[Event]]) -> None:
        self._scripted_event_lists = list(scripted_event_lists)
        self.calls: list[tuple[str, str, str]] = []

    async def run_async(self, *, user_id: str, session_id: str, new_message):
        text = "".join(part.text or "" for part in (new_message.parts or []))
        self.calls.append((user_id, session_id, text))
        events = self._scripted_event_lists.pop(0)
        for event in events:
            yield event


@pytest.fixture
async def store():
    async with SessionStore(database_path=":memory:") as store:
        yield store


async def _make_session(store, scripted_event_lists, confirm_booking=None) -> tuple[TravelAgentSession, FakeRunner]:
    session_service = InMemorySessionService()
    runner = FakeRunner(scripted_event_lists)
    session = TravelAgentSession(
        "sess-1",
        store=store,
        session_service=session_service,
        runner=runner,
        confirm_booking=confirm_booking,
    )
    return session, runner


# --- basic turn handling -------------------------------------------------


async def test_send_message_returns_final_agent_text(store):
    session, runner = await _make_session(
        store, [[_text_event("TravelCoordinatorAgent", "Here is your 3-day Kyoto plan.")]]
    )

    reply = await session.send_message("Plan a 3-day trip to Kyoto")

    assert reply == "Here is your 3-day Kyoto plan."
    assert runner.calls == [(DEFAULT_USER_ID, "sess-1", "Plan a 3-day trip to Kyoto")]
    await session.aclose()


async def test_send_message_persists_user_and_agent_turns(store):
    session, _ = await _make_session(store, [[_text_event("TravelCoordinatorAgent", "Sure thing.")]])

    await session.send_message("Hi there")

    turns = await store.get_turns("sess-1")
    assert [t.role for t in turns] == ["user", "agent"]
    assert turns[0].content == "Hi there"
    assert turns[1].content == "Sure thing."
    assert turns[1].agent_name == "TravelCoordinatorAgent"
    await session.aclose()


async def test_send_message_schedules_background_preference_extraction(store):
    session, _ = await _make_session(
        store, [[_text_event("TravelCoordinatorAgent", "Noted, vegetarian it is.")]]
    )

    await session.send_message("I'm vegetarian, please plan around that")
    await session.aclose()

    preferences = await store.get_traveler_preferences("sess-1")
    assert preferences is not None
    assert "vegetarian" in preferences.dietary_restrictions


async def test_send_message_sanitizes_input_before_persisting_and_running(store):
    session, runner = await _make_session(
        store, [[_text_event("TravelCoordinatorAgent", "Got it.")]]
    )

    await session.send_message("Book the hotel\x00\x07 for two nights")

    turns = await store.get_turns("sess-1")
    assert turns[0].content == "Book the hotel for two nights"
    assert runner.calls[0][2] == "Book the hotel for two nights"
    await session.aclose()


async def test_reused_session_reuses_the_same_adk_session(store):
    session, runner = await _make_session(
        store,
        [
            [_text_event("TravelCoordinatorAgent", "First reply.")],
            [_text_event("TravelCoordinatorAgent", "Second reply.")],
        ],
    )

    await session.send_message("First message")
    await session.send_message("Second message")

    assert len(runner.calls) == 2
    turns = await store.get_turns("sess-1")
    assert len(turns) == 4
    await session.aclose()


# --- HITL booking authorization round trip -------------------------------


async def test_blocked_booking_confirmation_prompts_and_retries_on_approval(store):
    async def approve(booking_args):
        assert booking_args["provisional_booking_id"] == "book_abc123"
        return True

    session, runner = await _make_session(
        store,
        [
            [
                _confirm_call_event("BookingAgent", "book_abc123"),
                _blocked_confirm_response_event("BookingAgent"),
            ],
            [_text_event("BookingAgent", "Your booking is confirmed!")],
        ],
        confirm_booking=approve,
    )

    reply = await session.send_message("Please confirm the hotel booking")

    assert reply == "Your booking is confirmed!"
    assert len(runner.calls) == 2
    assert "book_abc123" in runner.calls[1][2]

    adk_session = await session._session_service.get_session(
        app_name=APP_NAME, user_id=DEFAULT_USER_ID, session_id="sess-1"
    )
    assert adk_session.state[AUTHORIZED_BOOKING_IDS_STATE_KEY] == ["book_abc123"]
    await session.aclose()


async def test_blocked_booking_confirmation_stops_on_traveler_refusal(store):
    async def refuse(booking_args):
        return False

    session, runner = await _make_session(
        store,
        [
            [
                _confirm_call_event("BookingAgent", "book_abc123"),
                _blocked_confirm_response_event("BookingAgent"),
            ]
        ],
        confirm_booking=refuse,
    )

    reply = await session.send_message("Please confirm the hotel booking")

    assert "book_abc123" in reply
    assert len(runner.calls) == 1

    adk_session = await session._session_service.get_session(
        app_name=APP_NAME, user_id=DEFAULT_USER_ID, session_id="sess-1"
    )
    assert adk_session.state.get(AUTHORIZED_BOOKING_IDS_STATE_KEY, []) == []
    await session.aclose()
