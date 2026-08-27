"""Local interactive CLI entrypoint for the Smart Itinerary Planner & Booking Assistant.

Phase 6, Task 6.3. Wires together every piece built in Phases 1-5 into a
single chat loop against `TravelCoordinatorAgent`:

    - `src.memory.session_store.SessionStore`: persists every user/agent
      message, PII-scrubbed, to the local SQLite database so a session
      survives process restarts.
    - `src.memory.compaction.compact_session_history`: folds aged-out
      history into a summary before it grows unbounded.
    - `src.memory.async_memory.AsyncMemoryWorker`: extracts/updates traveler
      preferences in the background after each turn, never blocking the
      reply.
    - `src.observability.logger` / `src.observability.tracer`: paired
      `AGENT_INTENT`/`AGENT_OUTCOME` structured logs and an `invoke_agent`
      OTel span around every turn.
    - `src.guardrails.hitl_manager`: when the coordinator's `BookingAgent`
      attempts `confirm_reservation_booking` without prior authorization,
      the tool-level guardrail blocks it; this module surfaces that block to
      the traveler as an explicit CLI confirmation prompt, and only then
      authorizes the exact booking and asks the agent to retry.

Usage:
    uv run python -m src.main
    uv run python -m src.main --session-id trip-tokyo-2026
    uv run python -m src.main --message "Plan a 3-day trip to Kyoto"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import uuid
from typing import Any, Awaitable, Callable

from google.adk.events import Event, EventActions
from google.adk.runners import Runner
from google.adk.sessions import BaseSessionService, InMemorySessionService
from google.genai import types

from src.agents.coordinator import create_travel_coordinator_agent
from src.config import Settings, get_settings
from src.guardrails.hitl_manager import (
    AUTHORIZED_BOOKING_IDS_STATE_KEY,
    CONFIRM_BOOKING_TOOL_NAME,
    authorize_booking,
)
from src.guardrails.input_sanitizer import sanitize_user_input
from src.memory.async_memory import AsyncMemoryWorker
from src.memory.compaction import compact_session_history
from src.memory.session_store import ConversationTurn, SessionStore
from src.observability.logger import configure_logging, get_logger, log_agent_intent, log_agent_outcome
from src.observability.tracer import configure_tracing, get_tracer, start_agent_invocation_span

APP_NAME = "smart-itinerary-agent"
DEFAULT_USER_ID = "local-traveler"

# Bounds the authorize-then-retry loop for a single traveler message so a
# misbehaving model retrying confirm_reservation_booking indefinitely can't
# hang the CLI.
MAX_BOOKING_AUTHORIZATION_ROUNDS = 3

ConfirmBookingFn = Callable[[dict[str, Any]], Awaitable[bool]]


class TravelAgentSession:
    """One traveler's chat session against `TravelCoordinatorAgent`.

    Combines the ADK `Runner` (live model conversation) with this project's
    own persistent `SessionStore`, history compaction, async preference
    extraction, observability, and HITL booking authorization. A fresh ADK
    `session_service`/`runner` may be injected for testing so a chat turn
    can be exercised without a live Gemini API key.
    """

    def __init__(
        self,
        session_id: str,
        *,
        store: SessionStore,
        user_id: str = DEFAULT_USER_ID,
        settings: Settings | None = None,
        session_service: BaseSessionService | None = None,
        runner: Runner | None = None,
        confirm_booking: ConfirmBookingFn | None = None,
    ) -> None:
        """Wire up one session's agent, memory, and guardrail dependencies.

        Args:
            session_id: Persistent conversation session identifier, shared
                between `SessionStore` and the ADK session service.
            store: A connected `SessionStore` used for persistence.
            user_id: Traveler identifier passed to the ADK runner/session.
            settings: Application settings. Defaults to `get_settings()`.
            session_service: ADK session service backing live model
                conversation state. Defaults to a fresh `InMemorySessionService`
                (persistence across process restarts is handled separately by
                `store`, not by this service).
            runner: ADK `Runner` driving `TravelCoordinatorAgent`. Defaults to
                a `Runner` built from a fresh coordinator agent.
            confirm_booking: Async callable invoked with a blocked
                `confirm_reservation_booking` call's arguments; returns
                whether the traveler authorizes it. Defaults to an
                interactive stdin prompt.
        """
        self.session_id = session_id
        self.user_id = user_id
        self.settings = settings or get_settings()
        self.store = store
        self.async_memory = AsyncMemoryWorker(store)
        self.logger = get_logger("travel_agent_cli")
        self.tracer = get_tracer("TravelCoordinatorAgent")
        self._session_service = session_service or InMemorySessionService()
        self._agent = create_travel_coordinator_agent()
        self.runner = runner or Runner(
            app_name=APP_NAME, agent=self._agent, session_service=self._session_service
        )
        self._confirm_booking = confirm_booking or self._prompt_confirm_booking
        self._adk_session_ready = False

    @staticmethod
    async def _prompt_confirm_booking(booking_args: dict[str, Any]) -> bool:
        """Default `confirm_booking`: ask the traveler at the terminal."""
        booking_id = booking_args.get("provisional_booking_id", "<unknown>")
        prompt = (
            f"\nThe agent wants to CONFIRM booking '{booking_id}' -- this is a "
            "real reservation, not a reversible staging step. Authorize it? [y/N]: "
        )
        answer = await asyncio.to_thread(input, prompt)
        return answer.strip().lower() in {"y", "yes"}

    async def _ensure_adk_session(self) -> None:
        """Create the backing ADK session on first use of this instance."""
        if self._adk_session_ready:
            return
        existing = await self._session_service.get_session(
            app_name=APP_NAME, user_id=self.user_id, session_id=self.session_id
        )
        if existing is None:
            await self._session_service.create_session(
                app_name=APP_NAME, user_id=self.user_id, session_id=self.session_id
            )
        self._adk_session_ready = True

    async def send_message(self, text: str) -> str:
        """Handle one traveler message end-to-end and return the agent's reply.

        Persists the user message, runs the coordinator (looping through any
        HITL booking-authorization round trips), persists the agent's reply,
        and schedules background traveler-preference extraction.

        Args:
            text: The traveler's message.

        Returns:
            The coordinator's final text reply for this turn.
        """
        turn_id = uuid.uuid4().hex
        await self._ensure_adk_session()

        sanitized = sanitize_user_input(text)
        if sanitized.flagged:
            self.logger.warning(
                "INPUT_SANITIZATION_FLAGGED",
                session_id=self.session_id,
                turn_id=turn_id,
                reasons=sanitized.reasons,
            )
        text = sanitized.sanitized_text

        user_turn = ConversationTurn(
            session_id=self.session_id, turn_id=turn_id, role="user", content=text
        )
        await self.store.append_turn(user_turn)
        await compact_session_history(self.store, self.session_id, settings=self.settings)

        reply = await self._run_turn(text, turn_id)

        agent_turn = ConversationTurn(
            session_id=self.session_id,
            turn_id=turn_id,
            role="agent",
            agent_name=self._agent.name,
            content=reply,
        )
        await self.store.append_turn(agent_turn)
        self.async_memory.extract_preferences_in_background(
            self.session_id, [user_turn, agent_turn]
        )
        return reply

    async def _run_turn(self, message_text: str, turn_id: str) -> str:
        """Run the coordinator, transparently retrying once a blocked booking is authorized."""
        pending_text = message_text
        replies: list[str] = []
        for _round in range(MAX_BOOKING_AUTHORIZATION_ROUNDS):
            blocked_booking = await self._run_once(pending_text, turn_id, replies)
            if blocked_booking is None:
                break
            approved = await self._confirm_booking(blocked_booking)
            booking_id = blocked_booking.get("provisional_booking_id", "<unknown>")
            if not approved:
                replies.append(f"Understood -- I will not confirm booking '{booking_id}'.")
                break
            await self._authorize_booking(booking_id)
            pending_text = (
                f"The traveler has explicitly authorized booking '{booking_id}'. "
                "Proceed with confirm_reservation_booking now."
            )
        return "\n".join(part for part in replies if part)

    async def _run_once(
        self, message_text: str, turn_id: str, replies: list[str]
    ) -> dict[str, Any] | None:
        """Run the coordinator for a single model round trip.

        Args:
            message_text: The message to send this round.
            turn_id: The conversation turn these events belong to.
            replies: Accumulator that final-response text is appended to.

        Returns:
            The arguments of a `confirm_reservation_booking` call blocked by
            the HITL guardrail this round, or `None` if none was blocked.
        """
        model_name = self._agent.model
        content = types.Content(role="user", parts=[types.Part(text=message_text)])
        last_confirm_call_args: dict[str, Any] = {}
        blocked_booking: dict[str, Any] | None = None

        with start_agent_invocation_span(
            self.tracer,
            session_id=self.session_id,
            turn_id=turn_id,
            agent_name=self._agent.name,
            model_name=model_name,
        ):
            log_agent_intent(
                self.logger,
                session_id=self.session_id,
                turn_id=turn_id,
                agent_name=self._agent.name,
                model_name=model_name,
                action="run_coordinator_turn",
                reasoning="Handle traveler message and synthesize/relay a reply.",
            )
            start = time.monotonic()
            async for event in self.runner.run_async(
                user_id=self.user_id, session_id=self.session_id, new_message=content
            ):
                for call in event.get_function_calls():
                    if call.name == CONFIRM_BOOKING_TOOL_NAME:
                        last_confirm_call_args = dict(call.args or {})
                for func_response in event.get_function_responses():
                    if (
                        func_response.name == CONFIRM_BOOKING_TOOL_NAME
                        and isinstance(func_response.response, dict)
                        and func_response.response.get("error_code") == "AUTHORIZATION_REQUIRED"
                    ):
                        blocked_booking = last_confirm_call_args
                if event.is_final_response() and event.content and event.content.parts:
                    text_out = "".join(part.text or "" for part in event.content.parts)
                    if text_out:
                        replies.append(text_out)
            latency_ms = (time.monotonic() - start) * 1000
            log_agent_outcome(
                self.logger,
                session_id=self.session_id,
                turn_id=turn_id,
                agent_name=self._agent.name,
                model_name=model_name,
                status="blocked_awaiting_authorization" if blocked_booking else "success",
                latency_ms=latency_ms,
            )
        return blocked_booking

    async def _authorize_booking(self, provisional_booking_id: str) -> None:
        """Record traveler authorization for `provisional_booking_id` in the ADK session state.

        Applied via `append_event` with a `state_delta` (rather than
        mutating a fetched session's `.state` in place) since
        `InMemorySessionService.get_session` returns a copy; only a
        state-delta event is guaranteed to persist back into the service.
        """
        session = await self._session_service.get_session(
            app_name=APP_NAME, user_id=self.user_id, session_id=self.session_id
        )
        state = dict(session.state) if session else {}
        authorize_booking(state, provisional_booking_id)
        await self._session_service.append_event(
            session,
            Event(
                author="system",
                actions=EventActions(
                    state_delta={
                        AUTHORIZED_BOOKING_IDS_STATE_KEY: state[AUTHORIZED_BOOKING_IDS_STATE_KEY]
                    }
                ),
            ),
        )

    async def aclose(self) -> None:
        """Await any in-flight background preference extraction before shutdown."""
        await self.async_memory.wait_for_pending()


async def _run_session(
    session_id: str, user_id: str, on_ready: Callable[[TravelAgentSession], Awaitable[None]]
) -> None:
    """Connect a `SessionStore`, build a `TravelAgentSession`, and run `on_ready`."""
    settings = get_settings()
    configure_logging(settings)
    configure_tracing(settings)
    if not settings.has_gemini_api_key():
        print(
            "Warning: GEMINI_API_KEY is not set. Set it in .env before sending a "
            "message; the coordinator's model calls will fail without it.",
            file=sys.stderr,
        )

    async with SessionStore(settings=settings) as store:
        session = TravelAgentSession(
            session_id, store=store, user_id=user_id, settings=settings
        )
        try:
            await on_ready(session)
        finally:
            await session.aclose()


async def _run_single_message(session_id: str, user_id: str, message: str) -> None:
    async def _send(session: TravelAgentSession) -> None:
        reply = await session.send_message(message)
        print(reply)

    await _run_session(session_id, user_id, _send)


async def _run_interactive(session_id: str, user_id: str) -> None:
    async def _chat(session: TravelAgentSession) -> None:
        print(f"Smart Itinerary Planner & Booking Assistant -- session '{session_id}'.")
        print("Type 'exit' or 'quit' to end the conversation.\n")
        while True:
            try:
                user_input = await asyncio.to_thread(input, "You: ")
            except EOFError:
                break
            user_input = user_input.strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                break
            reply = await session.send_message(user_input)
            print(f"\nAgent: {reply}\n")

    await _run_session(session_id, user_id, _chat)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Smart Itinerary Planner & Booking Assistant -- local CLI."
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help=(
            "Persistent session id to resume (matches SessionStore history). "
            "Defaults to a freshly generated id."
        ),
    )
    parser.add_argument("--user-id", default=DEFAULT_USER_ID, help="Traveler identifier.")
    parser.add_argument(
        "--message",
        default=None,
        help="Send a single message non-interactively, print the reply, and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: `uv run python -m src.main`."""
    args = build_arg_parser().parse_args(argv)
    session_id = args.session_id or f"session-{uuid.uuid4().hex[:8]}"

    if args.message is not None:
        asyncio.run(_run_single_message(session_id, args.user_id, args.message))
    else:
        asyncio.run(_run_interactive(session_id, args.user_id))
    return 0


if __name__ == "__main__":
    sys.exit(main())
