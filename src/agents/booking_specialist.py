"""BookingAgent: reservation assembly, payment calculation, booking payload
generation, and HITL-gated confirmation.

Routed to `Settings.model_pro` (`gemini-3.1-pro`) per the project's
strategic model routing: precise price calculation and schema-strict
payload creation are the highest-stakes tool calls in this system and
warrant the stronger model. `before_tool_callback` wires in
`src.guardrails.hitl_manager.before_confirm_booking_tool_callback`, so
`confirm_reservation_booking` is halted with a guided error unless the
traveler has explicitly authorized that exact staged booking in this
session -- the human-in-the-loop hook required for this high-stakes action.
"""

from __future__ import annotations

from google.adk.agents import Agent

from src.agents.constitution import BOOKING_AGENT_MANDATE, build_instruction
from src.config import get_settings
from src.guardrails.hitl_manager import before_confirm_booking_tool_callback
from src.tools.booking_tool import (
    confirm_reservation_booking as _confirm_reservation_booking,
    stage_provisional_booking as _stage_provisional_booking,
)

AGENT_NAME = "BookingAgent"

AGENT_DESCRIPTION = (
    "Stages and confirms flight, hotel, activity, car rental, and "
    "restaurant reservations. Delegate here to price out or book a "
    "specific listing."
)


def stage_provisional_booking(
    reservation_type: str,
    provider_id: str,
    slot: str,
    price: float,
) -> dict:
    """Stage a provisional (non-charging, reversible) reservation for traveler review.

    Always the first step of any booking. Never charges a card or commits
    an irreversible reservation. Present the returned summary to the
    traveler and only call confirm_reservation_booking after the traveler
    has explicitly authorized this exact booking in conversation.

    Args:
        reservation_type: One of "flight", "hotel", "activity",
            "car_rental", "restaurant".
        provider_id: Non-empty identifier of the provider/listing, e.g. a
            flight number or hotel ID.
        slot: Non-empty description of the reservation time slot/date, e.g.
            "2026-09-01T14:00:00".
        price: Total price in US dollars. Must be greater than 0.

    Returns:
        On success, a dict with `status="success"` and a `data` key holding
        the staged booking, including the `confirmation_token` that must be
        relayed through confirm_reservation_booking. On failure, a dict
        with `status="error"`, `error_code`, `message`, and
        `recovery_instruction`.
    """
    result = _stage_provisional_booking(reservation_type, provider_id, slot, price)
    return result.model_dump(mode="json")


def confirm_reservation_booking(
    provisional_booking_id: str,
    user_confirmation_token: str,
) -> dict:
    """Finalize a staged reservation after explicit traveler authorization.

    Only call this after the traveler has explicitly authorized the exact
    booking identified by provisional_booking_id in conversation. A
    human-in-the-loop guardrail halts this call with an
    AUTHORIZATION_REQUIRED error if that authorization has not been
    recorded yet, regardless of whether user_confirmation_token is correct.

    Args:
        provisional_booking_id: The provisional_booking_id returned by a
            prior stage_provisional_booking call.
        user_confirmation_token: The confirmation code to verify against
            the one issued at staging time.

    Returns:
        On success, a dict with `status="success"` and a `data` key holding
        the finalized reservation with `status="confirmed"`. On failure, a
        dict with `status="error"`, `error_code`, `message`, and
        `recovery_instruction` (e.g. AUTHORIZATION_REQUIRED if the
        traveler's explicit authorization is still missing).
    """
    result = _confirm_reservation_booking(provisional_booking_id, user_confirmation_token)
    return result.model_dump(mode="json")


def create_booking_agent(model: str | None = None) -> Agent:
    """Build a fresh `BookingAgent` instance with the HITL confirmation guard attached.

    A factory rather than a module-level singleton: ADK assigns each
    sub-agent a single parent, so every coordinator built via
    `src.agents.coordinator.create_travel_coordinator_agent` needs its own
    specialist instances.

    Args:
        model: Model id override. Defaults to `Settings.model_pro`
            (`gemini-3.1-pro`).

    Returns:
        A configured `Agent` bound to `stage_provisional_booking` and
        `confirm_reservation_booking`, with
        `before_confirm_booking_tool_callback` enforcing HITL authorization.
    """
    settings = get_settings()
    return Agent(
        name=AGENT_NAME,
        model=model or settings.model_pro,
        description=AGENT_DESCRIPTION,
        instruction=build_instruction(BOOKING_AGENT_MANDATE),
        tools=[stage_provisional_booking, confirm_reservation_booking],
        before_tool_callback=before_confirm_booking_tool_callback,
    )
