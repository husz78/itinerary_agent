"""Staging & reservation tools: `stage_provisional_booking`, `confirm_reservation_booking`.

Implements the two-step, human-in-the-loop (HITL) booking flow required for
high-stakes actions: a booking is first *staged* (no charge, no irreversible
side effect) and returned with a single-use `confirmation_token`. The
booking is only finalized when `confirm_reservation_booking` is called with
that exact token. Token matching is enforced here; it is *not* sufficient
HITL enforcement on its own — the coordinator's guardrail layer
(`src/guardrails/hitl_manager.py`, Phase 5) is responsible for halting
execution and only ever surfacing/forwarding a token after the end user has
explicitly authorized the booking in conversation. This tool cannot see the
conversation, so it can only guarantee "the caller presented the exact token
issued for this specific staged booking", not "a human actually approved it".

Staged bookings are held in an in-memory, process-local store rather than
`data/travel_agent.db`; a real deployment would persist them via
`src/memory/session_store.py` (Phase 4) so staged bookings survive process
restarts, but the public function signatures and the `ToolResultEnvelope` /
`ToolErrorEnvelope` contract would stay unchanged.
"""

from __future__ import annotations

from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field

from src.tools.base import (
    ToolErrorCode,
    ToolErrorEnvelope,
    ToolOutcome,
    ToolResultEnvelope,
    make_error,
    make_success,
)

VALID_RESERVATION_TYPES = {"flight", "hotel", "activity", "car_rental", "restaurant"}


class BookingStatus(StrEnum):
    """Lifecycle state of a staged reservation."""

    PENDING_CONFIRMATION = "pending_confirmation"
    CONFIRMED = "confirmed"


class ProvisionalBooking(BaseModel):
    """A staged, not-yet-charged reservation awaiting user authorization."""

    provisional_booking_id: str = Field(
        ..., description="Unique identifier for this staged booking."
    )
    reservation_type: str = Field(
        ..., description=f"One of the known types: {sorted(VALID_RESERVATION_TYPES)}."
    )
    provider_id: str = Field(
        ..., description="Identifier of the provider/listing, e.g. a flight number or hotel ID."
    )
    slot: str = Field(
        ..., description="Reservation time slot/date, e.g. '2026-09-01T14:00:00'."
    )
    price: float = Field(..., gt=0, description="Total price in US dollars.")
    status: BookingStatus = Field(
        ..., description="Always 'pending_confirmation' immediately after staging."
    )
    confirmation_token: str = Field(
        ...,
        description="Single-use code that must be echoed back via "
        "confirm_reservation_booking's user_confirmation_token argument, only "
        "after the end user has explicitly authorized this exact booking.",
    )


class ConfirmedReservation(BaseModel):
    """A finalized reservation after successful token confirmation."""

    provisional_booking_id: str = Field(
        ..., description="Identifier of the booking that was confirmed."
    )
    reservation_type: str = Field(..., description="Type of the confirmed reservation.")
    provider_id: str = Field(..., description="Identifier of the provider/listing.")
    slot: str = Field(..., description="Reservation time slot/date.")
    price: float = Field(..., gt=0, description="Total price in US dollars.")
    status: BookingStatus = Field(..., description="Always 'confirmed'.")


# Process-local staging store: provisional_booking_id -> internal record.
# Not persisted; see module docstring.
_STAGED_BOOKINGS: dict[str, dict] = {}


def stage_provisional_booking(
    reservation_type: str,
    provider_id: str,
    slot: str,
    price: float,
) -> ToolOutcome[ProvisionalBooking]:
    """Stage a provisional (non-charging, reversible) reservation for user review.

    This is always the first step of a booking. It never charges a card or
    commits an irreversible reservation. The caller (agent) must present the
    returned summary to the user and only call `confirm_reservation_booking`
    with the returned `confirmation_token` after the user has explicitly
    authorized this exact booking in conversation.

    Args:
        reservation_type: One of "flight", "hotel", "activity", "car_rental",
            "restaurant".
        provider_id: Non-empty identifier of the provider/listing, e.g. a
            flight number or hotel ID.
        slot: Non-empty description of the reservation time slot/date, e.g.
            "2026-09-01T14:00:00" or "2026-09-01 to 2026-09-05".
        price: Total price in US dollars. Must be greater than 0.

    Returns:
        On success, `ToolResultEnvelope[ProvisionalBooking]` whose `data`
        holds the staged booking, including the `confirmation_token` the
        caller must relay through `confirm_reservation_booking`.
        On failure, `ToolErrorEnvelope` with `VALIDATION_ERROR` for an
        unknown `reservation_type`, a blank `provider_id`/`slot`, or a
        non-positive `price`.

    Example:
        >>> result = stage_provisional_booking("hotel", "HTL-42", "2026-09-01", 199.0)
        >>> result.data.status
        <BookingStatus.PENDING_CONFIRMATION: 'pending_confirmation'>
    """
    if reservation_type not in VALID_RESERVATION_TYPES:
        return make_error(
            ToolErrorCode.VALIDATION_ERROR,
            f"Unknown reservation_type '{reservation_type}'. "
            f"Valid values: {sorted(VALID_RESERVATION_TYPES)}.",
            f"Retry with one of: {sorted(VALID_RESERVATION_TYPES)}.",
        )
    if not provider_id.strip():
        return make_error(
            ToolErrorCode.VALIDATION_ERROR,
            "provider_id must not be blank.",
            "Retry with a non-empty provider_id identifying the specific listing.",
        )
    if not slot.strip():
        return make_error(
            ToolErrorCode.VALIDATION_ERROR,
            "slot must not be blank.",
            "Retry with a non-empty slot describing the reservation date/time.",
        )
    if price <= 0:
        return make_error(
            ToolErrorCode.VALIDATION_ERROR,
            f"price must be greater than 0, got {price}.",
            "Retry with a positive price in US dollars.",
        )

    booking_id = f"book_{uuid4().hex[:12]}"
    confirmation_token = uuid4().hex[:8].upper()
    _STAGED_BOOKINGS[booking_id] = {
        "reservation_type": reservation_type,
        "provider_id": provider_id,
        "slot": slot,
        "price": price,
        "status": BookingStatus.PENDING_CONFIRMATION,
        "confirmation_token": confirmation_token,
    }

    return make_success(
        ProvisionalBooking(
            provisional_booking_id=booking_id,
            reservation_type=reservation_type,
            provider_id=provider_id,
            slot=slot,
            price=price,
            status=BookingStatus.PENDING_CONFIRMATION,
            confirmation_token=confirmation_token,
        )
    )


def confirm_reservation_booking(
    provisional_booking_id: str,
    user_confirmation_token: str,
) -> ToolOutcome[ConfirmedReservation]:
    """Finalize a staged reservation after explicit user authorization.

    Only call this after the user has explicitly authorized the exact
    booking identified by `provisional_booking_id` in conversation. This
    function itself only verifies that `user_confirmation_token` matches the
    token issued when the booking was staged; the caller is responsible for
    never invoking it without genuine, in-conversation user approval.

    Args:
        provisional_booking_id: The `provisional_booking_id` returned by a
            prior `stage_provisional_booking` call.
        user_confirmation_token: The confirmation code to verify against the
            one issued at staging time.

    Returns:
        On success, `ToolResultEnvelope[ConfirmedReservation]` whose `data`
        holds the finalized reservation with `status="confirmed"`.
        On failure, `ToolErrorEnvelope` with one of:
            - `RESOURCE_NOT_FOUND`: no staged booking matches
              `provisional_booking_id` (unknown, or already expired from the
              in-memory store).
            - `VALIDATION_ERROR`: the booking was already confirmed
              previously.
            - `AUTHORIZATION_REQUIRED`: `user_confirmation_token` does not
              match the token issued for this booking.

    Example:
        >>> staged = stage_provisional_booking("hotel", "HTL-42", "2026-09-01", 199.0)
        >>> result = confirm_reservation_booking(
        ...     staged.data.provisional_booking_id, staged.data.confirmation_token
        ... )
        >>> result.data.status
        <BookingStatus.CONFIRMED: 'confirmed'>
    """
    record = _STAGED_BOOKINGS.get(provisional_booking_id)
    if record is None:
        return make_error(
            ToolErrorCode.RESOURCE_NOT_FOUND,
            f"No staged booking found for provisional_booking_id "
            f"'{provisional_booking_id}'.",
            "Call stage_provisional_booking first, or double-check the "
            "provisional_booking_id for typos.",
        )

    if record["status"] == BookingStatus.CONFIRMED:
        return make_error(
            ToolErrorCode.VALIDATION_ERROR,
            f"Booking '{provisional_booking_id}' has already been confirmed.",
            "No action needed; this booking was already finalized. Do not "
            "retry the confirmation.",
        )

    if user_confirmation_token != record["confirmation_token"]:
        return make_error(
            ToolErrorCode.AUTHORIZATION_REQUIRED,
            f"user_confirmation_token does not match the token issued for "
            f"booking '{provisional_booking_id}'.",
            "Ask the user to explicitly authorize this exact booking, then "
            "retry with the exact confirmation_token from stage_provisional_booking.",
        )

    record["status"] = BookingStatus.CONFIRMED
    return make_success(
        ConfirmedReservation(
            provisional_booking_id=provisional_booking_id,
            reservation_type=record["reservation_type"],
            provider_id=record["provider_id"],
            slot=record["slot"],
            price=record["price"],
            status=BookingStatus.CONFIRMED,
        )
    )
