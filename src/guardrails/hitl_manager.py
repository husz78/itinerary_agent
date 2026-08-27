"""Human-in-the-loop (HITL) guardrail for high-stakes booking confirmation.

`confirm_reservation_booking` (`src/tools/booking_tool.py`) only verifies
that a caller presented the exact `confirmation_token` issued when a booking
was staged; it has no visibility into the conversation, so it cannot itself
guarantee a human actually approved the charge. That enforcement lives here:
this module halts a `BookingAgent` tool call to `confirm_reservation_booking`
unless the traveler has explicitly authorized that exact
`provisional_booking_id` in this session, tracked via session state rather
than trusting the model's own judgment about whether "the user said yes".

Session state key `AUTHORIZED_BOOKING_IDS_STATE_KEY` holds the list of
`provisional_booking_id`s the traveler has explicitly authorized so far.
Authorization is granted by the surrounding application -- a CLI
confirmation prompt, or an ADK `request_confirmation` flow -- calling
`authorize_booking` only after the traveler has explicitly approved that
exact booking; `before_confirm_booking_tool_callback` never grants it on its
own.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, MutableMapping

from pydantic import BaseModel, Field

CONFIRM_BOOKING_TOOL_NAME = "confirm_reservation_booking"
AUTHORIZED_BOOKING_IDS_STATE_KEY = "authorized_booking_ids"


class HITLDecision(StrEnum):
    """Outcome of a human-in-the-loop authorization check."""

    ALLOW = "allow"
    BLOCK_MISSING_AUTHORIZATION = "block_missing_authorization"


class HITLGateResult(BaseModel):
    """Result of checking whether a high-stakes action may proceed.

    Attributes:
        decision: Whether the action may proceed.
        message: Human/agent-readable explanation of the decision.
    """

    decision: HITLDecision = Field(..., description="Whether the action may proceed.")
    message: str = Field(..., description="Explanation of the decision.")


def authorize_booking(state: MutableMapping[str, Any], provisional_booking_id: str) -> None:
    """Record that the traveler has explicitly authorized a staged booking.

    Must only be called by the surrounding application after the traveler
    has, in this conversation, explicitly confirmed the exact booking
    identified by `provisional_booking_id` (matching provider, slot, and
    price). Never call this speculatively or on the agent's own judgment.

    Args:
        state: A dict-like session/tool-context state object to mutate in
            place (e.g. a plain `dict` in tests, or an ADK `ToolContext`/
            `CallbackContext` `.state`).
        provisional_booking_id: The exact staged booking id the traveler
            authorized.

    Returns:
        None.
    """
    authorized = set(state.get(AUTHORIZED_BOOKING_IDS_STATE_KEY, []))
    authorized.add(provisional_booking_id)
    state[AUTHORIZED_BOOKING_IDS_STATE_KEY] = sorted(authorized)


def check_booking_authorization(
    state: MutableMapping[str, Any], provisional_booking_id: str
) -> HITLGateResult:
    """Decide whether `confirm_reservation_booking` may run for a given booking.

    Args:
        state: A dict-like session/tool-context state object to read from.
        provisional_booking_id: The staged booking id the tool call targets.

    Returns:
        `HITLGateResult` with `decision=ALLOW` if `provisional_booking_id`
        is present in `state[AUTHORIZED_BOOKING_IDS_STATE_KEY]`, otherwise
        `decision=BLOCK_MISSING_AUTHORIZATION` with a message instructing
        the caller to obtain explicit traveler authorization first.
    """
    authorized = set(state.get(AUTHORIZED_BOOKING_IDS_STATE_KEY, []))
    if provisional_booking_id and provisional_booking_id in authorized:
        return HITLGateResult(
            decision=HITLDecision.ALLOW,
            message=f"Booking '{provisional_booking_id}' is authorized.",
        )
    return HITLGateResult(
        decision=HITLDecision.BLOCK_MISSING_AUTHORIZATION,
        message=(
            f"Booking '{provisional_booking_id}' has not been explicitly "
            "authorized by the traveler in this conversation. Ask the "
            "traveler to confirm this exact booking (provider, price, and "
            "date) before calling confirm_reservation_booking again."
        ),
    )


async def before_confirm_booking_tool_callback(
    tool: Any, args: dict[str, Any], tool_context: Any
) -> dict[str, Any] | None:
    """ADK `before_tool_callback`: halt unauthorized `confirm_reservation_booking` calls.

    Attach to `BookingAgent` as `before_tool_callback`. Every other tool
    call passes straight through; only `confirm_reservation_booking` is
    gated on `check_booking_authorization`.

    Args:
        tool: The ADK `BaseTool` about to be invoked. Only `tool.name` is
            read.
        args: The arguments the model supplied for the call, expected to
            include `provisional_booking_id`.
        tool_context: ADK `ToolContext`; its `.state` is read to check
            authorization.

    Returns:
        `None` to let the call proceed unmodified when `tool.name` isn't
        `confirm_reservation_booking` or the booking is authorized.
        Otherwise a guided error dict matching this project's
        `ToolErrorEnvelope` shape, which ADK returns to the model as the
        tool's result instead of invoking the real function.
    """
    if getattr(tool, "name", None) != CONFIRM_BOOKING_TOOL_NAME:
        return None

    provisional_booking_id = args.get("provisional_booking_id", "")
    gate = check_booking_authorization(tool_context.state, provisional_booking_id)
    if gate.decision is HITLDecision.ALLOW:
        return None

    return {
        "status": "error",
        "error_code": "AUTHORIZATION_REQUIRED",
        "message": gate.message,
        "recovery_instruction": (
            "Ask the traveler to explicitly confirm this exact booking, "
            "then have the application call authorize_booking with the "
            "provisional_booking_id before retrying confirm_reservation_booking."
        ),
    }
