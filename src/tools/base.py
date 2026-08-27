"""Standard tool result envelopes and guided error models.

Every tool in this project returns a `ToolResultEnvelope` on success or a
`ToolErrorEnvelope` on failure instead of raising an unhandled exception.
This guarantees the calling agent always receives a structured, strictly
typed payload it can either act on directly or use to decide how to recover
(retry with different arguments, ask the user a clarifying question, or fall
back to another tool), per the project's guided error handling requirement.

Example:
    >>> from src.tools.base import ToolErrorCode, make_error, make_success
    >>> make_success({"temp_c": 22}).model_dump()
    {'status': <ToolStatus.SUCCESS: 'success'>, 'data': {'temp_c': 22}}
    >>> make_error(
    ...     ToolErrorCode.LOCATION_AMBIGUOUS,
    ...     "Multiple matches found for 'Paris' (France, Texas, Ontario).",
    ...     "Specify country or state, or retry with an ISO country code.",
    ... ).error_code
    <ToolErrorCode.LOCATION_AMBIGUOUS: 'LOCATION_AMBIGUOUS'>
"""

from enum import StrEnum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ToolStatus(StrEnum):
    """Discriminator value identifying which envelope a payload uses."""

    SUCCESS = "success"
    ERROR = "error"


class ToolErrorCode(StrEnum):
    """Stable, machine-readable error identifiers returned by tools.

    Agents (and tests) should branch on these codes rather than parsing the
    free-text `message` field, which may change wording over time.
    """

    LOCATION_AMBIGUOUS = "LOCATION_AMBIGUOUS"
    LOCATION_NOT_FOUND = "LOCATION_NOT_FOUND"
    INVALID_DATE_RANGE = "INVALID_DATE_RANGE"
    RATE_LIMITED = "RATE_LIMITED"
    UPSTREAM_API_ERROR = "UPSTREAM_API_ERROR"
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    AUTHORIZATION_REQUIRED = "AUTHORIZATION_REQUIRED"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    ROUTE_NOT_SUPPORTED = "ROUTE_NOT_SUPPORTED"


class ToolResultEnvelope(BaseModel, Generic[T]):
    """Successful tool response wrapper.

    Attributes:
        status: Always `ToolStatus.SUCCESS`.
        data: The tool-specific payload. Concrete tools should parameterize
            this generic with their own `pydantic.BaseModel` output schema
            (e.g. `ToolResultEnvelope[WeatherForecast]`).
    """

    status: ToolStatus = Field(
        default=ToolStatus.SUCCESS,
        description="Always 'success' for this envelope type.",
    )
    data: T = Field(..., description="Tool-specific successful result payload.")


class ToolErrorEnvelope(BaseModel):
    """Guided error response wrapper returned instead of raising.

    Attributes:
        status: Always `ToolStatus.ERROR`.
        error_code: Stable machine-readable error identifier from
            `ToolErrorCode`.
        message: Human-readable description of what went wrong, suitable for
            relaying to the end user.
        recovery_instruction: Actionable guidance telling the calling agent
            exactly what to do next (retry with different arguments, ask the
            user a clarifying question, or invoke a different tool).
    """

    status: ToolStatus = Field(
        default=ToolStatus.ERROR,
        description="Always 'error' for this envelope type.",
    )
    error_code: ToolErrorCode = Field(
        ..., description="Stable machine-readable error identifier."
    )
    message: str = Field(
        ..., description="Human-readable description of the failure."
    )
    recovery_instruction: str = Field(
        ...,
        description="Actionable next step for the agent to recover from this error.",
    )


ToolOutcome = ToolResultEnvelope[T] | ToolErrorEnvelope


def make_success(data: T) -> ToolResultEnvelope[T]:
    """Wrap a successful tool result payload in a `ToolResultEnvelope`.

    Args:
        data: The tool-specific result payload (typically a `BaseModel`,
            `dict`, or `list`).

    Returns:
        A `ToolResultEnvelope` with `status="success"` and the given data.
    """
    return ToolResultEnvelope(data=data)


def make_error(
    error_code: ToolErrorCode,
    message: str,
    recovery_instruction: str,
) -> ToolErrorEnvelope:
    """Build a guided `ToolErrorEnvelope` instead of raising an exception.

    Args:
        error_code: Stable machine-readable error identifier.
        message: Human-readable description of the failure.
        recovery_instruction: Actionable next step for the calling agent.

    Returns:
        A `ToolErrorEnvelope` with `status="error"` and the given details.
    """
    return ToolErrorEnvelope(
        error_code=error_code,
        message=message,
        recovery_instruction=recovery_instruction,
    )
