"""Input sanitization guardrail.

Runs over every raw traveler message before it reaches an agent, per the
project's "Guardrails & Policy Plugins" requirement (input sanitization,
alongside `budget_guardrail`'s hallucination/policy checks and
`hitl_manager`'s HITL gate). Two independent, deterministic concerns:

    - Control characters (other than newline/tab) are stripped and the
      message is truncated to `MAX_INPUT_LENGTH`, since neither can carry
      meaningful traveler intent and an unbounded string would otherwise
      flow straight into `SessionStore` and the model prompt.
    - A small set of known prompt-injection phrasings (e.g. "ignore previous
      instructions", "you are now") is flagged so the caller can log/audit
      the attempt. Detection only flags; it never blocks the message
      outright, since `constitution.CORE_CONSTITUTION`'s non-negotiable
      rules are the actual defense against a hijacked instruction set, and a
      false positive here should not silently drop a traveler's message.
"""

from __future__ import annotations

import re
import unicodedata

from pydantic import BaseModel, Field

MAX_INPUT_LENGTH = 4000

# Deliberately conservative phrasing list: short, common prompt-injection
# openers. Case-insensitive substring match, not a exhaustive classifier --
# this flags obvious attempts for logging/audit, it does not replace the
# constitution's non-negotiable rules as the actual defense.
_INJECTION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"ignore (all )?(previous|prior|above) instructions",
        r"disregard (all )?(previous|prior|above) instructions",
        r"you are now\b",
        r"forget (all )?(your|previous) instructions",
        r"reveal (your |the )?(system prompt|instructions)",
        r"act as (if you were|a) (?!a travel)",
        r"new instructions?:",
    )
)

# Control characters (Unicode category Cc) other than tab/newline/carriage
# return, which are legitimate in free-text traveler messages.
_ALLOWED_CONTROL_CHARS = {"\t", "\n", "\r"}


class SanitizationResult(BaseModel):
    """Result of sanitizing one raw traveler message.

    Attributes:
        sanitized_text: The message with control characters stripped and
            length-capped at `MAX_INPUT_LENGTH`. Safe to persist and pass to
            an agent regardless of `flagged`.
        flagged: Whether a known prompt-injection phrasing was detected.
            Informational only -- `sanitized_text` is never withheld.
        reasons: Human-readable descriptions of what was flagged/altered,
            empty if the input needed no changes and nothing was flagged.
        truncated: Whether the original input exceeded `MAX_INPUT_LENGTH`
            and was cut down.
    """

    sanitized_text: str = Field(..., description="Cleaned, length-capped message text.")
    flagged: bool = Field(
        default=False, description="Whether a known prompt-injection phrasing was detected."
    )
    reasons: list[str] = Field(
        default_factory=list, description="Explanations of what was altered or flagged."
    )
    truncated: bool = Field(
        default=False, description="Whether the input was cut down to MAX_INPUT_LENGTH."
    )


def _strip_control_characters(text: str) -> tuple[str, bool]:
    cleaned_chars = [
        char
        for char in text
        if char in _ALLOWED_CONTROL_CHARS or unicodedata.category(char) != "Cc"
    ]
    cleaned = "".join(cleaned_chars)
    return cleaned, cleaned != text


def sanitize_user_input(text: str) -> SanitizationResult:
    """Strip unsafe characters, cap length, and flag likely prompt-injection attempts.

    Args:
        text: Raw traveler message, exactly as typed/received.

    Returns:
        A `SanitizationResult`. `sanitized_text` is always safe to persist
        and forward to an agent; `flagged`/`reasons` surface anything an
        application layer may want to log or audit without dropping the
        traveler's message.
    """
    reasons: list[str] = []

    stripped, control_chars_removed = _strip_control_characters(text)
    if control_chars_removed:
        reasons.append("Removed non-printable control characters.")

    truncated = len(stripped) > MAX_INPUT_LENGTH
    if truncated:
        stripped = stripped[:MAX_INPUT_LENGTH]
        reasons.append(f"Truncated input to {MAX_INPUT_LENGTH} characters.")

    flagged = False
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(stripped):
            flagged = True
            reasons.append(f"Matched potential prompt-injection phrasing: {pattern.pattern!r}")

    return SanitizationResult(
        sanitized_text=stripped,
        flagged=flagged,
        reasons=reasons,
        truncated=truncated,
    )
