"""PII redaction engine.

Regex-based scrubbing middleware that masks credit card numbers, passport
numbers, email addresses, and phone numbers before any data is written to
logs, traces, or persistent local state (`data/travel_agent.db`). This module
has no dependency on `logger.py` or `tracer.py` so both can call into it
without a circular import, and so it can also be used directly by
`memory/session_store.py` before writing conversation turns to SQLite.

Scrubbing is applied at two levels:
    - `scrub_text`: regex substitution over a single string.
    - `scrub_value`: recursive walk over dicts/lists/tuples/`BaseModel`
      instances (the shapes structured log events, span attributes, and
      stored session state actually take), scrubbing every string leaf and
      additionally blanket-redacting any value whose dict key looks
      sensitive by name (e.g. `password`, `api_key`) regardless of whether
      its value matches a known PII pattern.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel


class PIICategory(StrEnum):
    """Machine-readable category for a redacted span of text."""

    EMAIL = "EMAIL"
    CREDIT_CARD = "CREDIT_CARD"
    PHONE_NUMBER = "PHONE_NUMBER"
    PASSPORT_NUMBER = "PASSPORT_NUMBER"


_REDACTION_TOKEN = {
    PIICategory.EMAIL: "[REDACTED_EMAIL]",
    PIICategory.CREDIT_CARD: "[REDACTED_CREDIT_CARD]",
    PIICategory.PHONE_NUMBER: "[REDACTED_PHONE_NUMBER]",
    PIICategory.PASSPORT_NUMBER: "[REDACTED_PASSPORT_NUMBER]",
}

# Generic redaction used for dict values whose *key* names them as sensitive,
# independent of whether the value itself matches a pattern below.
_REDACTED_FIELD_TOKEN = "[REDACTED_FIELD]"

# Dict keys (case-insensitive, substring match) that are always fully
# redacted regardless of their value's shape. Covers secrets and identifiers
# that don't have a reliable regex signature of their own.
_SENSITIVE_KEY_SUBSTRINGS = (
    "password",
    "api_key",
    "apikey",
    "secret",
    "credit_card",
    "card_number",
    "cvv",
    "ssn",
    "social_security",
    "passport",
    "auth_token",
    "confirmation_token",
)

_EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[A-Za-z]{2,}\b")

# 13-19 digits, optionally grouped with spaces or hyphens (e.g. the common
# 4-4-4-4 card layout). Candidate spans are additionally Luhn-validated
# before being treated as a credit card, which keeps this from firing on
# arbitrary long digit runs (order/booking IDs, timestamps, etc.).
_CREDIT_CARD_CANDIDATE_PATTERN = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")

# Phone numbers: optional leading country code and/or a parenthesized area
# code, then 2-4 groups of 3-4 digits separated by spaces/dots/dashes,
# totaling 7-15 digits. Groups are deliberately restricted to 3-4 digits
# (never 2) so this can't match a 2-digit day/month segment of an ISO 8601
# date (`YYYY-MM-DD`) or a timestamp — those never form two consecutive
# 3-4 digit groups.
_PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+\d{1,3}[-.\s]?)?(?:\(\d{3,4}\)[-.\s]?)?\d{3,4}(?:[-.\s]\d{3,4}){1,3}(?!\d)"
)

# 1-2 uppercase letters followed by 6-9 digits, e.g. "A12345678" (common
# ICAO-style machine-readable passport number shape).
_PASSPORT_PATTERN = re.compile(r"\b[A-Z]{1,2}[0-9]{6,9}\b")


def _luhn_checksum(digits: str) -> bool:
    """Return True if `digits` (a string of only digit characters) passes the Luhn check."""
    total = 0
    parity = len(digits) % 2
    for index, char in enumerate(digits):
        n = int(char)
        if index % 2 == parity:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _redact_credit_cards(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        digits_only = re.sub(r"[ -]", "", match.group(0))
        if len(digits_only) < 13 or not _luhn_checksum(digits_only):
            return match.group(0)
        return _REDACTION_TOKEN[PIICategory.CREDIT_CARD]

    return _CREDIT_CARD_CANDIDATE_PATTERN.sub(_replace, text)


def scrub_text(text: str) -> str:
    """Mask emails, credit cards, phone numbers, and passport numbers in `text`.

    Args:
        text: Free-text string that may contain PII, e.g. a log message, a
            user chat turn, or a serialized tool argument.

    Returns:
        A copy of `text` with every detected PII span replaced by a stable
        `[REDACTED_<CATEGORY>]` token. Text with no PII is returned
        unchanged. Detection order is: email, credit card (Luhn-validated),
        phone number, passport number — chosen so a credit card's digit run
        is consumed before the phone pattern can partially match it.

    Example:
        >>> scrub_text("Reach me at jane@example.com or 415-555-0100.")
        'Reach me at [REDACTED_EMAIL] or [REDACTED_PHONE_NUMBER].'
    """
    if not text:
        return text
    scrubbed = _EMAIL_PATTERN.sub(_REDACTION_TOKEN[PIICategory.EMAIL], text)
    scrubbed = _redact_credit_cards(scrubbed)
    scrubbed = _PHONE_PATTERN.sub(_REDACTION_TOKEN[PIICategory.PHONE_NUMBER], scrubbed)
    scrubbed = _PASSPORT_PATTERN.sub(_REDACTION_TOKEN[PIICategory.PASSPORT_NUMBER], scrubbed)
    return scrubbed


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(substring in lowered for substring in _SENSITIVE_KEY_SUBSTRINGS)


def scrub_value(value: Any) -> Any:
    """Recursively scrub PII from a string, dict, list/tuple, or `BaseModel`.

    Intended as the single choke point that `observability/logger.py`,
    `observability/tracer.py`, and `memory/session_store.py` call before a
    structured log event, span attribute set, or persisted session record
    leaves process memory.

    Args:
        value: Arbitrary structured data: a `str`, `dict`, `list`, `tuple`,
            `pydantic.BaseModel`, or a primitive (`int`/`float`/`bool`/
            `None`), possibly nested.

    Returns:
        A structurally equivalent value (dicts stay dicts, lists stay lists,
        `BaseModel` instances are converted to plain `dict` via
        `model_dump()`) with every string leaf passed through `scrub_text`
        and every dict value whose key name matches a known-sensitive
        substring (e.g. `password`, `api_key`, `passport`) fully replaced
        with `[REDACTED_FIELD]`, regardless of whether the value itself
        matched a PII pattern. Non-string primitives are returned unchanged.

    Example:
        >>> scrub_value({"email": "jane@example.com", "api_key": "sk-live-123"})
        {'email': '[REDACTED_EMAIL]', 'api_key': '[REDACTED_FIELD]'}
    """
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if isinstance(value, str):
        return scrub_text(value)
    if isinstance(value, dict):
        return {
            key: _REDACTED_FIELD_TOKEN if _is_sensitive_key(str(key)) else scrub_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        scrubbed_items = [scrub_value(item) for item in value]
        return type(value)(scrubbed_items) if isinstance(value, tuple) else scrubbed_items
    return value
