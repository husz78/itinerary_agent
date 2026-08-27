"""History compaction: sliding window + automatic summarization.

Two complementary mechanisms keep a session's conversation history from
unboundedly growing the prompt sent to the model:

    - `get_sliding_window`: a pure, synchronous windowing function that
      returns only the most recent `settings.history_window_turns` stored
      messages. Callers use this to build the LLM prompt from history
      without ever mutating what's persisted in `SessionStore`.
    - `compact_session_history`: once the *stored* history's estimated
      token count crosses `settings.summarization_token_threshold`,
      everything older than the sliding window is folded into a single
      `role="summary"` message via an LLM-backed (or, by default, a
      deterministic fallback) summarizer, and `SessionStore.replace_turns`
      rewrites the session's history to `[summary, *window]`. This is the
      local equivalent of ADK's context compaction.

Token counts are estimated locally via a `len(text) // 4` heuristic rather
than a live model tokenizer, since this project runs 100% locally and must
not spend an API call just to decide whether to summarize.
"""

from __future__ import annotations

from typing import Awaitable, Callable

from src.config import Settings, get_settings
from src.memory.session_store import ConversationTurn, SessionStore

SummarizerFn = Callable[[list[ConversationTurn]], Awaitable[str]]


def estimate_token_count(text: str) -> int:
    """Approximate the LLM token count of `text` using a local length heuristic.

    Args:
        text: Text to estimate.

    Returns:
        `len(text) // 4` (roughly the common "~4 chars per token" rule of
        thumb for English text), floored at 1 for any non-empty string, and
        0 for an empty string. This is an approximation only: it exists so
        compaction can decide when to summarize without a live tokenizer or
        API call.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_turns_token_count(turns: list[ConversationTurn]) -> int:
    """Sum the estimated token count of every message's content in `turns`."""
    return sum(estimate_token_count(turn.content) for turn in turns)


def get_sliding_window(
    turns: list[ConversationTurn], window_turns: int
) -> list[ConversationTurn]:
    """Return only the most recent `window_turns` messages from `turns`.

    Args:
        turns: Full chronological message history (oldest first).
        window_turns: Number of most recent messages to keep. Non-positive
            values return an empty window.

    Returns:
        The trailing slice of `turns` of length `min(len(turns), window_turns)`,
        in the same chronological order.
    """
    if window_turns <= 0:
        return []
    return turns[-window_turns:]


async def default_summarizer(turns: list[ConversationTurn]) -> str:
    """Deterministic fallback summarizer that does not call an LLM.

    Concatenates each message's role and content into a plain-text recap.
    Used for local testing and as a safe fallback; production callers
    should instead supply an LLM-backed `SummarizerFn` routed to
    `gemini-3.5-flash` (per this project's fast-task model routing) so the
    resulting summary is coherent prose rather than a raw transcript.

    Args:
        turns: The older-than-window messages being folded away.

    Returns:
        A single summary string.
    """
    lines = [f"{turn.role}: {turn.content}" for turn in turns]
    return "Summary of earlier conversation:\n" + "\n".join(lines)


async def compact_session_history(
    store: SessionStore,
    session_id: str,
    *,
    settings: Settings | None = None,
    summarizer: SummarizerFn | None = None,
) -> bool:
    """Fold a session's aged-out messages into a summary once history is large enough.

    No-op unless both are true: the session has more stored messages than
    `settings.history_window_turns`, and the full stored history's estimated
    token count exceeds `settings.summarization_token_threshold`. When both
    hold, everything older than the sliding window (including any prior
    `role="summary"` message, so summaries roll forward instead of chaining)
    is passed to `summarizer` and replaced by a single new summary message.

    Args:
        store: A connected `SessionStore`.
        session_id: Session to compact.
        settings: Supplies `history_window_turns` and
            `summarization_token_threshold`. Defaults to `get_settings()`.
        summarizer: Async callable turning the aged-out messages into a
            summary string. Defaults to `default_summarizer`.

    Returns:
        `True` if compaction ran and the stored history was rewritten,
        `False` if the session was already within the window/threshold and
        nothing changed.
    """
    settings = settings or get_settings()
    summarizer = summarizer or default_summarizer

    turns = await store.get_turns(session_id)
    if len(turns) <= settings.history_window_turns:
        return False
    if estimate_turns_token_count(turns) <= settings.summarization_token_threshold:
        return False

    window = get_sliding_window(turns, settings.history_window_turns)
    older = turns[: len(turns) - len(window)]

    summary_text = await summarizer(older)
    summary_turn = ConversationTurn(
        session_id=session_id,
        turn_id="summary",
        role="summary",
        content=summary_text,
    )
    await store.replace_turns(session_id, [summary_turn, *window])
    return True
