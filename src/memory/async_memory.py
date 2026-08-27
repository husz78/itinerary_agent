"""Asynchronous background memory worker.

Wraps `asyncio.create_task` so traveler profile-preference extraction runs
detached from the conversational response path: after handling a turn, the
coordinator schedules extraction via `AsyncMemoryWorker.extract_preferences_in_background`
and replies to the user immediately, while extraction and the resulting
`SessionStore.upsert_traveler_preferences` write happen in the background.
This satisfies the project's async memory operations requirement — response
latency is never blocked on preference extraction.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from src.memory.session_store import ConversationTurn, SessionStore, TravelerPreferences
from src.observability.logger import get_logger

ExtractorFn = Callable[
    [list[ConversationTurn], TravelerPreferences | None], Awaitable[TravelerPreferences]
]

_DIETARY_KEYWORDS = (
    "vegetarian",
    "vegan",
    "gluten-free",
    "nut-free",
    "halal",
    "kosher",
    "pescatarian",
)
_PACING_KEYWORDS = {
    "relaxed": "relaxed",
    "slow-paced": "relaxed",
    "packed": "packed",
    "fast-paced": "packed",
}


async def default_preference_extractor(
    turns: list[ConversationTurn], existing: TravelerPreferences | None
) -> TravelerPreferences:
    """Heuristic, non-LLM fallback preference extractor.

    Scans `turns` for a small fixed set of dietary/pacing keywords and merges
    any matches into `existing` (or a fresh profile). Used for local testing
    and as a safe fallback; production callers should supply an LLM-backed
    `ExtractorFn` routed to `gemini-3.5-flash` for genuine preference
    extraction (airlines, budget figures, free-text notes) instead of
    keyword matching.

    Args:
        turns: Newly observed messages to scan.
        existing: The traveler's current stored preferences, if any.

    Returns:
        A `TravelerPreferences` merging any newly detected keywords into
        `existing` (or a new profile scoped to the turns' `session_id` if
        `existing` is `None`).
    """
    session_id = turns[0].session_id if turns else (existing.session_id if existing else "")
    base = existing or TravelerPreferences(session_id=session_id)
    dietary = set(base.dietary_restrictions)
    pacing = base.pacing

    for turn in turns:
        lowered = turn.content.lower()
        for keyword in _DIETARY_KEYWORDS:
            if keyword in lowered:
                dietary.add(keyword)
        for keyword, value in _PACING_KEYWORDS.items():
            if keyword in lowered:
                pacing = value

    return base.model_copy(update={"dietary_restrictions": sorted(dietary), "pacing": pacing})


class AsyncMemoryWorker:
    """Schedules non-blocking background preference-extraction tasks.

    Keeps a reference to every scheduled `asyncio.Task` so it isn't
    garbage-collected mid-flight, and offers `wait_for_pending` so tests and
    graceful shutdown can deterministically await everything in flight.
    """

    def __init__(self, store: SessionStore, extractor: ExtractorFn | None = None) -> None:
        """Configure a background memory worker.

        Args:
            store: A connected `SessionStore` used to read the current
                profile and persist the extracted update.
            extractor: Async callable producing updated preferences from new
                turns and the existing profile. Defaults to
                `default_preference_extractor`.
        """
        self._store = store
        self._extractor = extractor or default_preference_extractor
        self._pending: set[asyncio.Task] = set()
        self._logger = get_logger("async_memory")

    def extract_preferences_in_background(
        self, session_id: str, turns: list[ConversationTurn]
    ) -> asyncio.Task:
        """Schedule background preference extraction/merge for `turns` and return immediately.

        Args:
            session_id: Session the new turns belong to.
            turns: Newly observed messages to extract preferences from
                (typically the current turn's user and agent messages).

        Returns:
            The scheduled `asyncio.Task`. Callers on the response path do
            not need to await it; the task persists its result via
            `SessionStore.upsert_traveler_preferences` when it completes.
        """
        task = asyncio.create_task(self._run(session_id, turns))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)
        return task

    async def _run(self, session_id: str, turns: list[ConversationTurn]) -> None:
        try:
            existing = await self._store.get_traveler_preferences(session_id)
            updated = await self._extractor(turns, existing)
            await self._store.upsert_traveler_preferences(updated)
        except Exception as exc:
            self._logger.error(
                "preference_extraction_failed", session_id=session_id, error=str(exc)
            )

    async def wait_for_pending(self) -> None:
        """Await every currently in-flight background extraction task.

        Intended for tests (to make background work deterministic) and
        graceful process shutdown. Safe to call with no pending tasks.
        """
        if self._pending:
            await asyncio.gather(*list(self._pending), return_exceptions=True)
