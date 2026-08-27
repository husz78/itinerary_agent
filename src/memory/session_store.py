"""Persistent local session store backed by SQLite.

Backs multi-turn conversations, traveler preference profiles, and itinerary
drafts with a local SQLite database (`Settings.database_path`, defaulting to
`data/travel_agent.db`) so a session survives process restarts instead of
living only in memory. Access is async (`aiosqlite`) so callers on the
agent's request path never block on disk I/O.

Every string field written to any table is passed through
`src.observability.pii_scrubber.scrub_value` first, matching the same
choke point used by the structured logger and tracer, so passports, credit
cards, emails, and phone numbers never land in the database in the clear.

`SessionStore` is a plain repository object rather than a process-wide
singleton: `aiosqlite` connections are bound to the event loop that created
them, and a singleton would break across the fresh event loops pytest-asyncio
creates per test. Callers construct one `SessionStore` per event loop (an
agent process, a CLI run, or a single test) and use it as an async context
manager.

Example:
    >>> async with SessionStore(database_path=":memory:") as store:
    ...     await store.append_turn(
    ...         ConversationTurn(session_id="s1", turn_id="t1", role="user", content="Hi")
    ...     )
    ...     turns = await store.get_turns("s1")
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Literal

import aiosqlite
from pydantic import BaseModel, Field

from src.config import Settings, get_settings
from src.observability.pii_scrubber import scrub_value

TurnRole = Literal["user", "agent", "tool", "summary"]


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string for `created_at`/`updated_at` fields."""
    return datetime.datetime.now(datetime.UTC).isoformat()


class ConversationTurn(BaseModel):
    """One stored message within a persisted conversation session.

    Attributes:
        session_id: Identifier of the persistent conversation session this
            message belongs to.
        turn_id: Identifier of the conversation turn this message is part
            of. A single turn may have several messages (e.g. a user
            message, an agent handoff, and a final agent reply) sharing the
            same `turn_id`.
        role: Who produced this message: `"user"`, `"agent"`, `"tool"`
            (a tool call/result summary), or `"summary"` (a compacted
            summary of older turns produced by history compaction).
        agent_name: Name of the specialist or coordinator agent that
            produced this message, e.g. `"BookingAgent"`. `None` for
            `role="user"`.
        content: The message text. PII-scrubbed before being persisted.
        created_at: ISO 8601 UTC timestamp this row was written.
    """

    session_id: str = Field(..., description="Persistent conversation session identifier.")
    turn_id: str = Field(..., description="Conversation turn this message belongs to.")
    role: TurnRole = Field(..., description="Message producer: user, agent, tool, or summary.")
    agent_name: str | None = Field(
        default=None, description="Name of the producing agent, if any."
    )
    content: str = Field(..., description="Message text, scrubbed of PII before persistence.")
    created_at: str = Field(
        default_factory=_utc_now_iso, description="ISO 8601 UTC timestamp of persistence."
    )


class TravelerPreferences(BaseModel):
    """Long-term traveler preference profile for a session.

    Attributes:
        session_id: Persistent conversation session this profile belongs to.
        dietary_restrictions: e.g. `["vegetarian", "nut-free"]`.
        preferred_airlines: e.g. `["Delta", "ANA"]`.
        pacing: Preferred itinerary pace, e.g. `"relaxed"` or `"packed"`.
        budget_ceiling: Traveler's stated total budget ceiling, if known.
        notes: Free-text notes on any other extracted preference.
        updated_at: ISO 8601 UTC timestamp of the last update.
    """

    session_id: str = Field(..., description="Persistent conversation session identifier.")
    dietary_restrictions: list[str] = Field(default_factory=list)
    preferred_airlines: list[str] = Field(default_factory=list)
    pacing: str | None = Field(default=None, description="Preferred itinerary pace.")
    budget_ceiling: float | None = Field(default=None, description="Total budget ceiling.")
    notes: str = Field(default="", description="Free-text notes on other preferences.")
    updated_at: str = Field(
        default_factory=_utc_now_iso, description="ISO 8601 UTC timestamp of last update."
    )


class ItineraryDraft(BaseModel):
    """A draft (or confirmed) itinerary snapshot for a session.

    Attributes:
        session_id: Persistent conversation session this draft belongs to.
        draft_id: Identifier for this draft, unique within the session.
        title: Short human-readable label, e.g. `"5-day Tokyo trip"`.
        content: Arbitrary structured itinerary payload (day-by-day plan,
            staged bookings, cost totals). Stored as JSON.
        status: `"draft"` while still being assembled/edited, `"confirmed"`
            once the traveler has authorized it.
        updated_at: ISO 8601 UTC timestamp of the last update.
    """

    session_id: str = Field(..., description="Persistent conversation session identifier.")
    draft_id: str = Field(..., description="Draft identifier, unique within the session.")
    title: str = Field(..., description="Short human-readable label for the draft.")
    content: dict[str, Any] = Field(
        default_factory=dict, description="Structured itinerary payload."
    )
    status: Literal["draft", "confirmed"] = Field(
        default="draft", description="Lifecycle status of the draft."
    )
    updated_at: str = Field(
        default_factory=_utc_now_iso, description="ISO 8601 UTC timestamp of last update."
    )


_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversation_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    role TEXT NOT NULL,
    agent_name TEXT,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversation_turns_session ON conversation_turns(session_id, id);

CREATE TABLE IF NOT EXISTS traveler_preferences (
    session_id TEXT PRIMARY KEY,
    dietary_restrictions TEXT NOT NULL,
    preferred_airlines TEXT NOT NULL,
    pacing TEXT,
    budget_ceiling REAL,
    notes TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS itinerary_drafts (
    session_id TEXT NOT NULL,
    draft_id TEXT NOT NULL,
    title TEXT NOT NULL,
    content_json TEXT NOT NULL,
    status TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (session_id, draft_id)
);
"""


class SessionStore:
    """Async repository for local SQLite-backed session persistence.

    Not a singleton: `aiosqlite` connections are bound to the event loop
    that opened them, so construct one instance per event loop (an agent
    process, a CLI run, or a test) and use it as an async context manager.
    """

    def __init__(self, database_path: str | None = None, settings: Settings | None = None) -> None:
        """Configure (without yet opening) a SQLite-backed session store.

        Args:
            database_path: Path to the SQLite database file, or `":memory:"`
                for an ephemeral in-process database (used by tests).
                Defaults to `settings.database_path`.
            settings: Application settings supplying the default database
                path. Defaults to `get_settings()`.
        """
        settings = settings or get_settings()
        self._database_path = database_path if database_path is not None else settings.database_path
        self._connection: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """Open the SQLite connection and create tables if they don't exist yet.

        Safe to call more than once; a second call is a no-op while already
        connected.
        """
        if self._connection is not None:
            return
        if self._database_path != ":memory:":
            Path(self._database_path).parent.mkdir(parents=True, exist_ok=True)
        connection = await aiosqlite.connect(self._database_path)
        connection.row_factory = aiosqlite.Row
        await connection.executescript(_SCHEMA)
        await connection.commit()
        self._connection = connection

    async def close(self) -> None:
        """Close the SQLite connection, if open."""
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    async def __aenter__(self) -> "SessionStore":
        await self.connect()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    def _require_connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError(
                "SessionStore is not connected; call connect() or use it as "
                "an async context manager before reading or writing."
            )
        return self._connection

    async def _insert_turn(self, connection: aiosqlite.Connection, turn: ConversationTurn) -> None:
        scrubbed = scrub_value(turn.model_dump())
        await connection.execute(
            """
            INSERT INTO conversation_turns
                (session_id, turn_id, role, agent_name, content, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                scrubbed["session_id"],
                scrubbed["turn_id"],
                scrubbed["role"],
                scrubbed["agent_name"],
                scrubbed["content"],
                scrubbed["created_at"],
            ),
        )

    async def append_turn(self, turn: ConversationTurn) -> None:
        """Persist one conversation message, scrubbing PII first.

        Args:
            turn: The message to store.

        Returns:
            None.
        """
        connection = self._require_connection()
        await self._insert_turn(connection, turn)
        await connection.commit()

    async def get_turns(self, session_id: str, limit: int | None = None) -> list[ConversationTurn]:
        """Return stored messages for a session in chronological order.

        Args:
            session_id: Session to read messages for.
            limit: If given, return only the most recent `limit` messages
                (still chronologically ordered), matching the sliding-window
                semantics `memory/compaction.py` needs. `None` returns the
                full stored history.

        Returns:
            A list of `ConversationTurn`, oldest first.
        """
        connection = self._require_connection()
        if limit is None:
            cursor = await connection.execute(
                """
                SELECT session_id, turn_id, role, agent_name, content, created_at
                FROM conversation_turns WHERE session_id = ? ORDER BY id ASC
                """,
                (session_id,),
            )
        else:
            cursor = await connection.execute(
                """
                SELECT session_id, turn_id, role, agent_name, content, created_at FROM (
                    SELECT id, session_id, turn_id, role, agent_name, content, created_at
                    FROM conversation_turns WHERE session_id = ? ORDER BY id DESC LIMIT ?
                ) recent ORDER BY id ASC
                """,
                (session_id, limit),
            )
        rows = await cursor.fetchall()
        return [ConversationTurn(**dict(row)) for row in rows]

    async def replace_turns(self, session_id: str, turns: list[ConversationTurn]) -> None:
        """Atomically replace all stored messages for a session.

        Used by history compaction to collapse older messages into a single
        `role="summary"` message while keeping the most recent sliding
        window verbatim.

        Args:
            session_id: Session whose stored messages should be replaced.
            turns: The full new message list to store, in chronological
                order.

        Returns:
            None.
        """
        connection = self._require_connection()
        await connection.execute(
            "DELETE FROM conversation_turns WHERE session_id = ?", (session_id,)
        )
        for turn in turns:
            await self._insert_turn(connection, turn)
        await connection.commit()

    async def upsert_traveler_preferences(self, preferences: TravelerPreferences) -> None:
        """Insert or update the traveler preference profile for a session.

        Args:
            preferences: The full preference profile to store, replacing any
                prior profile for `preferences.session_id`.

        Returns:
            None.
        """
        connection = self._require_connection()
        scrubbed = scrub_value(preferences.model_dump())
        await connection.execute(
            """
            INSERT INTO traveler_preferences
                (session_id, dietary_restrictions, preferred_airlines, pacing,
                 budget_ceiling, notes, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                dietary_restrictions = excluded.dietary_restrictions,
                preferred_airlines = excluded.preferred_airlines,
                pacing = excluded.pacing,
                budget_ceiling = excluded.budget_ceiling,
                notes = excluded.notes,
                updated_at = excluded.updated_at
            """,
            (
                scrubbed["session_id"],
                json.dumps(scrubbed["dietary_restrictions"]),
                json.dumps(scrubbed["preferred_airlines"]),
                scrubbed["pacing"],
                scrubbed["budget_ceiling"],
                scrubbed["notes"],
                scrubbed["updated_at"],
            ),
        )
        await connection.commit()

    async def get_traveler_preferences(self, session_id: str) -> TravelerPreferences | None:
        """Return the stored traveler preference profile for a session, if any.

        Args:
            session_id: Session to look up.

        Returns:
            The stored `TravelerPreferences`, or `None` if no profile has
            been saved for this session yet.
        """
        connection = self._require_connection()
        cursor = await connection.execute(
            """
            SELECT session_id, dietary_restrictions, preferred_airlines, pacing,
                   budget_ceiling, notes, updated_at
            FROM traveler_preferences WHERE session_id = ?
            """,
            (session_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        data = dict(row)
        data["dietary_restrictions"] = json.loads(data["dietary_restrictions"])
        data["preferred_airlines"] = json.loads(data["preferred_airlines"])
        return TravelerPreferences(**data)

    async def save_itinerary_draft(self, draft: ItineraryDraft) -> None:
        """Insert or update an itinerary draft.

        Args:
            draft: The draft to store, replacing any prior draft with the
                same `(session_id, draft_id)`.

        Returns:
            None.
        """
        connection = self._require_connection()
        scrubbed = scrub_value(draft.model_dump(exclude={"content"}))
        scrubbed_content = scrub_value(draft.content)
        await connection.execute(
            """
            INSERT INTO itinerary_drafts
                (session_id, draft_id, title, content_json, status, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, draft_id) DO UPDATE SET
                title = excluded.title,
                content_json = excluded.content_json,
                status = excluded.status,
                updated_at = excluded.updated_at
            """,
            (
                scrubbed["session_id"],
                scrubbed["draft_id"],
                scrubbed["title"],
                json.dumps(scrubbed_content),
                scrubbed["status"],
                scrubbed["updated_at"],
            ),
        )
        await connection.commit()

    async def get_itinerary_drafts(self, session_id: str) -> list[ItineraryDraft]:
        """Return all itinerary drafts stored for a session, oldest-updated first.

        Args:
            session_id: Session to look up.

        Returns:
            A list of `ItineraryDraft`.
        """
        connection = self._require_connection()
        cursor = await connection.execute(
            """
            SELECT session_id, draft_id, title, content_json, status, updated_at
            FROM itinerary_drafts WHERE session_id = ? ORDER BY updated_at ASC
            """,
            (session_id,),
        )
        rows = await cursor.fetchall()
        drafts = []
        for row in rows:
            data = dict(row)
            data["content"] = json.loads(data.pop("content_json"))
            drafts.append(ItineraryDraft(**data))
        return drafts
