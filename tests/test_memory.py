"""Tests for src/memory: SQLite session persistence, compaction, and async extraction."""

import pytest

from src.config import Settings
from src.memory.async_memory import AsyncMemoryWorker, default_preference_extractor
from src.memory.compaction import (
    compact_session_history,
    estimate_token_count,
    estimate_turns_token_count,
    get_sliding_window,
)
from src.memory.session_store import (
    ConversationTurn,
    ItineraryDraft,
    SessionStore,
    TravelerPreferences,
)


@pytest.fixture
async def store():
    async with SessionStore(database_path=":memory:") as store:
        yield store


# --- conversation turns ------------------------------------------------------


async def test_append_and_get_turns_round_trips_in_chronological_order(store):
    await store.append_turn(
        ConversationTurn(session_id="s1", turn_id="t1", role="user", content="Plan a trip to Kyoto")
    )
    await store.append_turn(
        ConversationTurn(
            session_id="s1",
            turn_id="t1",
            role="agent",
            agent_name="TravelCoordinatorAgent",
            content="Sure, what dates work?",
        )
    )

    turns = await store.get_turns("s1")

    assert [t.role for t in turns] == ["user", "agent"]
    assert turns[0].content == "Plan a trip to Kyoto"
    assert turns[1].agent_name == "TravelCoordinatorAgent"


async def test_get_turns_is_scoped_to_session_id(store):
    await store.append_turn(ConversationTurn(session_id="s1", turn_id="t1", role="user", content="a"))
    await store.append_turn(ConversationTurn(session_id="s2", turn_id="t1", role="user", content="b"))

    assert [t.content for t in await store.get_turns("s1")] == ["a"]
    assert [t.content for t in await store.get_turns("s2")] == ["b"]


async def test_get_turns_limit_returns_most_recent_in_chronological_order(store):
    for i in range(5):
        await store.append_turn(
            ConversationTurn(session_id="s1", turn_id=f"t{i}", role="user", content=str(i))
        )

    turns = await store.get_turns("s1", limit=2)

    assert [t.content for t in turns] == ["3", "4"]


async def test_append_turn_scrubs_pii_before_persisting(store):
    await store.append_turn(
        ConversationTurn(
            session_id="s1", turn_id="t1", role="user", content="Reach me at jane@example.com"
        )
    )

    turns = await store.get_turns("s1")

    assert turns[0].content == "Reach me at [REDACTED_EMAIL]"


async def test_replace_turns_collapses_history_atomically(store):
    for i in range(3):
        await store.append_turn(
            ConversationTurn(session_id="s1", turn_id=f"t{i}", role="user", content=str(i))
        )

    await store.replace_turns(
        "s1",
        [
            ConversationTurn(session_id="s1", turn_id="summary", role="summary", content="recap"),
            ConversationTurn(session_id="s1", turn_id="t2", role="user", content="2"),
        ],
    )

    turns = await store.get_turns("s1")
    assert [(t.role, t.content) for t in turns] == [("summary", "recap"), ("user", "2")]


async def test_operations_before_connect_raise_runtime_error():
    unconnected = SessionStore(database_path=":memory:")
    with pytest.raises(RuntimeError):
        await unconnected.get_turns("s1")


# --- traveler preferences ----------------------------------------------------


async def test_upsert_and_get_traveler_preferences_round_trips(store):
    await store.upsert_traveler_preferences(
        TravelerPreferences(
            session_id="s1",
            dietary_restrictions=["vegetarian"],
            preferred_airlines=["ANA"],
            pacing="relaxed",
            budget_ceiling=2500.0,
            notes="prefers window seats",
        )
    )

    prefs = await store.get_traveler_preferences("s1")

    assert prefs.dietary_restrictions == ["vegetarian"]
    assert prefs.preferred_airlines == ["ANA"]
    assert prefs.budget_ceiling == 2500.0


async def test_get_traveler_preferences_returns_none_when_absent(store):
    assert await store.get_traveler_preferences("missing") is None


async def test_upsert_traveler_preferences_overwrites_prior_profile(store):
    await store.upsert_traveler_preferences(TravelerPreferences(session_id="s1", pacing="relaxed"))
    await store.upsert_traveler_preferences(TravelerPreferences(session_id="s1", pacing="packed"))

    prefs = await store.get_traveler_preferences("s1")

    assert prefs.pacing == "packed"


async def test_upsert_traveler_preferences_scrubs_pii_in_notes(store):
    await store.upsert_traveler_preferences(
        TravelerPreferences(session_id="s1", notes="Emergency contact: jane@example.com")
    )

    prefs = await store.get_traveler_preferences("s1")

    assert prefs.notes == "Emergency contact: [REDACTED_EMAIL]"


# --- itinerary drafts ---------------------------------------------------------


async def test_save_and_get_itinerary_drafts_round_trips(store):
    await store.save_itinerary_draft(
        ItineraryDraft(
            session_id="s1",
            draft_id="d1",
            title="5-day Kyoto trip",
            content={"days": [{"date": "2026-09-01", "activities": ["temple visit"]}]},
        )
    )

    drafts = await store.get_itinerary_drafts("s1")

    assert len(drafts) == 1
    assert drafts[0].title == "5-day Kyoto trip"
    assert drafts[0].status == "draft"
    assert drafts[0].content["days"][0]["activities"] == ["temple visit"]


async def test_save_itinerary_draft_upserts_on_same_draft_id(store):
    await store.save_itinerary_draft(
        ItineraryDraft(session_id="s1", draft_id="d1", title="v1", content={"total_cost": 100})
    )
    await store.save_itinerary_draft(
        ItineraryDraft(
            session_id="s1", draft_id="d1", title="v2", content={"total_cost": 150}, status="confirmed"
        )
    )

    drafts = await store.get_itinerary_drafts("s1")

    assert len(drafts) == 1
    assert drafts[0].title == "v2"
    assert drafts[0].status == "confirmed"
    assert drafts[0].content["total_cost"] == 150


async def test_save_itinerary_draft_scrubs_pii_in_content(store):
    await store.save_itinerary_draft(
        ItineraryDraft(
            session_id="s1",
            draft_id="d1",
            title="trip",
            content={"contact_email": "jane@example.com"},
        )
    )

    drafts = await store.get_itinerary_drafts("s1")

    assert drafts[0].content["contact_email"] == "[REDACTED_EMAIL]"


# --- compaction ---------------------------------------------------------------


def _turn(i: int, content: str = "") -> ConversationTurn:
    return ConversationTurn(
        session_id="s1", turn_id=f"t{i}", role="user", content=content or f"message {i}"
    )


def test_estimate_token_count_uses_length_heuristic():
    assert estimate_token_count("") == 0
    assert estimate_token_count("ab") == 1
    assert estimate_token_count("a" * 40) == 10


def test_estimate_turns_token_count_sums_per_turn_estimates():
    turns = [_turn(0, "aaaa"), _turn(1, "bbbbbbbb")]
    assert estimate_turns_token_count(turns) == estimate_token_count("aaaa") + estimate_token_count(
        "bbbbbbbb"
    )


def test_get_sliding_window_returns_trailing_slice():
    turns = [_turn(i) for i in range(5)]
    window = get_sliding_window(turns, 2)
    assert [t.turn_id for t in window] == ["t3", "t4"]


def test_get_sliding_window_handles_non_positive_and_oversized_window():
    turns = [_turn(i) for i in range(3)]
    assert get_sliding_window(turns, 0) == []
    assert get_sliding_window(turns, 10) == turns


async def test_compact_session_history_noop_when_within_window(store):
    for i in range(3):
        await store.append_turn(_turn(i))

    settings = Settings(history_window_turns=20, summarization_token_threshold=4000)
    ran = await compact_session_history(store, "s1", settings=settings)

    assert ran is False
    assert len(await store.get_turns("s1")) == 3


async def test_compact_session_history_noop_when_under_token_threshold(store):
    for i in range(10):
        await store.append_turn(_turn(i))

    settings = Settings(history_window_turns=2, summarization_token_threshold=4000)
    ran = await compact_session_history(store, "s1", settings=settings)

    assert ran is False
    assert len(await store.get_turns("s1")) == 10


async def test_compact_session_history_folds_older_turns_into_summary(store):
    for i in range(10):
        await store.append_turn(_turn(i, content="x" * 40))

    settings = Settings(history_window_turns=3, summarization_token_threshold=10)

    async def fake_summarizer(turns):
        return f"recap of {len(turns)} turns"

    ran = await compact_session_history(store, "s1", settings=settings, summarizer=fake_summarizer)

    assert ran is True
    turns = await store.get_turns("s1")
    assert len(turns) == 4
    assert turns[0].role == "summary"
    assert turns[0].content == "recap of 7 turns"
    assert [t.turn_id for t in turns[1:]] == ["t7", "t8", "t9"]


async def test_compact_session_history_rolls_prior_summary_forward(store):
    await store.append_turn(
        ConversationTurn(session_id="s1", turn_id="summary", role="summary", content="old recap")
    )
    for i in range(5):
        await store.append_turn(_turn(i, content="y" * 40))

    settings = Settings(history_window_turns=2, summarization_token_threshold=10)
    seen_turns = []

    async def capturing_summarizer(turns):
        seen_turns.extend(turns)
        return "new recap"

    await compact_session_history(store, "s1", settings=settings, summarizer=capturing_summarizer)

    assert seen_turns[0].role == "summary"
    assert seen_turns[0].content == "old recap"
    turns = await store.get_turns("s1")
    assert turns[0].content == "new recap"


# --- async memory worker -------------------------------------------------------


async def test_default_preference_extractor_detects_dietary_and_pacing_keywords():
    turns = [
        ConversationTurn(
            session_id="s1", turn_id="t1", role="user", content="I'm vegetarian and prefer a relaxed pace"
        )
    ]

    updated = await default_preference_extractor(turns, existing=None)

    assert updated.session_id == "s1"
    assert updated.dietary_restrictions == ["vegetarian"]
    assert updated.pacing == "relaxed"


async def test_default_preference_extractor_merges_into_existing_profile():
    existing = TravelerPreferences(session_id="s1", dietary_restrictions=["vegan"], pacing="packed")
    turns = [ConversationTurn(session_id="s1", turn_id="t1", role="user", content="also nut-free please")]

    updated = await default_preference_extractor(turns, existing)

    assert updated.dietary_restrictions == ["nut-free", "vegan"]
    assert updated.pacing == "packed"


async def test_extract_preferences_in_background_persists_without_blocking_caller(store):
    worker = AsyncMemoryWorker(store)
    turns = [ConversationTurn(session_id="s1", turn_id="t1", role="user", content="I'm vegan")]

    task = worker.extract_preferences_in_background("s1", turns)

    assert not task.done()
    assert await store.get_traveler_preferences("s1") is None

    await worker.wait_for_pending()

    prefs = await store.get_traveler_preferences("s1")
    assert prefs.dietary_restrictions == ["vegan"]


async def test_worker_uses_injected_extractor(store):
    async def custom_extractor(turns, existing):
        return TravelerPreferences(session_id="s1", notes="custom-extracted")

    worker = AsyncMemoryWorker(store, extractor=custom_extractor)
    worker.extract_preferences_in_background(
        "s1", [ConversationTurn(session_id="s1", turn_id="t1", role="user", content="hi")]
    )
    await worker.wait_for_pending()

    prefs = await store.get_traveler_preferences("s1")
    assert prefs.notes == "custom-extracted"


async def test_worker_swallows_extractor_failures_without_raising(store):
    async def failing_extractor(turns, existing):
        raise ValueError("boom")

    worker = AsyncMemoryWorker(store, extractor=failing_extractor)
    worker.extract_preferences_in_background(
        "s1", [ConversationTurn(session_id="s1", turn_id="t1", role="user", content="hi")]
    )

    await worker.wait_for_pending()

    assert await store.get_traveler_preferences("s1") is None


async def test_wait_for_pending_is_a_noop_with_no_scheduled_tasks(store):
    worker = AsyncMemoryWorker(store)
    await worker.wait_for_pending()
