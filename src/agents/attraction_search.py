"""AttractionSearchAgent: places, events, opening hours, local search.

Routed to `Settings.model_fast` (`gemini-3.5-flash`) per the project's
strategic model routing: filtering a static catalog by category, budget
tier, and duration is high-throughput lookup work, not multi-step
reasoning, so the fast model is sufficient.
"""

from __future__ import annotations

from google.adk.agents import Agent

from src.agents.constitution import ATTRACTION_SEARCH_MANDATE, build_instruction
from src.config import get_settings
from src.tools.attraction_tool import (
    search_attractions_and_activities as _search_attractions_and_activities,
)

AGENT_NAME = "AttractionSearchAgent"

AGENT_DESCRIPTION = (
    "Searches for attractions and activities in a city by category, budget "
    "tier, and available duration. Delegate here for any question about "
    "what to see or do somewhere, opening hours, or ratings."
)


def search_attractions_and_activities(
    city: str,
    category: str | None,
    budget_tier: str,
    duration_hours: float,
) -> dict:
    """Search for attractions and activities in a city matching given filters.

    Args:
        city: Free-text city name, e.g. "Paris" or "Tokyo". Must resolve
            unambiguously; if ambiguous (e.g. "Paris" matches France,
            Texas, and Ontario), the result asks you to disambiguate
            instead of guessing.
        category: Optional category filter, one of "museum", "outdoor",
            "food", "nightlife", "landmark", "shopping", "family", "art",
            "history". Pass `None` to search all categories.
        budget_tier: Maximum price tier the traveler is willing to pay, one
            of "free", "budget", "moderate", "luxury" (each tier includes
            all cheaper tiers).
        duration_hours: Maximum time in hours available for a single
            activity. Must be greater than 0.

    Returns:
        On success, a dict with `status="success"` and a `data` key holding
        `resolved_city` and matching `results` sorted by rating descending.
        On failure, a dict with `status="error"`, `error_code`, `message`,
        and `recovery_instruction` describing exactly how to recover (e.g.
        disambiguate the city, broaden the search).
    """
    result = _search_attractions_and_activities(city, category, budget_tier, duration_hours)
    return result.model_dump(mode="json")


def create_attraction_search_agent(model: str | None = None) -> Agent:
    """Build a fresh `AttractionSearchAgent` instance.

    A factory rather than a module-level singleton: ADK assigns each
    sub-agent a single parent, so every coordinator built via
    `src.agents.coordinator.create_travel_coordinator_agent` needs its own
    specialist instances.

    Args:
        model: Model id override. Defaults to `Settings.model_fast`
            (`gemini-3.5-flash`).

    Returns:
        A configured `Agent` bound to `search_attractions_and_activities`.
    """
    settings = get_settings()
    return Agent(
        name=AGENT_NAME,
        model=model or settings.model_fast,
        description=AGENT_DESCRIPTION,
        instruction=build_instruction(ATTRACTION_SEARCH_MANDATE),
        tools=[search_attractions_and_activities],
    )
