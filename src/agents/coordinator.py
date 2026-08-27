"""TravelCoordinatorAgent: primary interaction, state maintenance, and
multi-stop itinerary synthesis.

Routed to `Settings.model_pro` (`gemini-3.1-pro`) per the project's
strategic model routing: multi-day constraint resolution, cross-city route
synthesis, and budget optimization need stronger reasoning than the fast
model is meant for. The coordinator holds the conversation with the
traveler directly and delegates to specialist sub-agents via LLM
delegation (see `references/adk-python.md` Sec 4, "LLM Delegation") rather
than calling their tools itself, so each specialist's dedicated
instruction/model stays in effect for its own sub-task. It also carries
`calculate_transit_route_estimate` directly, since sequencing stops and
estimating travel time/cost between them is itinerary-assembly work the
coordinator itself performs rather than delegating out.
"""

from __future__ import annotations

from google.adk.agents import Agent

from src.agents.attraction_search import create_attraction_search_agent
from src.agents.booking_specialist import create_booking_agent
from src.agents.constitution import COORDINATOR_MANDATE, build_instruction
from src.agents.weather_specialist import create_weather_specialist_agent
from src.config import get_settings
from src.tools.transit_tool import (
    calculate_transit_route_estimate as _calculate_transit_route_estimate,
)

AGENT_NAME = "TravelCoordinatorAgent"

AGENT_DESCRIPTION = (
    "Primary travel concierge. Plans multi-stop itineraries, tracks "
    "traveler preferences and budget, and coordinates specialist agents "
    "for weather, attractions, and bookings."
)


def calculate_transit_route_estimate(origin: str, destination: str, travel_mode: str) -> dict:
    """Estimate distance, duration, and cost for a single-mode trip between two stops.

    Args:
        origin: Free-text starting location, e.g. "Paris" or "Tokyo". Must
            resolve unambiguously.
        destination: Free-text ending location, e.g. "Paris" or "Tokyo".
            Must resolve unambiguously.
        travel_mode: One of "walking", "cycling", "driving", "transit", or
            "flight". Ground modes are only supported up to a realistic
            maximum distance for that mode; "flight" requires a minimum
            distance of 100 km.

    Returns:
        On success, a dict with `status="success"` and a `data` key holding
        the resolved origin/destination, distance, estimated duration, and
        estimated cost. On failure, a dict with `status="error"`,
        `error_code`, `message`, and `recovery_instruction` (e.g. switch
        travel_mode when the distance doesn't suit the requested mode).
    """
    result = _calculate_transit_route_estimate(origin, destination, travel_mode)
    return result.model_dump(mode="json")


def create_travel_coordinator_agent(model: str | None = None) -> Agent:
    """Build a fresh `TravelCoordinatorAgent` with all specialists attached.

    A factory rather than a module-level singleton: ADK's sub-agent wiring
    assigns each sub-agent a single parent, so a fresh coordinator needs
    fresh specialist instances (see the factory-function pattern in
    `references/adk-python.md`).

    Args:
        model: Model id override. Defaults to `Settings.model_pro`
            (`gemini-3.1-pro`).

    Returns:
        A configured root `Agent` with `WeatherSpecialistAgent`,
        `AttractionSearchAgent`, and `BookingAgent` as delegable
        sub-agents, and `calculate_transit_route_estimate` as its own tool.
    """
    settings = get_settings()
    return Agent(
        name=AGENT_NAME,
        model=model or settings.model_pro,
        description=AGENT_DESCRIPTION,
        instruction=build_instruction(COORDINATOR_MANDATE),
        tools=[calculate_transit_route_estimate],
        sub_agents=[
            create_weather_specialist_agent(),
            create_attraction_search_agent(),
            create_booking_agent(),
        ],
    )
