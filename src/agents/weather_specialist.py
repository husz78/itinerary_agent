"""WeatherSpecialistAgent: fast weather retrieval and forecast processing.

Routed to `Settings.model_fast` (`gemini-3.5-flash`) per the project's
strategic model routing: a weather lookup is a single-tool, low-latency
sub-task with no multi-step reasoning, so the fast model keeps this
specialist's turnaround (and the coordinator's overall latency) low.
"""

from __future__ import annotations

from google.adk.agents import Agent

from src.agents.constitution import WEATHER_SPECIALIST_MANDATE, build_instruction
from src.config import get_settings
from src.tools.weather_tool import (
    fetch_destination_weather_forecast as _fetch_destination_weather_forecast,
)

AGENT_NAME = "WeatherSpecialistAgent"

AGENT_DESCRIPTION = (
    "Fetches destination weather forecasts. Delegate here for any question "
    "about weather, temperature, rain, or forecast at a specific location "
    "and date range."
)


def fetch_destination_weather_forecast(location: str, start_date: str, end_date: str) -> dict:
    """Fetch a daily weather forecast for a destination over a date range.

    Args:
        location: Free-text destination name, e.g. "Paris" or "Tokyo". Must
            resolve unambiguously; if ambiguous (e.g. "Paris" matches
            France, Texas, and Ontario), the result asks you to
            disambiguate instead of guessing.
        start_date: Inclusive start of the forecast window, ISO 8601
            (YYYY-MM-DD).
        end_date: Inclusive end of the forecast window, ISO 8601
            (YYYY-MM-DD). Must not precede start_date, and the window must
            not exceed 14 days.

    Returns:
        On success, a dict with `status="success"` and a `data` key holding
        `resolved_location` and one `daily_forecasts` entry per day in
        range. On failure, a dict with `status="error"`, `error_code`,
        `message`, and `recovery_instruction` describing exactly how to
        recover (e.g. disambiguate the location, fix the date range).
    """
    result = _fetch_destination_weather_forecast(location, start_date, end_date)
    return result.model_dump(mode="json")


def create_weather_specialist_agent(model: str | None = None) -> Agent:
    """Build a fresh `WeatherSpecialistAgent` instance.

    A factory rather than a module-level singleton: ADK assigns each
    sub-agent a single parent, so every coordinator built via
    `src.agents.coordinator.create_travel_coordinator_agent` needs its own
    specialist instances.

    Args:
        model: Model id override. Defaults to `Settings.model_fast`
            (`gemini-3.5-flash`).

    Returns:
        A configured `Agent` bound to `fetch_destination_weather_forecast`.
    """
    settings = get_settings()
    return Agent(
        name=AGENT_NAME,
        model=model or settings.model_fast,
        description=AGENT_DESCRIPTION,
        instruction=build_instruction(WEATHER_SPECIALIST_MANDATE),
        tools=[fetch_destination_weather_forecast],
    )
