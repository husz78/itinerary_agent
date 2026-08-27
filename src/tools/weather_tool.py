"""Weather forecast tool: `fetch_destination_weather_forecast`.

Provides deterministic, locally computed weather forecasts for a destination
and date range. There is no outbound network call — a small local location
registry stands in for a geocoding lookup, and forecasts are derived from a
stable hash of `(location, date)` rather than `random`, so results are
reproducible across runs (required for deterministic tests and evals). A
real deployment would swap `_lookup_location` and `_deterministic_daily_forecast`
for calls to a geocoding/weather API; the public function signature and the
`ToolResultEnvelope` / `ToolErrorEnvelope` contract would stay unchanged.
"""

from __future__ import annotations

import hashlib
from datetime import date, timedelta

from pydantic import BaseModel, Field

from src.tools.base import (
    ToolErrorCode,
    ToolErrorEnvelope,
    ToolResultEnvelope,
    make_error,
    make_success,
)

MAX_FORECAST_HORIZON_DAYS = 14

# Local stand-in for a geocoding service. Maps a lowercased free-text query to
# every known (display_name, country) match; more than one match is ambiguous.
_LOCATION_REGISTRY: dict[str, list[tuple[str, str]]] = {
    "paris": [("Paris", "France"), ("Paris", "Texas, USA"), ("Paris", "Ontario, Canada")],
    "london": [("London", "United Kingdom"), ("London", "Ontario, Canada")],
    "springfield": [
        ("Springfield", "Illinois, USA"),
        ("Springfield", "Missouri, USA"),
        ("Springfield", "Massachusetts, USA"),
    ],
    "tokyo": [("Tokyo", "Japan")],
    "rome": [("Rome", "Italy")],
    "barcelona": [("Barcelona", "Spain")],
    "new york": [("New York City", "USA")],
    "san francisco": [("San Francisco", "USA")],
}

_CONDITIONS = ["sunny", "partly_cloudy", "cloudy", "light_rain", "rain", "thunderstorms"]


class DailyForecast(BaseModel):
    """A single day's forecast within a `WeatherForecast`."""

    forecast_date: date = Field(..., description="Calendar date for this forecast entry.")
    condition: str = Field(
        ..., description="Dominant weather condition, e.g. 'sunny', 'rain', 'thunderstorms'."
    )
    temp_high_c: float = Field(..., description="Forecast high temperature in Celsius.")
    temp_low_c: float = Field(..., description="Forecast low temperature in Celsius.")
    precipitation_probability_pct: int = Field(
        ..., ge=0, le=100, description="Probability of precipitation as a percentage, 0-100."
    )


class WeatherForecast(BaseModel):
    """Multi-day forecast payload returned for a resolved destination."""

    resolved_location: str = Field(
        ..., description="Fully disambiguated location name, e.g. 'Paris, France'."
    )
    daily_forecasts: list[DailyForecast] = Field(
        ...,
        description="One entry per calendar day in the requested [start_date, end_date] range, in order.",
    )


def _lookup_location(location: str) -> tuple[str, str] | list[str] | None:
    """Resolve a free-text location string against the local registry.

    Returns:
        A `(display_name, country)` tuple if exactly one match exists, a list
        of candidate "name, country" strings if the query is ambiguous
        (multiple matches), or `None` if there is no match at all.
    """
    matches = _LOCATION_REGISTRY.get(location.strip().lower())
    if matches is None:
        return None
    if len(matches) == 1:
        return matches[0]
    return [f"{name}, {country}" for name, country in matches]


def _deterministic_daily_forecast(resolved_location: str, day: date) -> DailyForecast:
    """Derive a stable pseudo-forecast for a location and date.

    Hashes `(resolved_location, day)` with SHA-256 rather than using
    `random`, so repeated calls with the same arguments always return the
    same forecast — required for reproducible tests and golden evals without
    a live network dependency.
    """
    digest = hashlib.sha256(f"{resolved_location}|{day.isoformat()}".encode()).hexdigest()
    seed = int(digest[:8], 16)
    temp_high_c = 10 + (seed % 25)
    temp_low_c = temp_high_c - 5 - (seed % 6)
    return DailyForecast(
        forecast_date=day,
        condition=_CONDITIONS[seed % len(_CONDITIONS)],
        temp_high_c=float(temp_high_c),
        temp_low_c=float(temp_low_c),
        precipitation_probability_pct=(seed // 7) % 101,
    )


def fetch_destination_weather_forecast(
    location: str,
    start_date: str,
    end_date: str,
) -> ToolResultEnvelope[WeatherForecast] | ToolErrorEnvelope:
    """Fetch a daily weather forecast for a destination over a date range.

    Args:
        location: Free-text destination name, e.g. "Paris" or "Tokyo". Must
            resolve unambiguously against the local location registry. If
            multiple cities share the name (e.g. "Paris" matches France,
            Texas, and Ontario), an error envelope asking the caller to
            disambiguate is returned instead of guessing.
        start_date: Inclusive start of the forecast window, ISO 8601
            (`YYYY-MM-DD`).
        end_date: Inclusive end of the forecast window, ISO 8601
            (`YYYY-MM-DD`). Must not precede `start_date`, and the window
            must not exceed `MAX_FORECAST_HORIZON_DAYS` (14) days.

    Returns:
        On success, `ToolResultEnvelope[WeatherForecast]` whose `data` holds
        the resolved location name and one `DailyForecast` per day in range.
        On failure, `ToolErrorEnvelope` with one of:
            - `LOCATION_NOT_FOUND`: no registry match for `location`.
            - `LOCATION_AMBIGUOUS`: multiple registry matches for `location`.
            - `INVALID_DATE_RANGE`: unparsable dates, `end_date` before
              `start_date`, or a window exceeding the forecast horizon.

    Example:
        >>> result = fetch_destination_weather_forecast("Tokyo", "2026-09-01", "2026-09-03")
        >>> result.data.resolved_location
        'Tokyo, Japan'
    """
    try:
        parsed_start = date.fromisoformat(start_date)
        parsed_end = date.fromisoformat(end_date)
    except ValueError:
        return make_error(
            ToolErrorCode.INVALID_DATE_RANGE,
            f"Could not parse start_date='{start_date}' / end_date='{end_date}' as ISO 8601 dates.",
            "Retry with dates formatted as YYYY-MM-DD, e.g. '2026-09-01'.",
        )

    if parsed_end < parsed_start:
        return make_error(
            ToolErrorCode.INVALID_DATE_RANGE,
            f"end_date ({end_date}) is before start_date ({start_date}).",
            "Swap the arguments or ask the user to confirm the intended travel dates.",
        )

    window_days = (parsed_end - parsed_start).days + 1
    if window_days > MAX_FORECAST_HORIZON_DAYS:
        return make_error(
            ToolErrorCode.INVALID_DATE_RANGE,
            f"Requested range spans {window_days} days; forecasts are only "
            f"available up to {MAX_FORECAST_HORIZON_DAYS} days out.",
            f"Narrow the date range to {MAX_FORECAST_HORIZON_DAYS} days or "
            "fewer, or tell the user long-range forecasts aren't available.",
        )

    resolution = _lookup_location(location)
    if resolution is None:
        return make_error(
            ToolErrorCode.LOCATION_NOT_FOUND,
            f"No location found matching '{location}'.",
            "Ask the user for a more specific or differently spelled destination.",
        )
    if isinstance(resolution, list):
        return make_error(
            ToolErrorCode.LOCATION_AMBIGUOUS,
            f"Multiple matches found for '{location}': {', '.join(resolution)}.",
            "Ask the user to specify the country/state, or retry with "
            "'City, Country' format, e.g. 'Paris, France'.",
        )

    name, country = resolution
    resolved_location = f"{name}, {country}"
    daily_forecasts = [
        _deterministic_daily_forecast(resolved_location, parsed_start + timedelta(days=offset))
        for offset in range(window_days)
    ]
    return make_success(
        WeatherForecast(resolved_location=resolved_location, daily_forecasts=daily_forecasts)
    )
