"""Transit & route estimation tool: `calculate_transit_route_estimate`.

Provides deterministic, locally computed route estimates (distance, duration,
cost) between two locations for a given travel mode. There is no outbound
network call — a small local city registry stands in for a geocoding lookup
(shared naming convention with `weather_tool` / `attraction_tool`), and
distances are derived from a stable hash of the resolved origin/destination
pair rather than a real routing engine, so results are reproducible across
runs (required for deterministic tests and evals). A real deployment would
swap `_lookup_city` and `_deterministic_distance_km` for calls to a mapping/
routing API; the public function signature and the `ToolResultEnvelope` /
`ToolErrorEnvelope` contract would stay unchanged.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, Field

from src.tools.base import (
    ToolErrorCode,
    ToolErrorEnvelope,
    ToolOutcome,
    ToolResultEnvelope,
    make_error,
    make_success,
)

TRAVEL_MODES = {"walking", "cycling", "driving", "transit", "flight"}

# Average cruising speed per mode, used to derive a duration estimate from
# the (mocked) distance.
_AVERAGE_SPEED_KMH = {
    "walking": 5.0,
    "cycling": 15.0,
    "driving": 60.0,
    "transit": 40.0,
    "flight": 800.0,
}

# Fixed overhead added to every trip of this mode (security lines, boarding,
# waiting for a vehicle, parking, etc.), in minutes.
_OVERHEAD_MINUTES = {
    "walking": 0.0,
    "cycling": 0.0,
    "driving": 5.0,
    "transit": 15.0,
    "flight": 120.0,
}

_BASE_FARE_USD = {
    "walking": 0.0,
    "cycling": 0.0,
    "driving": 0.0,
    "transit": 2.5,
    "flight": 80.0,
}

_COST_PER_KM_USD = {
    "walking": 0.0,
    "cycling": 0.0,
    "driving": 0.5,
    "transit": 0.15,
    "flight": 0.12,
}

# Ground modes can't realistically cover intercontinental distances; beyond
# this cap the caller should be redirected to travel_mode="flight". Flight
# has no cap.
_MAX_DISTANCE_KM = {
    "walking": 20.0,
    "cycling": 60.0,
    "driving": 800.0,
    "transit": 800.0,
}

# Flights aren't practical below this distance; the caller should be
# redirected to a ground travel_mode instead.
_MIN_FLIGHT_DISTANCE_KM = 100.0

# Local stand-in for a geocoding service. Maps a lowercased free-text query to
# every known (display_name, country) match; more than one match is ambiguous.
_CITY_REGISTRY: dict[str, list[tuple[str, str]]] = {
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


class RouteEstimate(BaseModel):
    """Estimated distance, duration, and cost for a single-mode trip."""

    resolved_origin: str = Field(
        ..., description="Fully disambiguated origin location, e.g. 'Paris, France'."
    )
    resolved_destination: str = Field(
        ..., description="Fully disambiguated destination location, e.g. 'Tokyo, Japan'."
    )
    travel_mode: str = Field(..., description="Requested travel mode for this estimate.")
    distance_km: float = Field(..., ge=0, description="Estimated route distance in kilometers.")
    estimated_duration_minutes: float = Field(
        ..., ge=0, description="Estimated total trip duration in minutes, including overhead."
    )
    estimated_cost_usd: float = Field(
        ..., ge=0, description="Estimated total trip cost in US dollars."
    )


def _lookup_city(city: str) -> tuple[str, str] | list[str] | None:
    """Resolve a free-text city string against the local registry.

    Returns:
        A `(display_name, country)` tuple if exactly one match exists, a list
        of candidate "name, country" strings if the query is ambiguous
        (multiple matches), or `None` if there is no match at all.
    """
    matches = _CITY_REGISTRY.get(city.strip().lower())
    if matches is None:
        return None
    if len(matches) == 1:
        return matches[0]
    return [f"{name}, {country}" for name, country in matches]


def _resolve_or_error(field_name: str, value: str) -> tuple[str, str] | ToolErrorEnvelope:
    """Resolve a location field, returning a guided error keyed to its field name."""
    resolution = _lookup_city(value)
    if resolution is None:
        return make_error(
            ToolErrorCode.LOCATION_NOT_FOUND,
            f"No location found matching {field_name}='{value}'.",
            f"Ask the user for a more specific or differently spelled {field_name}.",
        )
    if isinstance(resolution, list):
        return make_error(
            ToolErrorCode.LOCATION_AMBIGUOUS,
            f"Multiple matches found for {field_name}='{value}': {', '.join(resolution)}.",
            f"Ask the user to specify the country/state for {field_name}, or retry with "
            "'City, Country' format, e.g. 'Paris, France'.",
        )
    return resolution


def _deterministic_distance_km(resolved_origin: str, resolved_destination: str) -> float:
    """Derive a stable pseudo-distance for a pair of resolved locations.

    Hashes the order-independent pair with SHA-256 rather than using
    `random`, so repeated calls with the same arguments (in either order)
    always return the same distance — required for reproducible tests and
    golden evals without a live routing dependency.
    """
    pair_key = "|".join(sorted([resolved_origin, resolved_destination]))
    digest = hashlib.sha256(pair_key.encode()).hexdigest()
    seed = int(digest[:8], 16)
    return float(1 + seed % 12000)


def calculate_transit_route_estimate(
    origin: str,
    destination: str,
    travel_mode: str,
) -> ToolOutcome[RouteEstimate]:
    """Estimate distance, duration, and cost for a single-mode trip.

    Args:
        origin: Free-text starting location, e.g. "Paris" or "Tokyo". Must
            resolve unambiguously against the local location registry.
        destination: Free-text ending location, e.g. "Paris" or "Tokyo".
            Must resolve unambiguously against the local location registry.
        travel_mode: One of "walking", "cycling", "driving", "transit", or
            "flight". Ground modes ("walking", "cycling", "driving",
            "transit") are only supported up to a realistic maximum
            distance for that mode; "flight" has no maximum but requires a
            minimum distance of 100 km.

    Returns:
        On success, `ToolResultEnvelope[RouteEstimate]` whose `data` holds
        the resolved origin/destination, distance, estimated duration, and
        estimated cost.
        On failure, `ToolErrorEnvelope` with one of:
            - `LOCATION_NOT_FOUND`: no registry match for `origin` or
              `destination`.
            - `LOCATION_AMBIGUOUS`: multiple registry matches for `origin`
              or `destination`.
            - `VALIDATION_ERROR`: `travel_mode` is not one of the supported
              values.
            - `ROUTE_NOT_SUPPORTED`: the resolved distance exceeds the
              realistic maximum for a ground `travel_mode` (use "flight"
              instead), or is below the minimum practical distance for
              "flight" (use a ground `travel_mode` instead).

    Example:
        >>> result = calculate_transit_route_estimate("Paris, France", "Rome", "flight")
        >>> result.data.travel_mode
        'flight'
    """
    if travel_mode not in TRAVEL_MODES:
        return make_error(
            ToolErrorCode.VALIDATION_ERROR,
            f"Unknown travel_mode '{travel_mode}'. Valid values: {sorted(TRAVEL_MODES)}.",
            f"Retry with one of: {sorted(TRAVEL_MODES)}.",
        )

    origin_resolution = _resolve_or_error("origin", origin)
    if isinstance(origin_resolution, ToolErrorEnvelope):
        return origin_resolution
    destination_resolution = _resolve_or_error("destination", destination)
    if isinstance(destination_resolution, ToolErrorEnvelope):
        return destination_resolution

    origin_name, origin_country = origin_resolution
    destination_name, destination_country = destination_resolution
    resolved_origin = f"{origin_name}, {origin_country}"
    resolved_destination = f"{destination_name}, {destination_country}"

    distance_km = (
        0.0
        if resolved_origin == resolved_destination
        else _deterministic_distance_km(resolved_origin, resolved_destination)
    )

    if travel_mode == "flight":
        if distance_km < _MIN_FLIGHT_DISTANCE_KM:
            return make_error(
                ToolErrorCode.ROUTE_NOT_SUPPORTED,
                f"'{resolved_origin}' and '{resolved_destination}' are too close "
                f"({distance_km:.0f} km) for a flight.",
                "Retry with travel_mode='driving' or 'transit' for short-distance routes.",
            )
    else:
        max_distance = _MAX_DISTANCE_KM[travel_mode]
        if distance_km > max_distance:
            return make_error(
                ToolErrorCode.ROUTE_NOT_SUPPORTED,
                f"Route from '{resolved_origin}' to '{resolved_destination}' spans "
                f"~{distance_km:.0f} km, which exceeds the realistic maximum for "
                f"travel_mode='{travel_mode}' ({max_distance:.0f} km).",
                "Retry with travel_mode='flight' for long-distance routes.",
            )

    speed_kmh = _AVERAGE_SPEED_KMH[travel_mode]
    duration_minutes = (distance_km / speed_kmh) * 60 + _OVERHEAD_MINUTES[travel_mode]
    cost_usd = _BASE_FARE_USD[travel_mode] + distance_km * _COST_PER_KM_USD[travel_mode]

    return make_success(
        RouteEstimate(
            resolved_origin=resolved_origin,
            resolved_destination=resolved_destination,
            travel_mode=travel_mode,
            distance_km=distance_km,
            estimated_duration_minutes=round(duration_minutes, 1),
            estimated_cost_usd=round(cost_usd, 2),
        )
    )
