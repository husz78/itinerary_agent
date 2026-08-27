"""Attractions & activities search tool: `search_attractions_and_activities`.

Provides deterministic, locally computed attraction/activity search results
for a city. There is no outbound network call — a small local city registry
stands in for a geocoding lookup (shared naming convention with
`weather_tool`), and results come from a static local catalog filtered by
category, budget tier, and available duration. A real deployment would swap
`_lookup_city` and `_CATALOG_BY_CITY` for calls to a places/search API; the
public function signature and the `ToolResultEnvelope` / `ToolErrorEnvelope`
contract would stay unchanged.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.tools.base import (
    ToolErrorCode,
    ToolErrorEnvelope,
    ToolOutcome,
    ToolResultEnvelope,
    make_error,
    make_success,
)

VALID_CATEGORIES = {
    "museum",
    "outdoor",
    "food",
    "nightlife",
    "landmark",
    "shopping",
    "family",
    "art",
    "history",
}

# Ordered cheapest to most expensive; used to test "fits within budget_tier".
BUDGET_TIER_ORDER = {"free": 0, "budget": 1, "moderate": 2, "luxury": 3}

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


class Attraction(BaseModel):
    """A single bookable attraction or activity."""

    name: str = Field(..., description="Display name of the attraction or activity.")
    category: str = Field(
        ..., description=f"One of the known categories: {sorted(VALID_CATEGORIES)}."
    )
    price_tier: str = Field(
        ..., description="One of 'free', 'budget', 'moderate', 'luxury'."
    )
    typical_duration_hours: float = Field(
        ..., gt=0, description="Typical time in hours a visitor should budget for this item."
    )
    rating: float = Field(..., ge=0, le=5, description="Aggregate visitor rating, 0-5.")
    opening_hours: str = Field(
        ..., description="Human-readable opening hours summary, e.g. '09:00-18:00 daily'."
    )


class AttractionSearchResults(BaseModel):
    """Search results payload returned for a resolved city."""

    resolved_city: str = Field(
        ..., description="Fully disambiguated city name, e.g. 'Paris, France'."
    )
    results: list[Attraction] = Field(
        ...,
        description="Attractions matching the category/budget_tier/duration_hours filters, "
        "sorted by rating descending.",
    )


_CATALOG_BY_CITY: dict[str, list[Attraction]] = {
    "Paris, France": [
        Attraction(
            name="Louvre Museum",
            category="museum",
            price_tier="moderate",
            typical_duration_hours=3.0,
            rating=4.7,
            opening_hours="09:00-18:00, closed Tuesdays",
        ),
        Attraction(
            name="Eiffel Tower Summit",
            category="landmark",
            price_tier="moderate",
            typical_duration_hours=2.0,
            rating=4.6,
            opening_hours="09:00-23:45 daily",
        ),
        Attraction(
            name="Jardin du Luxembourg",
            category="outdoor",
            price_tier="free",
            typical_duration_hours=1.5,
            rating=4.5,
            opening_hours="07:30-dusk daily",
        ),
        Attraction(
            name="Le Marais Food Walk",
            category="food",
            price_tier="budget",
            typical_duration_hours=2.5,
            rating=4.4,
            opening_hours="Varies by vendor, typically 10:00-22:00",
        ),
        Attraction(
            name="Moulin Rouge Show",
            category="nightlife",
            price_tier="luxury",
            typical_duration_hours=2.5,
            rating=4.3,
            opening_hours="19:00-01:00 daily",
        ),
    ],
    "Tokyo, Japan": [
        Attraction(
            name="Senso-ji Temple",
            category="landmark",
            price_tier="free",
            typical_duration_hours=1.5,
            rating=4.6,
            opening_hours="06:00-17:00 daily",
        ),
        Attraction(
            name="teamLab Planets",
            category="art",
            price_tier="moderate",
            typical_duration_hours=2.5,
            rating=4.8,
            opening_hours="09:00-19:00 daily",
        ),
        Attraction(
            name="Tsukiji Outer Market Food Tour",
            category="food",
            price_tier="budget",
            typical_duration_hours=2.0,
            rating=4.5,
            opening_hours="05:00-14:00 daily",
        ),
        Attraction(
            name="Shibuya Sky Observation Deck",
            category="landmark",
            price_tier="moderate",
            typical_duration_hours=1.5,
            rating=4.6,
            opening_hours="10:00-22:30 daily",
        ),
        Attraction(
            name="Golden Gai Bar Hopping",
            category="nightlife",
            price_tier="luxury",
            typical_duration_hours=3.0,
            rating=4.2,
            opening_hours="20:00-04:00 daily",
        ),
    ],
    "Rome, Italy": [
        Attraction(
            name="Colosseum & Roman Forum",
            category="history",
            price_tier="moderate",
            typical_duration_hours=3.5,
            rating=4.7,
            opening_hours="08:30-19:00 daily",
        ),
        Attraction(
            name="Vatican Museums & Sistine Chapel",
            category="museum",
            price_tier="moderate",
            typical_duration_hours=3.0,
            rating=4.6,
            opening_hours="08:00-18:00, closed Sundays",
        ),
        Attraction(
            name="Trastevere Evening Food Crawl",
            category="food",
            price_tier="budget",
            typical_duration_hours=2.5,
            rating=4.5,
            opening_hours="18:00-23:00 daily",
        ),
        Attraction(
            name="Trevi Fountain",
            category="landmark",
            price_tier="free",
            typical_duration_hours=0.5,
            rating=4.4,
            opening_hours="Always open",
        ),
    ],
    "Barcelona, Spain": [
        Attraction(
            name="Sagrada Familia",
            category="landmark",
            price_tier="moderate",
            typical_duration_hours=2.0,
            rating=4.8,
            opening_hours="09:00-18:00 daily",
        ),
        Attraction(
            name="Park Guell",
            category="outdoor",
            price_tier="budget",
            typical_duration_hours=1.5,
            rating=4.5,
            opening_hours="09:30-19:30 daily",
        ),
        Attraction(
            name="La Boqueria Market Tasting",
            category="food",
            price_tier="budget",
            typical_duration_hours=2.0,
            rating=4.4,
            opening_hours="08:00-20:30, closed Sundays",
        ),
        Attraction(
            name="Razzmatazz Nightclub",
            category="nightlife",
            price_tier="moderate",
            typical_duration_hours=4.0,
            rating=4.1,
            opening_hours="00:00-06:00 Fri-Sat",
        ),
    ],
    "New York City, USA": [
        Attraction(
            name="Metropolitan Museum of Art",
            category="museum",
            price_tier="moderate",
            typical_duration_hours=3.0,
            rating=4.7,
            opening_hours="10:00-17:00, closed Wednesdays",
        ),
        Attraction(
            name="Central Park Stroll",
            category="outdoor",
            price_tier="free",
            typical_duration_hours=1.5,
            rating=4.6,
            opening_hours="06:00-01:00 daily",
        ),
        Attraction(
            name="Chelsea Market Food Tour",
            category="food",
            price_tier="budget",
            typical_duration_hours=2.0,
            rating=4.4,
            opening_hours="07:00-21:00 daily",
        ),
        Attraction(
            name="Broadway Show",
            category="nightlife",
            price_tier="luxury",
            typical_duration_hours=2.5,
            rating=4.7,
            opening_hours="Evening performances, 19:00-22:00",
        ),
        Attraction(
            name="Fifth Avenue Shopping",
            category="shopping",
            price_tier="luxury",
            typical_duration_hours=2.5,
            rating=4.2,
            opening_hours="10:00-20:00 daily",
        ),
    ],
    "San Francisco, USA": [
        Attraction(
            name="Golden Gate Bridge Walk",
            category="outdoor",
            price_tier="free",
            typical_duration_hours=1.5,
            rating=4.7,
            opening_hours="Always open",
        ),
        Attraction(
            name="Alcatraz Island Tour",
            category="history",
            price_tier="moderate",
            typical_duration_hours=3.0,
            rating=4.7,
            opening_hours="09:00-16:30 daily",
        ),
        Attraction(
            name="Ferry Building Marketplace",
            category="food",
            price_tier="budget",
            typical_duration_hours=1.5,
            rating=4.5,
            opening_hours="10:00-18:00 daily",
        ),
        Attraction(
            name="Union Square Shopping",
            category="shopping",
            price_tier="moderate",
            typical_duration_hours=2.0,
            rating=4.1,
            opening_hours="10:00-19:00 daily",
        ),
    ],
}


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


def search_attractions_and_activities(
    city: str,
    category: str | None,
    budget_tier: str,
    duration_hours: float,
) -> ToolOutcome[AttractionSearchResults]:
    """Search for attractions and activities in a city matching given filters.

    Args:
        city: Free-text city name, e.g. "Paris" or "Tokyo". Must resolve
            unambiguously against the local city registry. If multiple
            cities share the name (e.g. "Paris" matches France, Texas, and
            Ontario), an error envelope asking the caller to disambiguate is
            returned instead of guessing.
        category: Optional category filter. If provided, must be one of:
            "museum", "outdoor", "food", "nightlife", "landmark", "shopping",
            "family", "art", "history". Pass `None` to search all categories.
        budget_tier: Maximum price tier the traveler is willing to pay, one
            of "free", "budget", "moderate", "luxury" (each tier includes all
            cheaper tiers, e.g. "moderate" also returns "free" and "budget"
            items).
        duration_hours: Maximum time in hours available for a single
            activity. Only attractions whose `typical_duration_hours` fits
            within this budget are returned. Must be greater than 0.

    Returns:
        On success, `ToolResultEnvelope[AttractionSearchResults]` whose
        `data` holds the resolved city name and matching attractions sorted
        by rating descending.
        On failure, `ToolErrorEnvelope` with one of:
            - `LOCATION_NOT_FOUND`: no registry match for `city`.
            - `LOCATION_AMBIGUOUS`: multiple registry matches for `city`.
            - `VALIDATION_ERROR`: unknown `category`, unknown `budget_tier`,
              or non-positive `duration_hours`.
            - `RESOURCE_NOT_FOUND`: the city resolved and inputs were valid,
              but no catalog entries matched the combined filters.

    Example:
        >>> result = search_attractions_and_activities("Tokyo", "food", "budget", 3.0)
        >>> result.data.resolved_city
        'Tokyo, Japan'
    """
    if duration_hours <= 0:
        return make_error(
            ToolErrorCode.VALIDATION_ERROR,
            f"duration_hours must be greater than 0, got {duration_hours}.",
            "Retry with a positive duration_hours value, e.g. 2.0.",
        )

    if budget_tier not in BUDGET_TIER_ORDER:
        return make_error(
            ToolErrorCode.VALIDATION_ERROR,
            f"Unknown budget_tier '{budget_tier}'. Valid values: {sorted(BUDGET_TIER_ORDER)}.",
            f"Retry with one of: {sorted(BUDGET_TIER_ORDER)}.",
        )

    if category is not None and category not in VALID_CATEGORIES:
        return make_error(
            ToolErrorCode.VALIDATION_ERROR,
            f"Unknown category '{category}'. Valid values: {sorted(VALID_CATEGORIES)}.",
            f"Retry with one of {sorted(VALID_CATEGORIES)}, or omit category to search all.",
        )

    resolution = _lookup_city(city)
    if resolution is None:
        return make_error(
            ToolErrorCode.LOCATION_NOT_FOUND,
            f"No city found matching '{city}'.",
            "Ask the user for a more specific or differently spelled city.",
        )
    if isinstance(resolution, list):
        return make_error(
            ToolErrorCode.LOCATION_AMBIGUOUS,
            f"Multiple matches found for '{city}': {', '.join(resolution)}.",
            "Ask the user to specify the country/state, or retry with "
            "'City, Country' format, e.g. 'Paris, France'.",
        )

    name, country = resolution
    resolved_city = f"{name}, {country}"
    catalog = _CATALOG_BY_CITY.get(resolved_city, [])
    max_tier = BUDGET_TIER_ORDER[budget_tier]
    matches = [
        attraction
        for attraction in catalog
        if BUDGET_TIER_ORDER[attraction.price_tier] <= max_tier
        and attraction.typical_duration_hours <= duration_hours
        and (category is None or attraction.category == category)
    ]

    if not matches:
        return make_error(
            ToolErrorCode.RESOURCE_NOT_FOUND,
            f"No attractions found in {resolved_city} matching category={category!r}, "
            f"budget_tier='{budget_tier}', duration_hours={duration_hours}.",
            "Retry with a higher duration_hours, a higher budget_tier, or "
            "omit category to broaden the search.",
        )

    matches.sort(key=lambda a: (-a.rating, a.name))
    return make_success(AttractionSearchResults(resolved_city=resolved_city, results=matches))
