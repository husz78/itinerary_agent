"""Tests for src/tools: envelope schemas, tool implementations, and error recovery payloads."""

import pytest
from pydantic import ValidationError

from src.tools.base import (
    ToolErrorCode,
    ToolErrorEnvelope,
    ToolResultEnvelope,
    ToolStatus,
    make_error,
    make_success,
)
from src.tools.attraction_tool import search_attractions_and_activities
from src.tools.booking_tool import (
    BookingStatus,
    confirm_reservation_booking,
    stage_provisional_booking,
)
from src.tools.transit_tool import calculate_transit_route_estimate
from src.tools.weather_tool import fetch_destination_weather_forecast


def test_make_success_wraps_data_with_success_status():
    envelope = make_success({"temp_c": 22})

    assert envelope.status == ToolStatus.SUCCESS
    assert envelope.data == {"temp_c": 22}


def test_success_envelope_serializes_to_expected_shape():
    envelope = make_success({"city": "Paris"})

    assert envelope.model_dump() == {
        "status": ToolStatus.SUCCESS,
        "data": {"city": "Paris"},
    }


def test_make_error_wraps_recovery_guidance():
    envelope = make_error(
        ToolErrorCode.LOCATION_AMBIGUOUS,
        "Multiple matches found for 'Paris' (France, Texas, Ontario).",
        "Specify country or state, or retry with an ISO country code.",
    )

    assert envelope.status == ToolStatus.ERROR
    assert envelope.error_code == ToolErrorCode.LOCATION_AMBIGUOUS
    assert "Paris" in envelope.message
    assert envelope.recovery_instruction


def test_error_envelope_requires_all_fields():
    with pytest.raises(ValidationError):
        ToolErrorEnvelope(error_code=ToolErrorCode.RATE_LIMITED, message="Too many requests")


def test_result_envelope_defaults_status_to_success():
    envelope = ToolResultEnvelope(data=[1, 2, 3])

    assert envelope.status == ToolStatus.SUCCESS
    assert envelope.data == [1, 2, 3]


def test_error_envelope_defaults_status_to_error():
    envelope = ToolErrorEnvelope(
        error_code=ToolErrorCode.UPSTREAM_API_ERROR,
        message="Weather API timed out.",
        recovery_instruction="Retry once after a short backoff.",
    )

    assert envelope.status == ToolStatus.ERROR


# --- fetch_destination_weather_forecast -------------------------------------


def test_weather_forecast_success_resolves_unambiguous_location():
    result = fetch_destination_weather_forecast("Tokyo", "2026-09-01", "2026-09-03")

    assert isinstance(result, ToolResultEnvelope)
    assert result.data.resolved_location == "Tokyo, Japan"
    assert len(result.data.daily_forecasts) == 3
    assert [f.forecast_date.isoformat() for f in result.data.daily_forecasts] == [
        "2026-09-01",
        "2026-09-02",
        "2026-09-03",
    ]


def test_weather_forecast_is_deterministic_across_calls():
    first = fetch_destination_weather_forecast("Rome", "2026-09-01", "2026-09-01")
    second = fetch_destination_weather_forecast("Rome", "2026-09-01", "2026-09-01")

    assert first.data.daily_forecasts[0] == second.data.daily_forecasts[0]


def test_weather_forecast_ambiguous_location_returns_guided_error():
    result = fetch_destination_weather_forecast("Paris", "2026-09-01", "2026-09-02")

    assert isinstance(result, ToolErrorEnvelope)
    assert result.error_code == ToolErrorCode.LOCATION_AMBIGUOUS
    assert "France" in result.message and "Texas" in result.message
    assert result.recovery_instruction


def test_weather_forecast_unknown_location_returns_not_found_error():
    result = fetch_destination_weather_forecast("Atlantis", "2026-09-01", "2026-09-02")

    assert isinstance(result, ToolErrorEnvelope)
    assert result.error_code == ToolErrorCode.LOCATION_NOT_FOUND


def test_weather_forecast_end_before_start_returns_invalid_date_range():
    result = fetch_destination_weather_forecast("Tokyo", "2026-09-05", "2026-09-01")

    assert isinstance(result, ToolErrorEnvelope)
    assert result.error_code == ToolErrorCode.INVALID_DATE_RANGE


def test_weather_forecast_unparsable_dates_returns_invalid_date_range():
    result = fetch_destination_weather_forecast("Tokyo", "not-a-date", "2026-09-01")

    assert isinstance(result, ToolErrorEnvelope)
    assert result.error_code == ToolErrorCode.INVALID_DATE_RANGE


def test_weather_forecast_exceeding_horizon_returns_invalid_date_range():
    result = fetch_destination_weather_forecast("Tokyo", "2026-09-01", "2026-10-01")

    assert isinstance(result, ToolErrorEnvelope)
    assert result.error_code == ToolErrorCode.INVALID_DATE_RANGE


# --- search_attractions_and_activities --------------------------------------


def test_attraction_search_success_resolves_unambiguous_city():
    result = search_attractions_and_activities("Tokyo", None, "luxury", 4.0)

    assert isinstance(result, ToolResultEnvelope)
    assert result.data.resolved_city == "Tokyo, Japan"
    assert len(result.data.results) > 0


def test_attraction_search_is_deterministic_across_calls():
    first = search_attractions_and_activities("Rome", None, "luxury", 4.0)
    second = search_attractions_and_activities("Rome", None, "luxury", 4.0)

    assert first.data.results == second.data.results


def test_attraction_search_results_sorted_by_rating_descending():
    result = search_attractions_and_activities("Tokyo", None, "luxury", 4.0)

    ratings = [attraction.rating for attraction in result.data.results]
    assert ratings == sorted(ratings, reverse=True)


def test_attraction_search_filters_by_category():
    result = search_attractions_and_activities("Tokyo", "food", "luxury", 4.0)

    assert isinstance(result, ToolResultEnvelope)
    assert all(attraction.category == "food" for attraction in result.data.results)


def test_attraction_search_filters_by_budget_tier():
    result = search_attractions_and_activities("Tokyo", None, "free", 4.0)

    assert isinstance(result, ToolResultEnvelope)
    assert all(attraction.price_tier == "free" for attraction in result.data.results)


def test_attraction_search_filters_by_duration_hours():
    result = search_attractions_and_activities("Tokyo", None, "luxury", 1.5)

    assert isinstance(result, ToolResultEnvelope)
    assert all(attraction.typical_duration_hours <= 1.5 for attraction in result.data.results)


def test_attraction_search_ambiguous_city_name_returns_guided_error():
    result = search_attractions_and_activities("Springfield", None, "luxury", 4.0)

    assert isinstance(result, ToolErrorEnvelope)
    assert result.error_code == ToolErrorCode.LOCATION_AMBIGUOUS
    assert "Illinois" in result.message and "Missouri" in result.message


def test_attraction_search_unknown_city_returns_not_found_error():
    result = search_attractions_and_activities("Atlantis", None, "luxury", 4.0)

    assert isinstance(result, ToolErrorEnvelope)
    assert result.error_code == ToolErrorCode.LOCATION_NOT_FOUND


def test_attraction_search_invalid_budget_tier_returns_validation_error():
    result = search_attractions_and_activities("Tokyo", None, "cheap", 4.0)

    assert isinstance(result, ToolErrorEnvelope)
    assert result.error_code == ToolErrorCode.VALIDATION_ERROR


def test_attraction_search_invalid_category_returns_validation_error():
    result = search_attractions_and_activities("Tokyo", "spa", "luxury", 4.0)

    assert isinstance(result, ToolErrorEnvelope)
    assert result.error_code == ToolErrorCode.VALIDATION_ERROR


def test_attraction_search_non_positive_duration_returns_validation_error():
    result = search_attractions_and_activities("Tokyo", None, "luxury", 0)

    assert isinstance(result, ToolErrorEnvelope)
    assert result.error_code == ToolErrorCode.VALIDATION_ERROR


def test_attraction_search_no_matches_returns_resource_not_found():
    result = search_attractions_and_activities("Tokyo", "shopping", "free", 4.0)

    assert isinstance(result, ToolErrorEnvelope)
    assert result.error_code == ToolErrorCode.RESOURCE_NOT_FOUND


# --- calculate_transit_route_estimate ---------------------------------------


def test_transit_route_flight_success_resolves_unambiguous_locations():
    result = calculate_transit_route_estimate("Tokyo", "Rome", "flight")

    assert isinstance(result, ToolResultEnvelope)
    assert result.data.resolved_origin == "Tokyo, Japan"
    assert result.data.resolved_destination == "Rome, Italy"
    assert result.data.travel_mode == "flight"
    assert result.data.distance_km == pytest.approx(7787.0)
    assert result.data.estimated_duration_minutes > 0
    assert result.data.estimated_cost_usd > 0


def test_transit_route_is_deterministic_across_calls():
    first = calculate_transit_route_estimate("Tokyo", "Rome", "flight")
    second = calculate_transit_route_estimate("Tokyo", "Rome", "flight")

    assert first.data == second.data


def test_transit_route_distance_is_order_independent():
    forward = calculate_transit_route_estimate("Tokyo", "Rome", "flight")
    backward = calculate_transit_route_estimate("Rome", "Tokyo", "flight")

    assert forward.data.distance_km == backward.data.distance_km


def test_transit_route_same_origin_and_destination_has_zero_distance():
    result = calculate_transit_route_estimate("Tokyo", "Tokyo", "driving")

    assert isinstance(result, ToolResultEnvelope)
    assert result.data.distance_km == 0.0


def test_transit_route_ground_mode_exceeding_max_distance_returns_route_not_supported():
    result = calculate_transit_route_estimate("Tokyo", "Rome", "driving")

    assert isinstance(result, ToolErrorEnvelope)
    assert result.error_code == ToolErrorCode.ROUTE_NOT_SUPPORTED
    assert "flight" in result.recovery_instruction


def test_transit_route_walking_exceeding_max_distance_returns_route_not_supported():
    result = calculate_transit_route_estimate("Tokyo", "San Francisco", "walking")

    assert isinstance(result, ToolErrorEnvelope)
    assert result.error_code == ToolErrorCode.ROUTE_NOT_SUPPORTED


def test_transit_route_flight_below_min_distance_returns_route_not_supported():
    result = calculate_transit_route_estimate("Tokyo", "Tokyo", "flight")

    assert isinstance(result, ToolErrorEnvelope)
    assert result.error_code == ToolErrorCode.ROUTE_NOT_SUPPORTED


def test_transit_route_invalid_travel_mode_returns_validation_error():
    result = calculate_transit_route_estimate("Tokyo", "Rome", "teleport")

    assert isinstance(result, ToolErrorEnvelope)
    assert result.error_code == ToolErrorCode.VALIDATION_ERROR


def test_transit_route_ambiguous_origin_returns_guided_error():
    result = calculate_transit_route_estimate("Paris", "Tokyo", "flight")

    assert isinstance(result, ToolErrorEnvelope)
    assert result.error_code == ToolErrorCode.LOCATION_AMBIGUOUS
    assert "origin" in result.message


def test_transit_route_ambiguous_destination_returns_guided_error():
    result = calculate_transit_route_estimate("Tokyo", "Paris", "flight")

    assert isinstance(result, ToolErrorEnvelope)
    assert result.error_code == ToolErrorCode.LOCATION_AMBIGUOUS
    assert "destination" in result.message


def test_transit_route_unknown_origin_returns_not_found_error():
    result = calculate_transit_route_estimate("Atlantis", "Tokyo", "flight")

    assert isinstance(result, ToolErrorEnvelope)
    assert result.error_code == ToolErrorCode.LOCATION_NOT_FOUND
    assert "origin" in result.message


def test_transit_route_unknown_destination_returns_not_found_error():
    result = calculate_transit_route_estimate("Tokyo", "Atlantis", "flight")

    assert isinstance(result, ToolErrorEnvelope)
    assert result.error_code == ToolErrorCode.LOCATION_NOT_FOUND
    assert "destination" in result.message


# --- stage_provisional_booking / confirm_reservation_booking ----------------


def test_stage_booking_success_returns_pending_confirmation():
    result = stage_provisional_booking("hotel", "HTL-42", "2026-09-01", 199.0)

    assert isinstance(result, ToolResultEnvelope)
    assert result.data.status == BookingStatus.PENDING_CONFIRMATION
    assert result.data.provisional_booking_id
    assert result.data.confirmation_token


def test_stage_booking_generates_unique_ids_and_tokens():
    first = stage_provisional_booking("hotel", "HTL-42", "2026-09-01", 199.0)
    second = stage_provisional_booking("hotel", "HTL-42", "2026-09-01", 199.0)

    assert first.data.provisional_booking_id != second.data.provisional_booking_id
    assert first.data.confirmation_token != second.data.confirmation_token


def test_stage_booking_invalid_reservation_type_returns_validation_error():
    result = stage_provisional_booking("cruise", "SHIP-1", "2026-09-01", 199.0)

    assert isinstance(result, ToolErrorEnvelope)
    assert result.error_code == ToolErrorCode.VALIDATION_ERROR


def test_stage_booking_blank_provider_id_returns_validation_error():
    result = stage_provisional_booking("hotel", "   ", "2026-09-01", 199.0)

    assert isinstance(result, ToolErrorEnvelope)
    assert result.error_code == ToolErrorCode.VALIDATION_ERROR


def test_stage_booking_blank_slot_returns_validation_error():
    result = stage_provisional_booking("hotel", "HTL-42", "   ", 199.0)

    assert isinstance(result, ToolErrorEnvelope)
    assert result.error_code == ToolErrorCode.VALIDATION_ERROR


def test_stage_booking_non_positive_price_returns_validation_error():
    result = stage_provisional_booking("hotel", "HTL-42", "2026-09-01", 0)

    assert isinstance(result, ToolErrorEnvelope)
    assert result.error_code == ToolErrorCode.VALIDATION_ERROR


def test_confirm_booking_success_with_matching_token():
    staged = stage_provisional_booking("flight", "AA1234", "2026-09-01T08:00:00", 450.0)

    result = confirm_reservation_booking(
        staged.data.provisional_booking_id, staged.data.confirmation_token
    )

    assert isinstance(result, ToolResultEnvelope)
    assert result.data.status == BookingStatus.CONFIRMED
    assert result.data.provisional_booking_id == staged.data.provisional_booking_id
    assert result.data.price == 450.0


def test_confirm_booking_unknown_id_returns_resource_not_found():
    result = confirm_reservation_booking("book_doesnotexist", "ABCD1234")

    assert isinstance(result, ToolErrorEnvelope)
    assert result.error_code == ToolErrorCode.RESOURCE_NOT_FOUND


def test_confirm_booking_wrong_token_returns_authorization_required():
    staged = stage_provisional_booking("activity", "TOUR-9", "2026-09-01T10:00:00", 75.0)

    result = confirm_reservation_booking(staged.data.provisional_booking_id, "WRONGTOK")

    assert isinstance(result, ToolErrorEnvelope)
    assert result.error_code == ToolErrorCode.AUTHORIZATION_REQUIRED


def test_confirm_booking_already_confirmed_returns_validation_error():
    staged = stage_provisional_booking("restaurant", "REST-7", "2026-09-01T19:00:00", 60.0)
    confirm_reservation_booking(staged.data.provisional_booking_id, staged.data.confirmation_token)

    result = confirm_reservation_booking(
        staged.data.provisional_booking_id, staged.data.confirmation_token
    )

    assert isinstance(result, ToolErrorEnvelope)
    assert result.error_code == ToolErrorCode.VALIDATION_ERROR
