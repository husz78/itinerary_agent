"""Tests for strategic Gemini 3 family model routing across agents (Task 5.2/5.3)."""

import pytest

from src.agents.attraction_search import create_attraction_search_agent
from src.agents.booking_specialist import create_booking_agent
from src.agents.coordinator import create_travel_coordinator_agent
from src.agents.weather_specialist import create_weather_specialist_agent
from src.config import get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_weather_and_attraction_specialists_default_to_fast_model(monkeypatch):
    monkeypatch.setenv("MODEL_FAST", "gemini-3.5-flash-custom")

    weather = create_weather_specialist_agent()
    attraction = create_attraction_search_agent()

    assert weather.model == "gemini-3.5-flash-custom"
    assert attraction.model == "gemini-3.5-flash-custom"


def test_booking_agent_and_coordinator_default_to_pro_model(monkeypatch):
    monkeypatch.setenv("MODEL_PRO", "gemini-3.1-pro-custom")

    booking = create_booking_agent()
    coordinator = create_travel_coordinator_agent()

    assert booking.model == "gemini-3.1-pro-custom"
    assert coordinator.model == "gemini-3.1-pro-custom"


def test_coordinator_wires_fast_model_specialists_under_a_pro_model_root(monkeypatch):
    monkeypatch.setenv("MODEL_FAST", "gemini-3.5-flash-custom")
    monkeypatch.setenv("MODEL_PRO", "gemini-3.1-pro-custom")

    coordinator = create_travel_coordinator_agent()
    models_by_name = {agent.name: agent.model for agent in coordinator.sub_agents}

    assert coordinator.model == "gemini-3.1-pro-custom"
    assert models_by_name["WeatherSpecialistAgent"] == "gemini-3.5-flash-custom"
    assert models_by_name["AttractionSearchAgent"] == "gemini-3.5-flash-custom"
    assert models_by_name["BookingAgent"] == "gemini-3.1-pro-custom"


def test_default_routing_matches_gemini_3_family_defaults():
    weather = create_weather_specialist_agent()
    attraction = create_attraction_search_agent()
    booking = create_booking_agent()
    coordinator = create_travel_coordinator_agent()

    assert weather.model == "gemini-3.5-flash"
    assert attraction.model == "gemini-3.5-flash"
    assert booking.model == "gemini-3.1-pro"
    assert coordinator.model == "gemini-3.1-pro"


def test_explicit_model_override_takes_precedence_over_settings(monkeypatch):
    monkeypatch.setenv("MODEL_FAST", "gemini-3.5-flash-custom")

    weather = create_weather_specialist_agent(model="explicit-override-model")

    assert weather.model == "explicit-override-model"
