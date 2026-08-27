"""Task 1.2 verification: settings load strictly from the environment."""

import pydantic

from src.config import Settings, get_settings


def test_settings_load_defaults_when_env_absent(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings = Settings(_env_file=None)

    assert settings.model_fast == "gemini-3.5-flash"
    assert settings.model_pro == "gemini-3.1-pro"
    assert settings.database_path == "data/travel_agent.db"
    assert settings.has_gemini_api_key() is False


def test_settings_read_values_from_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-secret-key")
    monkeypatch.setenv("MODEL_FAST", "gemini-3.5-flash-custom")
    monkeypatch.setenv("HISTORY_WINDOW_TURNS", "42")

    settings = Settings(_env_file=None)

    assert settings.has_gemini_api_key() is True
    assert settings.model_fast == "gemini-3.5-flash-custom"
    assert settings.history_window_turns == 42


def test_gemini_api_key_is_a_secret_and_not_leaked_in_repr(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-value")
    settings = Settings(_env_file=None)

    assert "super-secret-value" not in repr(settings)
    assert "super-secret-value" not in str(settings)
    assert settings.gemini_api_key.get_secret_value() == "super-secret-value"


def test_history_window_turns_rejects_non_integer(monkeypatch):
    monkeypatch.setenv("HISTORY_WINDOW_TURNS", "not-a-number")

    try:
        Settings(_env_file=None)
    except pydantic.ValidationError:
        pass
    else:
        raise AssertionError("Expected ValidationError for invalid integer env var")


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
