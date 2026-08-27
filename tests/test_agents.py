"""Tests for src/agents: constitution, specialist/coordinator factories, and HITL guardrails."""

from src.agents.attraction_search import AGENT_NAME as ATTRACTION_AGENT_NAME
from src.agents.attraction_search import (
    create_attraction_search_agent,
    search_attractions_and_activities,
)
from src.agents.booking_specialist import AGENT_NAME as BOOKING_AGENT_NAME
from src.agents.booking_specialist import (
    confirm_reservation_booking,
    create_booking_agent,
    stage_provisional_booking,
)
from src.agents.constitution import (
    CORE_CONSTITUTION,
    NON_NEGOTIABLE_RULES,
    PERSONA,
    WEATHER_SPECIALIST_MANDATE,
    build_instruction,
)
from src.agents.coordinator import AGENT_NAME as COORDINATOR_AGENT_NAME
from src.agents.coordinator import calculate_transit_route_estimate, create_travel_coordinator_agent
from src.agents.weather_specialist import AGENT_NAME as WEATHER_AGENT_NAME
from src.agents.weather_specialist import (
    create_weather_specialist_agent,
    fetch_destination_weather_forecast,
)
from src.guardrails.budget_guardrail import (
    BudgetStatus,
    GroundingVerdict,
    default_grounding_evaluator,
    evaluate_budget,
    run_grounding_check,
)
from src.guardrails.hitl_manager import (
    AUTHORIZED_BOOKING_IDS_STATE_KEY,
    HITLDecision,
    authorize_booking,
    before_confirm_booking_tool_callback,
    check_booking_authorization,
)


# --- constitution -------------------------------------------------------


def test_build_instruction_includes_persona_rules_and_mandate():
    instruction = build_instruction(WEATHER_SPECIALIST_MANDATE)

    assert PERSONA in instruction
    assert NON_NEGOTIABLE_RULES in instruction
    assert WEATHER_SPECIALIST_MANDATE in instruction
    assert instruction.startswith(CORE_CONSTITUTION)


def test_non_negotiable_rules_cover_hitl_weather_and_budget_safety():
    lowered = NON_NEGOTIABLE_RULES.lower()

    assert "explicit" in lowered and "authoriz" in lowered
    assert "weather" in lowered
    assert "budget" in lowered


# --- specialist agent factories -------------------------------------------


def test_weather_specialist_agent_is_routed_to_fast_model_by_default():
    agent = create_weather_specialist_agent()

    assert agent.name == WEATHER_AGENT_NAME
    assert agent.model == "gemini-3.5-flash"
    assert fetch_destination_weather_forecast in agent.tools
    assert agent.sub_agents == []


def test_attraction_search_agent_is_routed_to_fast_model_by_default():
    agent = create_attraction_search_agent()

    assert agent.name == ATTRACTION_AGENT_NAME
    assert agent.model == "gemini-3.5-flash"
    assert search_attractions_and_activities in agent.tools


def test_booking_agent_is_routed_to_pro_model_and_has_hitl_guard():
    agent = create_booking_agent()

    assert agent.name == BOOKING_AGENT_NAME
    assert agent.model == "gemini-3.1-pro"
    assert stage_provisional_booking in agent.tools
    assert confirm_reservation_booking in agent.tools
    assert agent.before_tool_callback is before_confirm_booking_tool_callback


def test_agent_factory_accepts_model_override():
    agent = create_weather_specialist_agent(model="custom-fast-model")

    assert agent.model == "custom-fast-model"


def test_wrapped_weather_tool_returns_json_serializable_success_envelope():
    payload = fetch_destination_weather_forecast("Tokyo", "2026-09-01", "2026-09-02")

    assert payload["status"] == "success"
    assert payload["data"]["resolved_location"] == "Tokyo, Japan"


def test_wrapped_weather_tool_returns_json_serializable_error_envelope():
    payload = fetch_destination_weather_forecast("Nowhereville", "2026-09-01", "2026-09-02")

    assert payload["status"] == "error"
    assert payload["error_code"] == "LOCATION_NOT_FOUND"
    assert payload["recovery_instruction"]


def test_coordinator_tool_wraps_transit_tool():
    payload = calculate_transit_route_estimate("Tokyo", "Rome", "flight")

    assert payload["status"] == "success"
    assert payload["data"]["travel_mode"] == "flight"


# --- coordinator wiring / multi-agent handoffs ----------------------------


def test_coordinator_is_routed_to_pro_model_with_all_specialists_attached():
    coordinator = create_travel_coordinator_agent()

    assert coordinator.name == COORDINATOR_AGENT_NAME
    assert coordinator.model == "gemini-3.1-pro"
    sub_agent_names = {sub.name for sub in coordinator.sub_agents}
    assert sub_agent_names == {WEATHER_AGENT_NAME, ATTRACTION_AGENT_NAME, BOOKING_AGENT_NAME}
    assert calculate_transit_route_estimate in coordinator.tools


def test_coordinator_sub_agents_have_coordinator_as_parent():
    coordinator = create_travel_coordinator_agent()

    for sub_agent in coordinator.sub_agents:
        assert sub_agent.parent_agent is coordinator


def test_each_factory_call_produces_independent_sub_agent_instances():
    coordinator_a = create_travel_coordinator_agent()
    coordinator_b = create_travel_coordinator_agent()

    assert coordinator_a.sub_agents[0] is not coordinator_b.sub_agents[0]


# --- HITL guardrail ---------------------------------------------------------


def test_check_booking_authorization_blocks_by_default():
    result = check_booking_authorization({}, "book_abc123")

    assert result.decision is HITLDecision.BLOCK_MISSING_AUTHORIZATION


def test_authorize_booking_then_check_allows():
    state = {}
    authorize_booking(state, "book_abc123")

    result = check_booking_authorization(state, "book_abc123")

    assert result.decision is HITLDecision.ALLOW
    assert state[AUTHORIZED_BOOKING_IDS_STATE_KEY] == ["book_abc123"]


def test_authorize_booking_does_not_authorize_other_bookings():
    state = {}
    authorize_booking(state, "book_abc123")

    result = check_booking_authorization(state, "book_other")

    assert result.decision is HITLDecision.BLOCK_MISSING_AUTHORIZATION


async def test_before_confirm_booking_tool_callback_blocks_unauthorized_confirmation():
    class DummyTool:
        name = "confirm_reservation_booking"

    class DummyToolContext:
        state = {}

    response = await before_confirm_booking_tool_callback(
        DummyTool(), {"provisional_booking_id": "book_abc123"}, DummyToolContext()
    )

    assert response["status"] == "error"
    assert response["error_code"] == "AUTHORIZATION_REQUIRED"
    assert response["recovery_instruction"]


async def test_before_confirm_booking_tool_callback_allows_authorized_confirmation():
    class DummyTool:
        name = "confirm_reservation_booking"

    class DummyToolContext:
        state = {"authorized_booking_ids": ["book_abc123"]}

    response = await before_confirm_booking_tool_callback(
        DummyTool(), {"provisional_booking_id": "book_abc123"}, DummyToolContext()
    )

    assert response is None


async def test_before_confirm_booking_tool_callback_ignores_other_tools():
    class DummyTool:
        name = "stage_provisional_booking"

    class DummyToolContext:
        state = {}

    response = await before_confirm_booking_tool_callback(DummyTool(), {}, DummyToolContext())

    assert response is None


# --- budget & grounding guardrails ------------------------------------------


def test_evaluate_budget_within_ceiling():
    result = evaluate_budget([100.0, 50.0], budget_ceiling=200.0)

    assert result.status is BudgetStatus.WITHIN_BUDGET
    assert result.total_cost == 150.0
    assert result.overage == 0.0


def test_evaluate_budget_exceeded_reports_overage():
    result = evaluate_budget([150.0, 100.0], budget_ceiling=200.0)

    assert result.status is BudgetStatus.EXCEEDED
    assert result.total_cost == 250.0
    assert result.overage == 50.0
    assert "exceeds" in result.message


def test_evaluate_budget_with_no_ceiling_set():
    result = evaluate_budget([100.0], budget_ceiling=None)

    assert result.status is BudgetStatus.NO_BUDGET_SET
    assert result.overage == 0.0


async def test_default_grounding_evaluator_passes_when_items_match():
    result = await default_grounding_evaluator(
        ["Louvre Museum", "eiffel tower summit"],
        ["Louvre Museum", "Eiffel Tower Summit"],
    )

    assert result.verdict is GroundingVerdict.GROUNDED
    assert result.ungrounded_items == []


async def test_default_grounding_evaluator_flags_ungrounded_items():
    result = await default_grounding_evaluator(
        ["Louvre Museum", "Made Up Castle Tour"],
        ["Louvre Museum", "Eiffel Tower Summit"],
    )

    assert result.verdict is GroundingVerdict.UNGROUNDED_ITEMS_FOUND
    assert result.ungrounded_items == ["Made Up Castle Tour"]


async def test_run_grounding_check_uses_injected_evaluator():
    async def always_grounded(itinerary_items, tool_output_items):
        from src.guardrails.budget_guardrail import GroundingCheckResult

        return GroundingCheckResult(verdict=GroundingVerdict.GROUNDED, message="stubbed")

    result = await run_grounding_check(["Anything"], [], evaluator=always_grounded)

    assert result.verdict is GroundingVerdict.GROUNDED
    assert result.message == "stubbed"
