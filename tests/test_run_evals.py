"""Tests for the golden-dataset regression runner (Phase 6, Task 6.2)."""

import json

import pytest

from evals.run_evals import (
    DEFAULT_DATASET_PATH,
    _build_agent_model_map,
    _diff,
    _resolve_step_refs,
    run_scenario,
    summarize,
)


@pytest.fixture(scope="module")
def golden_dataset():
    return json.loads(DEFAULT_DATASET_PATH.read_text())


@pytest.fixture(scope="module")
def agent_model_map():
    return _build_agent_model_map()


# --- _diff --------------------------------------------------------------


def test_diff_reports_no_errors_for_matching_payloads():
    assert _diff({"status": "success", "data": {"a": 1}}, {"status": "success", "data": {"a": 1}}) == []


def test_diff_reports_mismatched_scalar():
    errors = _diff({"status": "success"}, {"status": "error"})
    assert len(errors) == 1
    assert "status" in errors[0]


def test_diff_checks_generated_field_against_its_pattern():
    expected = {"token": "<generated>", "token_pattern": "^[0-9A-F]{8}$"}
    assert _diff(expected, {"token": "ABCD1234"}) == []
    assert _diff(expected, {"token": "not-hex"}) != []


def test_diff_message_contains_is_substring_match():
    expected = {"message_contains": "has already been confirmed"}
    assert _diff(expected, {"message": "Booking 'x' has already been confirmed."}) == []
    assert _diff(expected, {"message": "something else"}) != []


def test_diff_ignores_annotation_only_keys():
    expected = {"status": "error", "source": "some docstring pointer"}
    assert _diff(expected, {"status": "error"}) == []


def test_diff_any_sentinel_matches_anything():
    assert _diff({"field": "<any>"}, {"field": "literally anything"}) == []


# --- _resolve_step_refs ---------------------------------------------------


def test_resolve_step_refs_substitutes_booking_id_and_token():
    step_results = {7: {"data": {"provisional_booking_id": "book_abc", "confirmation_token": "TOK1"}}}
    resolved = _resolve_step_refs(
        {"provisional_booking_id": "<from step 7>", "user_confirmation_token": "<from step 7>"},
        step_results,
    )
    assert resolved == {"provisional_booking_id": "book_abc", "user_confirmation_token": "TOK1"}


def test_resolve_step_refs_leaves_non_reference_values_untouched():
    resolved = _resolve_step_refs({"user_confirmation_token": "WRONGTOK"}, {})
    assert resolved == {"user_confirmation_token": "WRONGTOK"}


def test_resolve_step_refs_recurses_into_nested_expected_payloads():
    step_results = {3: {"data": {"provisional_booking_id": "book_xyz"}}}
    resolved = _resolve_step_refs(
        {"status": "success", "data": {"provisional_booking_id": "<from step 3>"}}, step_results
    )
    assert resolved["data"]["provisional_booking_id"] == "book_xyz"


# --- run_scenario / full dataset replay ----------------------------------


def test_every_golden_scenario_replays_clean(golden_dataset, agent_model_map):
    for scenario in golden_dataset["scenarios"]:
        report = run_scenario(scenario, agent_model_map)
        tool_call_failures = [tc for tc in report["tool_calls"] if not tc["passed"]]
        routing_failures = [r for r in report["routing"] if not r["passed"]]
        assert not tool_call_failures, f"{scenario['scenario_id']}: {tool_call_failures}"
        assert not routing_failures, f"{scenario['scenario_id']}: {routing_failures}"
        if report["budget_check"] is not None:
            assert report["budget_check"]["passed"], report["budget_check"]["errors"]
        if report["grounding_check"] is not None:
            assert report["grounding_check"]["passed"], report["grounding_check"]["errors"]
        if report["hitl"] is not None:
            assert report["hitl"]["passed"], report["hitl"]["errors"]


def test_run_scenario_detects_a_genuine_tool_regression(golden_dataset, agent_model_map):
    scenario = next(
        s for s in golden_dataset["scenarios"] if s["scenario_id"] == "tokyo_luxury_over_budget"
    )
    tampered = json.loads(json.dumps(scenario))
    tampered["expected_tool_calls"][0]["expected"]["data"]["resolved_location"] = "Nowhere, Nowhere"

    report = run_scenario(tampered, agent_model_map)

    assert any(not tc["passed"] for tc in report["tool_calls"])


def test_run_scenario_detects_a_routing_regression(golden_dataset, agent_model_map):
    scenario = next(
        s for s in golden_dataset["scenarios"] if s["scenario_id"] == "tokyo_luxury_over_budget"
    )
    tampered = json.loads(json.dumps(scenario))
    tampered["expected_routing"][0]["model"] = "gemini-3.1-pro"

    report = run_scenario(tampered, agent_model_map)

    assert any(not r["passed"] for r in report["routing"])


def test_run_scenario_detects_a_budget_regression(golden_dataset, agent_model_map):
    scenario = next(
        s for s in golden_dataset["scenarios"] if s["scenario_id"] == "tokyo_luxury_over_budget"
    )
    tampered = json.loads(json.dumps(scenario))
    tampered["expected_guardrails"]["budget_check"]["expected_status"] = "within_budget"

    report = run_scenario(tampered, agent_model_map)

    assert report["budget_check"]["passed"] is False


# --- summarize -------------------------------------------------------------


def test_summarize_computes_pass_rates_across_scenarios(golden_dataset, agent_model_map):
    reports = [run_scenario(s, agent_model_map) for s in golden_dataset["scenarios"]]
    metrics = summarize(reports)

    assert metrics["tool_success_rate"] == 1.0
    assert metrics["model_routing_accuracy"] == 1.0
    assert metrics["budget_adherence_rate"] == 1.0
    assert metrics["grounding_pass_rate"] == 1.0
    assert metrics["hitl_compliance_rate"] == 1.0
    assert metrics["tool_call_count"] > 0
