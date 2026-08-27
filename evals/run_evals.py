"""Deterministic regression & benchmark runner for the golden evaluation dataset.

Phase 6, Task 6.2. Replays `evals/golden_dataset.json` (Task 6.1) directly
against the real, local, no-network `src/tools/*`, `src/guardrails/*`, and
`src/agents/*` code -- no Gemini API calls are made -- and reports four
metrics aligned with `plan.md` / `spec.md`:

    - tool success rate: fraction of `expected_tool_calls` whose actual
      envelope (success or guided error) matches the golden expectation.
    - model routing accuracy: fraction of `expected_routing` steps whose
      `delegate_to` agent is actually wired to the stated Gemini model by
      the `src/agents/*` factories.
    - budget adherence: fraction of `expected_guardrails.budget_check`
      assertions where `src/guardrails/budget_guardrail.evaluate_budget`
      reproduces the golden status/total/overage.
    - HITL compliance: fraction of scenarios with a `confirm_reservation_booking`
      guardrail story where every such tool-call assertion passed, replayed
      through the real `before_confirm_booking_tool_callback` gate.

`success_criteria` and any `weather_safety_rule` entries describe the
*final synthesized itinerary*, which depends on live, non-deterministic LLM
output (see the dataset's own `description` field). They cannot be scored
without running the coordinator against a real model (Task 6.3's CLI), so
this runner surfaces them as manual-review items rather than faking a score.

Usage:
    uv run python evals/run_evals.py
    uv run python evals/run_evals.py --dataset evals/golden_dataset.json --report evals/results/latest.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

from src.agents.attraction_search import create_attraction_search_agent
from src.agents.booking_specialist import create_booking_agent
from src.agents.coordinator import create_travel_coordinator_agent
from src.agents.weather_specialist import create_weather_specialist_agent
from src.guardrails.budget_guardrail import evaluate_budget, run_grounding_check
from src.guardrails.hitl_manager import before_confirm_booking_tool_callback
from src.tools.attraction_tool import search_attractions_and_activities
from src.tools.booking_tool import confirm_reservation_booking, stage_provisional_booking
from src.tools.transit_tool import calculate_transit_route_estimate
from src.tools.weather_tool import fetch_destination_weather_forecast

DEFAULT_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
DEFAULT_REPORT_PATH = Path(__file__).parent / "results" / "latest_run.json"

# Tools that return a pydantic `ToolResultEnvelope` / `ToolErrorEnvelope`
# directly; dumped to a plain JSON-shaped dict for comparison against the
# golden dataset's `expected` payloads.
TOOL_REGISTRY = {
    "fetch_destination_weather_forecast": fetch_destination_weather_forecast,
    "search_attractions_and_activities": search_attractions_and_activities,
    "calculate_transit_route_estimate": calculate_transit_route_estimate,
    "stage_provisional_booking": stage_provisional_booking,
    "confirm_reservation_booking": confirm_reservation_booking,
}

CONFIRM_BOOKING_TOOL = "confirm_reservation_booking"

# Maps a `<from step N>` argument's key name to the field on step N's
# `data` payload it should be pulled from (booking id vs. its token).
STEP_REF_FIELD_BY_ARG = {
    "provisional_booking_id": "provisional_booking_id",
    "user_confirmation_token": "confirmation_token",
}

_STEP_REF_RE = re.compile(r"^<from step (\d+)>$")

# Dataset keys that annotate an assertion but aren't part of the tool's
# actual output contract, so they're skipped during comparison.
_ANNOTATION_ONLY_KEYS = {"source"}


class _FakeTool:
    """Minimal stand-in for an ADK `BaseTool`; only `.name` is read."""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeToolContext:
    """Minimal stand-in for an ADK `ToolContext`; only `.state` is read."""

    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state


def _resolve_step_refs(value: Any, step_results: dict[int, dict[str, Any]]) -> Any:
    """Recursively substitute `"<from step N>"` placeholders with a prior step's output.

    Used on both `args` (e.g. `confirm_reservation_booking`'s
    `provisional_booking_id`) and `expected` payloads (a confirmed
    reservation's `data.provisional_booking_id` golden value is itself
    `"<from step N>"`, since it must equal whatever id that step's
    `stage_provisional_booking` call actually generated).
    """
    if isinstance(value, dict):
        resolved = {}
        for key, sub_value in value.items():
            if isinstance(sub_value, str):
                match = _STEP_REF_RE.match(sub_value)
                if match:
                    ref_step = int(match.group(1))
                    field_name = STEP_REF_FIELD_BY_ARG.get(key, key)
                    resolved[key] = step_results[ref_step]["data"][field_name]
                    continue
            resolved[key] = _resolve_step_refs(sub_value, step_results)
        return resolved
    if isinstance(value, list):
        return [_resolve_step_refs(item, step_results) for item in value]
    return value


def _diff(expected: Any, actual: Any, path: str = "$") -> list[str]:
    """Recursively compare `actual` against `expected`, honoring dataset sentinels.

    Sentinels handled (see golden_dataset.json's `schema_notes`):
        - `"<generated>"`: the field is runtime-generated; only its sibling
          `<field>_pattern` regex is checked against the actual value.
        - `"<any>"`: the field is intentionally unconstrained.
        - `"message_contains"`: substring match against `actual["message"]`
          instead of full equality.
        - `_pattern`-suffixed keys and `_ANNOTATION_ONLY_KEYS`: metadata for
          the assertion itself, not part of the tool's output contract.
    """
    errors: list[str] = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path}: expected an object, got {type(actual).__name__}={actual!r}"]
        for key, expected_value in expected.items():
            if key in _ANNOTATION_ONLY_KEYS or key.endswith("_pattern"):
                continue
            if key == "message_contains":
                actual_message = actual.get("message", "")
                if expected_value not in actual_message:
                    errors.append(
                        f"{path}.message: expected to contain {expected_value!r}, "
                        f"got {actual_message!r}"
                    )
                continue
            if expected_value == "<generated>":
                actual_value = actual.get(key)
                pattern = expected.get(f"{key}_pattern")
                if actual_value is None:
                    errors.append(f"{path}.{key}: expected a generated value, got None")
                elif pattern and not re.match(pattern, str(actual_value)):
                    errors.append(
                        f"{path}.{key}: value {actual_value!r} does not match "
                        f"pattern {pattern!r}"
                    )
                continue
            if key not in actual:
                errors.append(f"{path}.{key}: missing from actual output")
                continue
            errors.extend(_diff(expected_value, actual[key], f"{path}.{key}"))
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            errors.append(
                f"{path}: expected a list of length {len(expected)}, got {actual!r}"
            )
        else:
            for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
                errors.extend(_diff(expected_item, actual_item, f"{path}[{index}]"))
    else:
        if expected != "<any>" and actual != expected:
            errors.append(f"{path}: expected {expected!r}, got {actual!r}")
    return errors


def _run_tool_call(
    tool_call: dict[str, Any],
    step_results: dict[int, dict[str, Any]],
    hitl_state: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Replay a single `expected_tool_calls` entry and diff it against `expected`."""
    tool_name = tool_call["tool"]
    args = _resolve_step_refs(tool_call["args"], step_results)
    preconditions = tool_call.get("preconditions")

    if tool_name == CONFIRM_BOOKING_TOOL and preconditions is not None:
        # This step is exercising the HITL gate, not just the raw tool -- replay
        # both layers exactly as BookingAgent's before_tool_callback wiring does.
        if preconditions.get("traveler_authorized_booking_first"):
            hitl_state.setdefault("authorized_booking_ids", [])
            authorized = set(hitl_state["authorized_booking_ids"])
            authorized.add(args["provisional_booking_id"])
            hitl_state["authorized_booking_ids"] = sorted(authorized)

        gate_result = asyncio.run(
            before_confirm_booking_tool_callback(
                _FakeTool(CONFIRM_BOOKING_TOOL), args, _FakeToolContext(hitl_state)
            )
        )
        actual = gate_result if gate_result is not None else TOOL_REGISTRY[tool_name](
            **args
        ).model_dump(mode="json")
    else:
        actual = TOOL_REGISTRY[tool_name](**args).model_dump(mode="json")

    expected = _resolve_step_refs(tool_call["expected"], step_results)
    errors = _diff(expected, actual, path=f"step {tool_call['step']} ({tool_name})")
    return actual, errors


def _build_agent_model_map() -> dict[str, str]:
    """Instantiate each `src/agents/*` factory once and record its wired model."""
    agents = [
        create_weather_specialist_agent(),
        create_attraction_search_agent(),
        create_booking_agent(),
        create_travel_coordinator_agent(),
    ]
    return {agent.name: agent.model for agent in agents}


def _evaluate_routing(
    expected_routing: list[dict[str, Any]], agent_model_map: dict[str, str]
) -> list[dict[str, Any]]:
    results = []
    for step in expected_routing:
        delegate_to = step["delegate_to"]
        actual_model = agent_model_map.get(delegate_to)
        passed = actual_model == step["model"]
        results.append(
            {
                "step": step["step"],
                "delegate_to": delegate_to,
                "expected_model": step["model"],
                "actual_model": actual_model,
                "passed": passed,
            }
        )
    return results


def _evaluate_budget_check(budget_check: dict[str, Any]) -> dict[str, Any]:
    result = evaluate_budget(budget_check["line_items_usd"], budget_check["budget_ceiling_usd"])
    errors = []
    if result.status.value != budget_check["expected_status"]:
        errors.append(
            f"status: expected {budget_check['expected_status']!r}, got {result.status.value!r}"
        )
    if result.total_cost != budget_check["expected_total_cost"]:
        errors.append(
            f"total_cost: expected {budget_check['expected_total_cost']}, got {result.total_cost}"
        )
    if result.overage != budget_check["expected_overage"]:
        errors.append(f"overage: expected {budget_check['expected_overage']}, got {result.overage}")
    return {"passed": not errors, "errors": errors, "actual": result.model_dump(mode="json")}


def _evaluate_grounding_check(grounding_check: dict[str, Any]) -> dict[str, Any]:
    result = asyncio.run(
        run_grounding_check(
            grounding_check["itinerary_items"], grounding_check["tool_output_items"]
        )
    )
    passed = result.verdict.value == grounding_check["expected_verdict"]
    errors = (
        []
        if passed
        else [f"verdict: expected {grounding_check['expected_verdict']!r}, got {result.verdict.value!r}"]
    )
    return {"passed": passed, "errors": errors, "actual": result.model_dump(mode="json")}


def run_scenario(scenario: dict[str, Any], agent_model_map: dict[str, str]) -> dict[str, Any]:
    """Replay one golden scenario's tool calls, routing, and guardrails."""
    hitl_state: dict[str, Any] = {}
    step_results: dict[int, dict[str, Any]] = {}

    tool_call_results = []
    for tool_call in scenario.get("expected_tool_calls", []):
        actual, errors = _run_tool_call(tool_call, step_results, hitl_state)
        step_results[tool_call["step"]] = actual
        tool_call_results.append(
            {
                "step": tool_call["step"],
                "tool": tool_call["tool"],
                "passed": not errors,
                "errors": errors,
            }
        )

    routing_results = _evaluate_routing(scenario.get("expected_routing", []), agent_model_map)

    guardrails = scenario.get("expected_guardrails", {})
    budget_result = (
        _evaluate_budget_check(guardrails["budget_check"]) if "budget_check" in guardrails else None
    )
    grounding_result = (
        _evaluate_grounding_check(guardrails["grounding_check"])
        if "grounding_check" in guardrails
        else None
    )

    hitl_guardrail = guardrails.get("hitl")
    hitl_result = None
    if hitl_guardrail is not None:
        confirm_steps = [r for r in tool_call_results if r["tool"] == CONFIRM_BOOKING_TOOL]
        errors = [f"step {r['step']}: {e}" for r in confirm_steps if not r["passed"] for e in r["errors"]]
        expected_call_count = hitl_guardrail.get("expected_calls_to_confirm_reservation_booking")
        if expected_call_count is not None and len(confirm_steps) != expected_call_count:
            errors.append(
                f"expected {expected_call_count} confirm_reservation_booking call(s), "
                f"dataset declares {len(confirm_steps)}"
            )
        hitl_result = {"passed": not errors, "errors": errors}

    manual_review_items = list(scenario.get("success_criteria", []))
    weather_safety_rule = guardrails.get("weather_safety_rule")
    if weather_safety_rule is not None:
        manual_review_items.append(
            "weather_safety_rule requires live scheduling output -- see expected_guardrails.weather_safety_rule"
        )

    return {
        "scenario_id": scenario["scenario_id"],
        "title": scenario["title"],
        "tool_calls": tool_call_results,
        "routing": routing_results,
        "budget_check": budget_result,
        "grounding_check": grounding_result,
        "hitl": hitl_result,
        "known_limitations": scenario.get("known_limitations", []),
        "manual_review_items": manual_review_items,
    }


def summarize(scenario_reports: list[dict[str, Any]]) -> dict[str, Any]:
    def _rate(passed: int, total: int) -> float | None:
        return round(passed / total, 4) if total else None

    tool_calls = [tc for report in scenario_reports for tc in report["tool_calls"]]
    routing_steps = [rs for report in scenario_reports for rs in report["routing"]]
    budget_checks = [report["budget_check"] for report in scenario_reports if report["budget_check"]]
    grounding_checks = [
        report["grounding_check"] for report in scenario_reports if report["grounding_check"]
    ]
    hitl_checks = [report["hitl"] for report in scenario_reports if report["hitl"]]

    return {
        "tool_success_rate": _rate(sum(tc["passed"] for tc in tool_calls), len(tool_calls)),
        "tool_call_count": len(tool_calls),
        "model_routing_accuracy": _rate(sum(rs["passed"] for rs in routing_steps), len(routing_steps)),
        "routing_step_count": len(routing_steps),
        "budget_adherence_rate": _rate(
            sum(bc["passed"] for bc in budget_checks), len(budget_checks)
        ),
        "budget_check_count": len(budget_checks),
        "grounding_pass_rate": _rate(
            sum(gc["passed"] for gc in grounding_checks), len(grounding_checks)
        ),
        "grounding_check_count": len(grounding_checks),
        "hitl_compliance_rate": _rate(sum(hc["passed"] for hc in hitl_checks), len(hitl_checks)),
        "hitl_check_count": len(hitl_checks),
        "note": (
            "Planning-accuracy (success_criteria) and token/latency efficiency require "
            "a live model-backed coordinator run (see src/main.py, Task 6.3) and are "
            "not scored by this deterministic pass; see manual_review_items per scenario."
        ),
    }


def _print_report(scenario_reports: list[dict[str, Any]], metrics: dict[str, Any]) -> bool:
    all_passed = True
    for report in scenario_reports:
        failures = (
            [tc for tc in report["tool_calls"] if not tc["passed"]]
            + [rs for rs in report["routing"] if not rs["passed"]]
        )
        if report["budget_check"] and not report["budget_check"]["passed"]:
            failures.append(report["budget_check"])
        if report["grounding_check"] and not report["grounding_check"]["passed"]:
            failures.append(report["grounding_check"])
        if report["hitl"] and not report["hitl"]["passed"]:
            failures.append(report["hitl"])

        status = "PASS" if not failures else "FAIL"
        if failures:
            all_passed = False
        print(f"[{status}] {report['scenario_id']} -- {report['title']}")
        for failure in failures:
            for error in failure.get("errors", []):
                print(f"    - {error}")
        if report["known_limitations"]:
            for limitation in report["known_limitations"]:
                print(f"    (known limitation, not scored as a defect: {limitation[:120]}...)")

    print("\n--- Aggregate metrics ---")
    for key in (
        "tool_success_rate",
        "model_routing_accuracy",
        "budget_adherence_rate",
        "grounding_pass_rate",
        "hitl_compliance_rate",
    ):
        value = metrics[key]
        print(f"{key}: {value if value is not None else 'n/a'}")
    print(f"\n{metrics['note']}")
    return all_passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    dataset = json.loads(args.dataset.read_text())
    agent_model_map = _build_agent_model_map()

    scenario_reports = [
        run_scenario(scenario, agent_model_map) for scenario in dataset["scenarios"]
    ]
    metrics = summarize(scenario_reports)
    all_passed = _print_report(scenario_reports, metrics)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps({"metrics": metrics, "scenarios": scenario_reports}, indent=2, default=str)
    )
    print(f"\nFull report written to {args.report}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
