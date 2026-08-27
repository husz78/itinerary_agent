"""Guardrails: budget policy evaluator and fast self-eval grounding checks.

Two independent guardrails run before a final itinerary reaches the
traveler:
    - `evaluate_budget`: deterministic arithmetic check that the itinerary's
      total estimated cost does not exceed the traveler's stated budget
      ceiling.
    - `run_grounding_check`: a hallucination guardrail comparing the item
      names an agent's final itinerary claims to include against the item
      names actually returned by tool calls this session, per the project's
      "Guardrails & Evaluation Policies" requirement.

Both are plain, deterministic functions with no network dependency, so they
run instantly as a pre-response check. `run_grounding_check` additionally
accepts a `GroundingEvaluatorFn` hook: production callers can supply an
LLM-backed self-eval routed to `Settings.model_fast` (`gemini-3.5-flash`,
per this project's fast-task model routing) for fuzzier matching
(paraphrases, partial names) while `default_grounding_evaluator` stays a
deterministic, no-API-call fallback for local tests and evals -- the same
injectable-fallback pattern `memory/compaction.py` and
`memory/async_memory.py` use for summarization/preference extraction.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Awaitable, Callable

from pydantic import BaseModel, Field


class BudgetStatus(StrEnum):
    """Outcome of a budget compliance check."""

    WITHIN_BUDGET = "within_budget"
    EXCEEDED = "exceeded"
    NO_BUDGET_SET = "no_budget_set"


class BudgetCheckResult(BaseModel):
    """Result of comparing an itinerary's total cost against a budget ceiling.

    Attributes:
        status: Whether the total is within budget, over budget, or no
            ceiling was set to check against.
        total_cost: Sum of all supplied line items, rounded to cents.
        budget_ceiling: The ceiling checked against, or `None` if unset.
        overage: How much `total_cost` exceeds `budget_ceiling` by, 0 if
            within budget or no ceiling was set.
        message: Human/agent-readable explanation of the result.
    """

    status: BudgetStatus = Field(..., description="Outcome of the budget check.")
    total_cost: float = Field(..., ge=0, description="Sum of all line items, rounded to cents.")
    budget_ceiling: float | None = Field(
        default=None, description="Budget ceiling checked against, if any."
    )
    overage: float = Field(
        default=0.0, ge=0, description="Amount total_cost exceeds budget_ceiling by."
    )
    message: str = Field(..., description="Explanation of the result.")


def evaluate_budget(
    line_items: list[float], budget_ceiling: float | None
) -> BudgetCheckResult:
    """Check whether the sum of itinerary line items fits within a budget ceiling.

    Args:
        line_items: Cost of every itinerary line item (attractions, transit,
            staged bookings) in US dollars.
        budget_ceiling: The traveler's stated total budget ceiling in US
            dollars, or `None` if the traveler has not stated one yet.

    Returns:
        `BudgetCheckResult` with `status=NO_BUDGET_SET` if `budget_ceiling`
        is `None`, `status=WITHIN_BUDGET` if the total is at or under the
        ceiling, or `status=EXCEEDED` (with `overage` set) otherwise.
    """
    total_cost = round(sum(line_items), 2)

    if budget_ceiling is None:
        return BudgetCheckResult(
            status=BudgetStatus.NO_BUDGET_SET,
            total_cost=total_cost,
            budget_ceiling=None,
            message="No budget ceiling has been set; skipping the budget check.",
        )

    if total_cost <= budget_ceiling:
        return BudgetCheckResult(
            status=BudgetStatus.WITHIN_BUDGET,
            total_cost=total_cost,
            budget_ceiling=budget_ceiling,
            message=f"Total cost ${total_cost:.2f} is within the ${budget_ceiling:.2f} budget.",
        )

    overage = round(total_cost - budget_ceiling, 2)
    return BudgetCheckResult(
        status=BudgetStatus.EXCEEDED,
        total_cost=total_cost,
        budget_ceiling=budget_ceiling,
        overage=overage,
        message=(
            f"Total cost ${total_cost:.2f} exceeds the ${budget_ceiling:.2f} "
            f"budget by ${overage:.2f}. Remove or downgrade line items, or "
            "ask the traveler to confirm a higher budget before proceeding."
        ),
    )


class GroundingVerdict(StrEnum):
    """Outcome of a hallucination/grounding check."""

    GROUNDED = "grounded"
    UNGROUNDED_ITEMS_FOUND = "ungrounded_items_found"


class GroundingCheckResult(BaseModel):
    """Result of comparing itinerary items against tool search output.

    Attributes:
        verdict: Whether every itinerary item was grounded in tool output.
        ungrounded_items: Itinerary item names with no matching tool result;
            empty when `verdict=GROUNDED`.
        message: Human/agent-readable explanation of the result.
    """

    verdict: GroundingVerdict = Field(..., description="Whether the itinerary is fully grounded.")
    ungrounded_items: list[str] = Field(
        default_factory=list, description="Itinerary items with no matching tool result."
    )
    message: str = Field(..., description="Explanation of the result.")


GroundingEvaluatorFn = Callable[[list[str], list[str]], Awaitable[GroundingCheckResult]]


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


async def default_grounding_evaluator(
    itinerary_items: list[str], tool_output_items: list[str]
) -> GroundingCheckResult:
    """Deterministic fallback grounding check: case/whitespace-insensitive membership.

    Used for local testing and as a safe fallback; production callers
    should supply a `GroundingEvaluatorFn` routed to `gemini-3.5-flash` to
    catch paraphrased or partially-matching hallucinations this exact-match
    heuristic would miss.

    Args:
        itinerary_items: Item names (attractions, flights, hotels, etc.) the
            final itinerary claims to include.
        tool_output_items: Item names actually returned by tool calls this
            session (e.g. `Attraction.name`, a staged booking's
            `provider_id`).

    Returns:
        `GroundingCheckResult` with `verdict=GROUNDED` if every itinerary
        item normalizes to a match in `tool_output_items`, otherwise
        `verdict=UNGROUNDED_ITEMS_FOUND` listing the unmatched items.
    """
    known = {_normalize(item) for item in tool_output_items}
    ungrounded = [item for item in itinerary_items if _normalize(item) not in known]

    if not ungrounded:
        return GroundingCheckResult(
            verdict=GroundingVerdict.GROUNDED,
            message="Every itinerary item matches a tool search result.",
        )

    return GroundingCheckResult(
        verdict=GroundingVerdict.UNGROUNDED_ITEMS_FOUND,
        ungrounded_items=ungrounded,
        message=(
            f"{len(ungrounded)} itinerary item(s) do not match any tool "
            f"search result and may be hallucinated: {', '.join(ungrounded)}. "
            "Re-verify with the appropriate search tool or remove them "
            "before presenting the itinerary."
        ),
    )


async def run_grounding_check(
    itinerary_items: list[str],
    tool_output_items: list[str],
    evaluator: GroundingEvaluatorFn | None = None,
) -> GroundingCheckResult:
    """Run the hallucination/grounding guardrail, defaulting to the local heuristic.

    Args:
        itinerary_items: Item names the final itinerary claims to include.
        tool_output_items: Item names actually returned by tool calls this
            session.
        evaluator: Async callable performing the comparison. Defaults to
            `default_grounding_evaluator`.

    Returns:
        The `GroundingCheckResult` produced by `evaluator`.
    """
    evaluator = evaluator or default_grounding_evaluator
    return await evaluator(itinerary_items, tool_output_items)
