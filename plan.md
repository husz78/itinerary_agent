# Implementation Plan: Smart Itinerary Planner & Booking Assistant (100% Local)

This implementation plan breaks down the development of the travel agent into discrete, testable units directly aligned with the **AI in 5 Days Assessment Agent Rubric (95/95 Points)** for a **100% local architecture** using the **Gemini 3 model family (`gemini-3.5-flash` and `gemini-3.1-pro`)**.

---

## Phase 1: Local Environment, Dependencies & Configuration
- [x] **Task 1.1: Project Setup & Dependency Configuration**
  - Initialize Python project with `pyproject.toml` (ADK, google-genai, structlog, opentelemetry, pydantic-settings, pytest, aiosqlite).
  - Create `.env.example` with local config variables (`GEMINI_API_KEY`, log levels, local DB path).
  - Create a `Makefile` with targets for running the agent, running tests, and executing evals locally.
- [x] **Task 1.2: Local Settings & Secret Management**
  - Implement `src/config.py` using `pydantic-settings` to securely load API keys and local configs from `.env` with strict `.gitignore` protection.

---

## Phase 2: Tool & Interface Design (20 pts Rubric Alignment)
- [x] **Task 2.1: Tool Schemas & Guided Error Framework**
  - Define standard `ToolResultEnvelope` and recovery error models in `src/tools/base.py`.
- [x] **Task 2.2: Weather Tool**
  - Implement `fetch_destination_weather_forecast` in `src/tools/weather_tool.py` with descriptive docstrings, Pydantic schemas, and structured recovery guidance.
- [x] **Task 2.3: Attractions & Places Search Tool**
  - Implement `search_attractions_and_activities` in `src/tools/attraction_tool.py`.
- [x] **Task 2.4: Transit & Route Estimation Tool**
  - Implement `calculate_transit_route_estimate` in `src/tools/transit_tool.py`.
- [x] **Task 2.5: Staging & Reservation Tool**
  - Implement `stage_provisional_booking` and `confirm_reservation_booking` in `src/tools/booking_tool.py`.
- [x] **Task 2.6: Tool Unit Tests**
  - Write test suite in `tests/test_tools.py` verifying schema validation, mock responses, and error recovery payloads.

---

## Phase 3: Observability, Tracing & PII Redaction (20 pts Rubric Alignment)
- [x] **Task 3.1: PII Redaction Engine**
  - Implement `src/observability/pii_scrubber.py` to scrub credit cards, passports, phone numbers, and emails before logging/storing.
- [x] **Task 3.2: Structured JSON Logging & Intent/Outcome Capture**
  - Implement `src/observability/logger.py` with structured JSON output and paired `AGENT_INTENT` / `AGENT_OUTCOME` event capture.
- [x] **Task 3.3: OpenTelemetry Distributed Tracing**
  - Implement `src/observability/tracer.py` with local Console / Memory / OTLP span management for agent tool invocations and LLM requests.
- [x] **Task 3.4: Observability Verification Tests**
  - Add `tests/test_observability.py` testing PII masking, trace generation, and structured log formatting.

---

## Phase 4: Context & Persistent Local Memory Architecture (20 pts Rubric Alignment)
- [x] **Task 4.1: Persistent Local Session Store (SQLite)**
  - Implement `src/memory/session_store.py` backed by SQLite (`data/travel_agent.db`) for multi-turn conversations and user profiles.
- [x] **Task 4.2: History Compaction & Summarization**
  - Implement sliding window token management and automatic summarization (ADK compaction) in `src/memory/compaction.py`.
- [x] **Task 4.3: Asynchronous Memory Worker**
  - Implement `src/memory/async_memory.py` using `asyncio.create_task` for background profile extraction and preference updates without blocking the UI.
- [x] **Task 4.4: Memory Unit Tests**
  - Add `tests/test_memory.py` verifying SQLite persistence, token windowing, and async extraction.

---

## Phase 5: Multi-Agent Orchestration, Model Routing & Guardrails (20 pts Rubric Alignment)
- [x] **Task 5.1: Agent Constitution & Base Prompts**
  - Define system prompts and domain constraints in `src/agents/constitution.py`.
- [x] **Task 5.2: Specialized Sub-Agents with Strategic Model Routing**
  - Implement `WeatherSpecialistAgent` (`gemini-3.5-flash`) in `src/agents/weather_specialist.py`.
  - Implement `AttractionSearchAgent` (`gemini-3.5-flash`) in `src/agents/attraction_search.py`.
  - Implement `BookingAgent` (`gemini-3.1-pro`) in `src/agents/booking_specialist.py`.
- [x] **Task 5.3: Travel Coordinator Agent**
  - Implement `TravelCoordinatorAgent` (`gemini-3.1-pro`) in `src/agents/coordinator.py` for complex multi-stop planning.
- [x] **Task 5.4: Human-in-the-Loop (HITL) Hooks**
  - Implement `src/guardrails/hitl_manager.py` gating final booking confirmation until the user explicitly authorizes.
- [x] **Task 5.5: Guardrails & Budget Policy Evaluator**
  - Implement `src/guardrails/budget_guardrail.py` and fast self-eval quality checks (`gemini-3.5-flash`).
- [x] **Task 5.6: Agent & Orchestration Tests**
  - Add `tests/test_agents.py` and `tests/test_routing.py` testing multi-agent handoffs, model routing, and HITL stops.

---

## Phase 6: Automated Evaluation Suite, CLI & Local Verification (15 pts Rubric Alignment)
- [x] **Task 6.1: Golden Dataset Definition**
  - Create `evals/golden_dataset.json` containing 5+ multi-stop travel scenarios with complex constraints.
- [x] **Task 6.2: Regression & Benchmark Test Runner**
  - Implement `evals/run_evals.py` measuring planning accuracy, budget adherence, model routing, and tool success rate.
- [x] **Task 6.3: Local CLI Entrypoint**
  - Implement `src/main.py` providing an interactive CLI experience for chatting with the agent.
- [ ] **Task 6.4: Full Assessment Verification**
  - Run the full test suite and golden evals, ensuring 100% compliance across all 19 rubric criteria for final submission.
