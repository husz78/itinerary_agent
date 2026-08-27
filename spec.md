# Feature Specification: Smart Itinerary Planner & Booking Assistant (100% Local)

## 1. Project Overview & Objective
The **Smart Itinerary Planner & Booking Assistant** is a production-grade, multi-agent travel concierge built on the **Google Agent Development Kit (ADK)**. It ingests user travel preferences, budget, constraints, and dates to plan complex multi-stop itineraries, query live/mocked weather conditions, find local attractions/activities, and coordinate provisional and confirmed booking reservations.

The entire system is designed for **100% local development and execution** (using local SQLite database, local `.env` configuration, local OpenTelemetry exporter, and local evaluation harness) while fulfilling the **AI in 5 Days Assessment Agent Rubric (95/95 points)**.

---

## 2. System Architecture & Multi-Agent Design

### Multi-Agent Hierarchy (Coordinator-Specialist Pattern)
```
                          ┌───────────────────────────────┐
                          │     TravelCoordinatorAgent    │
                          │        (gemini-3.1-pro)       │
                          └──────────────┬────────────────┘
                                         │
                 ┌───────────────────────┼───────────────────────┐
                 │                       │                       │
                 ▼                       ▼                       ▼
    ┌─────────────────────────┐ ┌───────────────────┐ ┌───────────────────────┐
    │  WeatherSpecialistAgent │ │ AttractionSearch  │ │     BookingAgent      │
    │    (gemini-3.5-flash)   │ │(gemini-3.5-flash) │ │(gemini-3.1-pro + HITL)│
    └─────────────────────────┘ └───────────────────┘ └───────────────────────┘
```

### Strategic Model Routing (Gemini 3 Family)
| Agent / Component | Model | Rationale |
| :--- | :--- | :--- |
| **TravelCoordinatorAgent** | `gemini-3.1-pro` | Multi-day constraint resolution, complex route synthesis, budget optimization |
| **WeatherSpecialistAgent** | `gemini-3.5-flash` | Fast, low-latency forecast extraction and weather sanity checks |
| **AttractionSearchAgent** | `gemini-3.5-flash` | High-throughput search filtering, opening hour verification, tag classification |
| **BookingAgent** | `gemini-3.1-pro` | Precise price calculation, schema-strict payload creation, confirmation gating |
| **Evaluator / Guardrail** | `gemini-3.5-flash` | Fast self-evaluation of output quality, PII scrubbing, safety policy enforcement |

---

## 3. Rubric-Aligned Technical Requirements

### A. Tool & Interface Design (20 Points)
1. **Descriptive Naming**:
   - `fetch_destination_weather_forecast(location, start_date, end_date)`
   - `search_attractions_and_activities(city, category, budget_tier, duration_hours)`
   - `calculate_transit_route_estimate(origin, destination, travel_mode)`
   - `stage_provisional_booking(reservation_type, provider_id, slot, price)`
   - `confirm_reservation_booking(provisional_booking_id, user_confirmation_token)`
2. **Comprehensive Docstrings**: Every tool function has Google-style docstrings specifying parameter types, return schemas, failure modes, and usage examples.
3. **Explicit JSON Schemas**: All inputs and outputs inherit from `pydantic.BaseModel` with `Field(..., description=...)` and strict type validation.
4. **Guided Error Handling**: Tools return structured recovery envelopes:
   ```json
   {
     "status": "error",
     "error_code": "LOCATION_AMBIGUOUS",
     "message": "Multiple matches found for 'Paris' (France, Texas, Ontario).",
     "recovery_instruction": "Specify country or state in the location parameter or invoke search_attractions_and_activities with ISO country code."
   }
   ```

### B. Context & Memory Management (20 Points)
1. **Robust System Instructions ("Constitution")**:
   - Explicit persona: Professional, detail-oriented concierge.
   - Non-negotiable safety rules: No unauthorized bookings, zero hallucinated flight times, mandatory weather checks for outdoor activities.
2. **History Compaction**:
   - Sliding window for the last $N$ turns.
   - Automatic background LLM-driven conversation summarization (ADK Compaction) once prompt exceeds token threshold.
3. **Persistent Local Session State**:
   - Local SQLite database (`data/travel_agent.db`) backing conversation history, itinerary drafts, and traveler preferences across sessions.
4. **Async Memory Operations**:
   - Traveler profile updates and long-term memory extraction (dietary preferences, preferred airlines, pacing) run as detached background tasks (`asyncio.create_task`).

### C. Orchestration & Logic (20 Points)
1. **Multi-Agent Orchestration**: ADK Coordinator managing specialized sub-agents via explicit task delegation.
2. **Human-in-the-Loop (HITL) Hooks**:
   - High-stakes action (`confirm_reservation_booking`) requires explicit user confirmation via an authorization token or confirmation prompt before executing booking state changes.
3. **Guardrails & Evaluation Policies**:
   - Budget constraint validation filter (verifies total estimated cost <= user budget).
   - Hallucination check comparing final itinerary items with tool search outputs.

### D. Observability & Tracing (20 Points)
1. **Structured JSON Logging**: Standard JSON logs with `timestamp`, `level`, `session_id`, `trace_id`, `agent_name`, `model`, `latency_ms`, and `tokens`.
2. **Intent vs. Outcome Capture**:
   - `EVENT: AGENT_INTENT` -> Emitted when agent decides to call a tool or plan a step.
   - `EVENT: AGENT_OUTCOME` -> Emitted after tool returns with success/failure metadata.
3. **Distributed Tracing**: OpenTelemetry (OTel) instrumentation with local Console/Memory/OTLP exporters linking requests across sub-agents and tool calls.
4. **PII Redaction Engine**: Automated regex/DLP scrubbing middleware that masks credit cards, passport numbers, email addresses, and phone numbers in all logs, traces, and stored state.

### E. Local Infrastructure, CI/CD & Security (15 Points)
1. **Automated Evaluation Suite**:
   - Golden test dataset (`evals/golden_dataset.json`) with multi-city trip scenarios.
   - Deterministic test harness (`evals/run_evals.py`) measuring itinerary feasibility, budget compliance, model routing accuracy, and token efficiency.
2. **Local Reproducibility**:
   - Makefile and local CLI runner (`src/main.py`) for easy testing and evaluation.
3. **Secure Secret Management**:
   - Clean environment variable loading via `pydantic-settings` from local `.env` with `.gitignore` ensuring zero secret leaks or hardcoded credentials.

---

## 4. Directory Structure

```text
smart-itinerary-agent/
├── CLAUDE.md                    # Project-level rules and rubric requirements (Local)
├── spec.md                      # Feature & architecture specification
├── plan.md                      # Step-by-step implementation plan & checklist
├── pyproject.toml               # Poetry/pip dependency definitions
├── Makefile                     # Local run, test, and eval commands
├── .env.example                 # Local environment template (GEMINI_API_KEY)
├── data/                        # Local SQLite database & memory persistence
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── config.py                # Pydantic Settings (.env loader)
│   ├── main.py                  # Local interactive CLI / execution entrypoint
│   ├── agents/                  # Multi-agent definitions (Gemini 3 Family)
│   │   ├── constitution.py      # System prompts & safety rules
│   │   ├── coordinator.py       # TravelCoordinatorAgent (gemini-3.1-pro)
│   │   ├── weather_specialist.py# WeatherSpecialistAgent (gemini-3.5-flash)
│   │   ├── attraction_search.py # AttractionSearchAgent (gemini-3.5-flash)
│   │   └── booking_specialist.py# BookingAgent (gemini-3.1-pro + HITL)
│   ├── tools/                   # Strictly typed tools with recovery guidance
│   │   ├── base.py              # ToolResultEnvelope & error models
│   │   ├── weather_tool.py
│   │   ├── attraction_tool.py
│   │   ├── transit_tool.py
│   │   └── booking_tool.py
│   ├── memory/                  # Local SQLite session management & compaction
│   │   ├── session_store.py
│   │   ├── compaction.py
│   │   └── async_memory.py
│   ├── guardrails/              # HITL, budget & safety policies
│   │   ├── hitl_manager.py
│   │   └── budget_guardrail.py
│   └── observability/           # OTel, JSON logging, PII scrubber
│       ├── logger.py
│       ├── tracer.py
│       └── pii_scrubber.py
├── evals/                       # Automated Golden Dataset test harness
│   ├── golden_dataset.json
│   └── run_evals.py
└── tests/                       # Unit and integration test suites
    ├── test_tools.py
    ├── test_agents.py
    ├── test_routing.py
    ├── test_memory.py
    └── test_observability.py
```

---

## 5. Definition of Done (DoD)
1. Complete local execution without any cloud provisioning.
2. Full test coverage on all tools, agents, memory compaction, and guardrails via `pytest`.
3. Golden evaluation script passes >= 90% benchmark on travel planning scenarios.
4. Zero hardcoded secrets (validated by static checks and strict `.env` loading).
5. Full OTel local traces and structured JSON logs captured for multi-turn sessions.
6. All 19 rubric criteria from the AI in 5 Days Assessment matrix verified and demonstrably satisfied.
