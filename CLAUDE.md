# Smart Itinerary Planner & Booking Assistant (Travel Agent)
## Project Context & AI-in-5-Days Rubric Alignment (Max Score: 95/95 - 100% Local Execution)

You are Claude Code working on the **Smart Itinerary Planner & Booking Assistant** project. This project is built with the **Google Agent Development Kit (ADK)** and runs **100% locally** (no cloud provisioning, no Terraform, no remote database/secret manager). It is designed to achieve a perfect 95/95 on the **AI in 5 Days Assessment Agent** grading rubric.

---

### Core Architectural Rules & Standards

#### 1. Tool & Interface Design (Rubric: 20/20)
- **Descriptive Naming**: Use hyper-specific, intent-revealing tool names (e.g., `fetch_destination_weather_forecast`, `search_places_and_attractions`, `confirm_reservation_booking`).
- **Comprehensive Docstrings**: Every tool function MUST have full docstrings detailing purpose, input arguments, types, edge cases, and return format.
- **Strict JSON Schemas**: Use `pydantic.BaseModel` for both input parameters and structured output schemas.
- **Guided Error Handling**: When a tool fails (API error, rate limit, missing location), return structured, actionable recovery guidance to the LLM (e.g., `{"status": "error", "message": "Location 'X' ambiguous", "suggested_action": "Ask user for state/country or retry with search_places"}`) rather than raising unhandled exceptions.

#### 2. Context & Memory Architecture (Rubric: 20/20)
- **Agent Constitution**: Maintain strict system instructions defining persona, domain scope (travel/itinerary planning), tone, and non-negotiable safety/operational constraints.
- **History Compaction**: Implement active token management (sliding window + automatic LLM-based summarization / ADK compaction) to prevent context bloat.
- **Persistent Local Session State**: Persist conversation turns, user travel preferences, and itinerary drafts using a local **SQLite** database (`data/travel_agent.db`) or local JSON storage.
- **Async Memory Operations**: Run memory consolidation and background user profile preference extraction asynchronously (`asyncio.create_task`) so conversational response latency is never blocked.

#### 3. Multi-Agent Orchestration & Model Routing (Rubric: 20/20)
- **Multi-Agent Pattern**: Implement a **Coordinator-Specialist** pattern using Google ADK:
  - `TravelCoordinatorAgent`: Primary interaction, state maintenance, user communication.
  - `WeatherSpecialistAgent`: Quick weather retrieval and forecast processing.
  - `AttractionSearchAgent`: Places, events, opening hours, local search.
  - `BookingAgent`: Reservation assembly, payment calculation, booking payload generation.
- **Strategic Model Routing (Gemini 3 Family)**:
  - `gemini-3.5-flash`: For high-throughput, low-latency sub-tasks (weather queries, tool parameter extraction, JSON formatting, intent classification, fast self-eval guardrails).
  - `gemini-3.1-pro`: For complex multi-stop constraint satisfaction, day-by-day scheduling, budget optimization, booking logic, and final synthesis.
- **Human-in-the-Loop (HITL) Hooks**: High-stakes actions (final flight/hotel reservations, credit card/points charges) MUST halt execution and require explicit user authorization before invoking the booking backend.
- **Guardrails & Policy Plugins**: Implement input sanitization, hallucination guardrails, and post-generation policy validation before returning responses to the user.

#### 4. Observability, Logging & Tracing (Rubric: 20/20)
- **Structured JSON Logging**: Use `structlog` or `python-json-logger` emitting standard JSON logs to stdout/local file with timestamps, `session_id`, `turn_id`, `model_name`, `token_counts`, and `latency_ms`.
- **Intent vs. Outcome Logging**: For every agent action, emit two paired log events:
  1. `AGENT_INTENT`: What the agent decided to execute and why.
  2. `AGENT_OUTCOME`: The verified tool result or execution status.
- **Distributed Tracing**: Instrument OpenTelemetry spans (OTel) with local console / memory / OTLP exporters linking incoming user queries through sub-agent calls, model requests, and tool invocations.
- **PII Redaction**: Scrub sensitive user data (passports, credit card numbers, emails, phone numbers) before emitting to logs, traces, or local memory.

#### 5. Local Infrastructure, CI/CD & Security (Rubric: 15/15)
- **Secure Secret Management**: NEVER hardcode API keys or credentials. Load all secrets strictly from local environment variables via `.env` and `pydantic-settings` (with `.gitignore` configured to ensure zero credential leaks).
- **Local Reproducibility & CLI**: Provide complete local setup automation (Makefile / local launch scripts / Dockerfile) for running the agent CLI or web UI locally.
- **Automated Evaluation Suite**: Provide an automated evaluation suite (`evals/`) executing golden test datasets (multi-day itinerary scenarios, budget limits, edge cases) to statically measure agent performance and regression locally.

---

### Spec-Driven Development Workflow

1. **Check Spec**: Always read `spec.md` and `plan.md` before starting any task.
2. **One Task at a Time**: Implement one modular component at a time according to `plan.md`.
3. **Verify with Tests**: Write and execute unit/integration tests with `pytest` before checking off any plan item.
4. **Update Progress**: Mark completed tasks in `plan.md` upon passing tests.

### Github
Push your changes to github. The repository is here: git@github.com:husz78/itinerary_agent.git
Use ssh since you have ssh keys set up.

### Google ADK Skills
Use the available `google-agents-cli-*` skills whenever they apply instead of writing ADK patterns from memory:
- **google-agents-cli-workflow**: entrypoint for the ADK dev lifecycle (build, run locally, debug, test, deploy) — always active as the default guide for agent development.
- **google-agents-cli-adk-code**: agent/tool/callback definitions, orchestration patterns, state management — use when writing agent or tool code in `src/agents/` and `src/tools/`.
- **google-agents-cli-scaffold**: project scaffolding, `scaffold enhance`/`upgrade`, CI/CD/deployment additions.
- **google-agents-cli-eval**: eval dataset design, LLM-as-judge scoring, failure analysis — use for `evals/`.
- **google-agents-cli-observability**: tracing/logging setup and troubleshooting for deployed/running agents — use alongside `src/observability/`.
