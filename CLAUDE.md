# caip-responses-lib

Multi-provider LLM library that implements the OpenAI Responses API interface for any LLM provider.

## Core Principle
The API which we call to use LLMs should ALWAYS be the same format/template irrespective of which LLM is behind the scene.

## Architecture
- `src/caip_responses/models/` — Pydantic v2 data models (request, response, items, tools, streaming events)
- `src/caip_responses/client/` — AsyncClient and Client with `responses.create()` interface
- `src/caip_responses/providers/` — BaseProvider ABC + per-provider adapters (openai, anthropic, gemini, sarvam)
- `src/caip_responses/stream/` — Unified streaming with per-provider adapters
- `src/caip_responses/loop/` — Client-side agentic tool-calling loop
- `src/caip_responses/utils/` — ID generation, JSON schema helpers

## Conventions
- Python 3.11+, Pydantic v2 with `model_config = ConfigDict(extra="forbid")`
- Async-first, sync wrapper available
- All provider SDKs are optional dependencies
- Run tests: `pytest tests/`
- Lint: `ruff check src/ tests/`
