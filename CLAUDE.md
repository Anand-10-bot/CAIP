# caip-responses-lib

Multi-provider LLM library that implements the OpenAI Responses API interface for any LLM provider. Not a proxy server — an in-process client library.

## Core Principle (NON-NEGOTIABLE)

> **The API which we call to use LLMs should ALWAYS be the same format/template irrespective of which LLM is behind the scene.**

No provider-specific parameters, no conditional logic in caller code, no different response shapes. Switching providers is a one-line model name change.

## Architecture

All source code is under `src/caip_responses/`.

### Core modules

- `models/` — Pydantic v2 data models: `CreateResponseRequest`, `Response` (with `.output_text`, `.function_calls`, `.has_function_calls`), `StreamEvent`, `Usage`, `Reasoning`, `TextConfig`, tool definitions, items, and error types (`CaipResponsesError`, `ProviderError`, `MaxStepsExceededError`)
- `client/` — `AsyncClient` and `Client` (sync wrapper) with `client.responses.create()` interface. `_ResponsesNamespace` handles routing, conversation store, structured output validation, caching, cost tracking, and rate limiting. `CaipResponsesConfig` (pydantic-settings) reads env vars with `CAIP_RESPONSES_` prefix.
- `providers/` — `BaseProvider` ABC with 4 adapters:
  - `openai_provider.py` — Near pass-through to OpenAI SDK `responses.create()`
  - `anthropic_provider.py` — Translates to/from Anthropic Messages API
  - `gemini_provider.py` — Translates to/from Google GenerateContent API (google-genai SDK)
  - `sarvam_provider.py` — Translates to/from standard Chat Completions API via httpx
  - `registry.py` — `ProviderRegistry` routes models to providers by prefix

### Tool system (two-tier dispatch)

- `tool_handlers/` — Client-side built-in tool emulation for non-OpenAI providers:
  - `base.py` — `BuiltinToolHandler` ABC: `to_function_tools()`, `execute()`, `system_prompt_addendum()`, `to_output_item()`
  - `registry.py` — `BuiltinToolRegistry`: maps tool types to handlers, `preprocess_tools()` replaces unsupported tools with synthetic function definitions, `resolve_for_function()` finds handler by function name prefix `_builtin_`
  - **7 handlers**: `web_search.py`, `code_interpreter.py`, `shell.py`, `mcp.py`, `file_search.py`, `image_generation.py`, `computer_use.py`
  - `openai_delegator.py` — Base for handlers that delegate execution to Azure OpenAI (MCP, file_search, image_generation, computer_use)
- `loop/` — `AgentLoop` runs the client-side agentic loop for non-OpenAI providers. Uses `asyncio.gather` for parallel tool execution. `ToolExecutor` dispatches user-registered function callbacks.

### Infrastructure modules

- `store/` — `ConversationStore` (in-memory LRU, default 1000) + `RedisConversationStore` for `previous_response_id` simulation on non-OpenAI providers. Stores response ID → (input items, instructions) mapping.
- `cache/` — `ResponseCache` (in-memory LRU) + `RedisResponseCache` for caching deterministic (temperature=0) responses.
- `cost/` — `CostTracker` with per-model `ModelPricing` (input/output cost per million tokens). Tracks `UsageRecord` per model. Methods: `record()`, `by_model()`, `by_provider()`, `total_cost`.
- `ratelimit/` — `RateLimiter` with token-bucket algorithm per provider. Configurable via `RateLimitConfig` (requests_per_minute, tokens_per_minute).
- `plugins/` — `PluginManager` discovers custom providers via `caip_responses.providers` entry points (stevedore).
- `stream/` — Placeholder for per-provider stream adapter layer (streaming currently lives inside each provider).
- `utils/` — `id_gen.py` generates IDs with prefixes (`resp_`, `item_`, `fc_`, `call_`). `json_schema.py` validates output against JSON schemas.

## Key Design Patterns

### Provider prefix routing

`ProviderRegistry` maps model name prefixes to provider names:

| Prefix | Provider |
|---|---|
| `gpt-`, `o1-`, `o3-`, `o4-` | openai |
| `claude-` | anthropic |
| `gemini-` | gemini |
| `sarvam-` | sarvam |

Override with `provider="anthropic"` parameter. Custom prefixes via `registry.add_prefix_mapping()`.

### Two-tier tool dispatch (non-OpenAI)

When the caller passes tools like `web_search` or `mcp` to a non-OpenAI provider:

1. `BuiltinToolRegistry.preprocess_tools()` replaces unsupported tool types with synthetic function definitions (e.g. `_builtin_web_search_query`) + injects system prompt addenda
2. The model calls these synthetic functions via standard function calling
3. `AgentLoop` resolves function names: `BuiltinToolRegistry.resolve_for_function()` checks for `_builtin_` prefix → routes to builtin handler. Otherwise → `ToolExecutor` dispatches to user-registered callback.
4. Results are appended as `function_call_output` items, loop continues

### Canonical streaming event sequence

All providers must emit events in this order:

```
response.created → response.in_progress →
  response.output_item.added →
    response.content_part.added →
      response.output_text.delta (repeated) →
    response.output_text.done →
    response.content_part.done →
  response.output_item.done →
response.completed
```

Function calls use: `response.function_call_arguments.delta` → `response.function_call_arguments.done`
Reasoning uses: `response.reasoning_text.delta`

### previous_response_id (non-OpenAI)

OpenAI stores state server-side. For other providers, `ConversationStore` saves response ID → (input items, instructions) after each call. When `previous_response_id` is passed, the client reconstitutes the full history and prepends it to the new input.

### Structured output enforcement (non-OpenAI)

When `text.format.type == "json_schema"` is requested, the client post-validates the response output against the schema. If validation fails, `response.error` is set with type `json_schema_validation_error`. Providers also inject the schema into system instructions.

## Provider Translation Summary

| Responses API field | OpenAI | Anthropic | Gemini | Sarvam |
|---|---|---|---|---|
| `input` | pass-through | `messages` | `contents` with `parts` | `messages` |
| `instructions` | pass-through | `system` param | `system_instruction` config | system message |
| `tools` (function) | pass-through | `tools` with `input_schema` | `function_declarations` | Chat Completions format |
| `tool_choice` | pass-through | `{type: "auto"/"any"/"tool"}` | `function_calling_config` (AUTO/ANY/NONE) | Chat Completions format |
| `reasoning.effort` | pass-through | `thinking.budget_tokens` | `thinking_config.thinking_budget` | `reasoning_effort` |
| `stream` | pass-through | `stream=True` + SSE | `generate_content_stream` | SSE lines `data: {...}` |
| `previous_response_id` | server-side | client ConversationStore | client ConversationStore | client ConversationStore |

## Configuration

`CaipResponsesConfig` (pydantic-settings) reads env vars with `CAIP_RESPONSES_` prefix:

```
CAIP_RESPONSES_OPENAI_API_KEY      CAIP_RESPONSES_OPENAI_BASE_URL
CAIP_RESPONSES_ANTHROPIC_API_KEY   CAIP_RESPONSES_ANTHROPIC_BASE_URL
CAIP_RESPONSES_GEMINI_API_KEY
CAIP_RESPONSES_SARVAM_API_KEY      CAIP_RESPONSES_SARVAM_BASE_URL
CAIP_RESPONSES_REDIS_URL           CAIP_RESPONSES_DEFAULT_PROVIDER
CAIP_RESPONSES_AGENT_LOOP_MAX_STEPS (default: 10)
CAIP_RESPONSES_CONVERSATION_TTL    (default: 86400)
```

Constructor params override env vars. Redis URL enables persistent conversation store + response cache.

## Commands

```bash
# Run tests (630 tests, all mocked — no API keys needed)
pytest tests/

# Lint
ruff check src/ tests/

# Build wheel
python -m build

# Install in dev mode
pip install -e ".[dev]"

# Install with specific provider
pip install -e ".[openai,anthropic,gemini]"
```

## Conventions

- Python 3.11+, Pydantic v2 with `ConfigDict(extra="forbid")` on strict models
- Async-first (`AsyncClient`), sync wrapper (`Client`) uses `asyncio.new_event_loop()`
- All provider SDKs are optional dependencies — install via extras `[openai]`, `[anthropic]`, `[gemini]`, `[redis]`, `[mcp]`, or `[all]`
- Build: hatchling backend, source in `src/caip_responses/`
- Ruff config: `target-version = "py311"`, rules `E,F,I,UP`, `E501` ignored
- Test: pytest with `asyncio_mode = "auto"`, marker `@pytest.mark.integration` for live API tests

## Test Structure

Tests mirror the source layout under `tests/`:

| Test directory | Source module | What it tests |
|---|---|---|
| `test_models/` | `models/` | Pydantic model validation, serialization, properties |
| `test_providers/` | `providers/` | Translation logic per provider (mocked SDK calls) |
| `test_client/` | `client/` | Client routing, conversation store integration, structured output validation, phase 4 features |
| `test_loop/` | `loop/` | Agent loop steps, tool executor dispatch |
| `test_tool_handlers/` | `tool_handlers/` | Each handler's to_function_tools/execute/is_synthetic, two-tier dispatch, client wiring |
| `test_store/` | `store/` | ConversationStore LRU, RedisConversationStore |
| `test_cache/` | `cache/` | ResponseCache LRU, RedisResponseCache |
| `test_cost/` | `cost/` | CostTracker recording, pricing, breakdown |
| `test_ratelimit/` | `ratelimit/` | Token-bucket rate limiter |
| `test_plugins/` | `plugins/` | Entry-point discovery |
| `test_stream/` | `stream/` | (placeholder) |
| `test_integration/` | cross-cutting | Provider switching, parallel tools, cross-provider streaming, cost tracking |

All tests use mocked providers/HTTP — no API keys required. 630 tests total.
