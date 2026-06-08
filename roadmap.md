# CAIP Responses Library — Project Roadmap

## Assignment Objective
Integrate a responses.create-compatible API endpoint that can serve as a fallback or cost-optimized alternative to GPT-4o for CAIP's inbound calling use case. The same API format must work regardless of which model provider (OpenAI, Anthropic, Gemini, Sarvam, or open-source Llama/Mistral/Qwen) is behind the scenes.

---

## Current Status: ~70% Complete

### ✅ Done

- **responses.create-compatible wrapper** — `client.responses.create()` works identically across providers
- **Provider adapters** — OpenAI (pass-through), Anthropic, Gemini, Sarvam (all working; confirmed via quicktest.py with sarvam-30b)
- **Tool use** — function calling across all providers; two-tier dispatch for built-in tools (web search, code interpreter, shell, MCP, file search, image generation, computer use)
- **Multi-turn conversation** — `previous_response_id` via `ConversationStore`
- **Streaming** — canonical event sequence implemented for all providers
- **Infrastructure** — rate limiting, response caching, cost tracker (no pricing loaded yet), Redis support, plugin system
- **Tests** — 630 unit tests (all mocked, no API keys needed)

### ❌ Not Done

| # | Gap | Details |
|---|---|---|
| 1 | **Open-source models (Llama, Mistral, Qwen)** | Assignment explicitly requires these. Need a generic `OllamaProvider` / `LocalProvider` that accepts any `base_url` + model name (OpenAI-compatible Chat Completions format). Sarvam provider already does this pattern — reuse its translation logic. |
| 2 | **Benchmark script** | No `benchmark.py` exists. Need to measure latency (TTFT + total), accuracy (CAIP inbound calling prompts), and cost vs GPT-4o across all providers. `CostTracker` exists but has no pricing loaded. |
| 3 | **Pricing data** | `CostTracker.set_pricing()` must be called manually — no defaults. Need to pre-load pricing for GPT-4o, Claude, Gemini, Sarvam-30b, and any open-source models. |
| 4 | **Live MCP test with CAIP's tool definitions** | `MCPHandler` is built and can connect to MCP servers. Need a test script using CAIP's actual MCP tool definitions with a non-OpenAI provider (Sarvam/Anthropic/Gemini). |
| 5 | **Anthropic + OpenAI in quicktest.py** | Currently only Gemini and Sarvam are in the live smoke test. |

---

## Roadmap (Priority Order)

### Step 1 — `OllamaProvider` / `LocalProvider` (open-source models) — ✅ DONE
- [x] Extracted shared Chat Completions logic into `providers/openai_compatible.py` (`OpenAICompatibleProvider`)
- [x] Refactored `SarvamProvider` to subclass it (thin — name + base_url only)
- [x] Added `providers/ollama_provider.py` (`OllamaProvider`) — works with Ollama, vLLM, LM Studio
- [x] Registered prefixes: `llama`, `mistral`, `mixtral`, `qwen`, `qwq`, `gemma`, `phi`, `deepseek`, `ollama/`
- [x] Wired into client: `ollama_api_key` / `ollama_base_url` params + config + `_init_ollama` (gated on base_url)
- [x] Added 21 tests (`tests/test_providers/test_ollama_provider.py`); full suite 668 pass, lint clean
- [ ] REMAINING: install Ollama locally + live-test Llama/Mistral/Qwen through the library

### Step 2 — Pre-load pricing table
- Add a `DEFAULT_PRICING` dict in `cost/tracker.py` or a new `cost/pricing.py`
- Cover: `gpt-4o`, `gpt-4.1`, `claude-sonnet-4`, `gemini-2.0-flash`, `sarvam-30b`, `sarvam-105b`
- Open-source models: cost = 0 (self-hosted) or infra cost per token

### Step 3 — `benchmark.py`
- Run a fixed prompt set across all providers (same prompts)
- Measure: time to first token, total latency, input/output tokens, cost
- Output a comparison table (provider | model | latency | cost | answer quality)
- Use CAIP's inbound calling prompts for accuracy evaluation

### Step 4 — Live MCP test with CAIP's tool definitions
- Get CAIP's actual MCP server URL + tool definitions
- Write `mcp_test.py` that passes those tools through the library to Sarvam/Anthropic
- Validate tool discovery, invocation, and result handling end-to-end

### Step 5 — Expand quicktest.py
- Add Anthropic (Claude) test block with `anthropic_api_key`
- Add OpenAI test block with `openai_api_key`
- Add Ollama test block (no key needed, just local base_url)

---

## Key Files Reference

| File | Purpose |
|---|---|
| `src/caip_responses/providers/openai_compatible.py` | Shared OpenAI Chat Completions adapter — base for Sarvam + Ollama |
| `src/caip_responses/providers/ollama_provider.py` | Open-source model provider (Ollama/vLLM/LM Studio) |
| `src/caip_responses/providers/sarvam_provider.py` | Thin subclass of OpenAICompatibleProvider |
| `src/caip_responses/providers/registry.py` | Add prefix mappings for new providers here |
| `src/caip_responses/cost/tracker.py` | Add DEFAULT_PRICING here |
| `src/caip_responses/tool_handlers/mcp.py` | MCP handler — direct SDK + OpenAI fallback |
| `quicktest.py` | Live smoke test (currently Sarvam only, 5/5 passing) |

---

## Notes
- `sarvam-m` is deprecated — use `sarvam-30b` or `sarvam-105b`
- `sarvam-30b` is a reasoning model — needs `max_output_tokens >= 2000` or reasoning tokens exhaust the budget before final answer
- Gemini daily free-tier quota exhausted on the test key — need a fresh Google Cloud project or billing enabled
- All quicktest fixes committed: `1e3d20c`
