# CAIP Responses Library — Project Roadmap

## Assignment Objective
Integrate a responses.create-compatible API endpoint that can serve as a fallback or cost-optimized alternative to GPT-4o for CAIP's inbound calling use case. The same API format must work regardless of which model provider (OpenAI, Anthropic, Gemini, Sarvam, or open-source Llama/Mistral/Qwen) is behind the scenes.

---

## Current Status: ~70% Complete

### ✅ Done

- **responses.create-compatible wrapper** — `client.responses.create()` works identically across providers
- **Provider adapters** — OpenAI (pass-through), Anthropic, Gemini, Sarvam (all working; confirmed via quicktest.py with sarvam-30b)
- **Tool use** — function calling across all providers; two-tier dispatch for built-in tools (web search, code interpreter, shell, MCP, file search, image generation, computer use)
- **Gemini native web search** — `{"type": "web_search"}` maps to Gemini's native Google Search grounding (server-side, no OpenAI fallback); surfaces `web_search_call` items + `url_citation` annotations. Verified live in `quicktest.py` (Test 7) against `gemini-2.5-flash`.
- **Multi-turn conversation** — `previous_response_id` via `ConversationStore`
- **Streaming** — canonical event sequence implemented for all providers
- **Infrastructure** — rate limiting, response caching, cost tracker (no pricing loaded yet), Redis support, plugin system
- **Tests** — 676 unit tests (all mocked, no API keys needed)

### ❌ Not Done

| # | Gap | Details |
|---|---|---|
| 1 | **Open-source models (Llama, Mistral, Qwen)** | Assignment explicitly requires these. Need a generic `OllamaProvider` / `LocalProvider` that accepts any `base_url` + model name (OpenAI-compatible Chat Completions format). Sarvam provider already does this pattern — reuse its translation logic. |
| 2 | **Benchmark script** | No `benchmark.py` exists. Need to measure latency (TTFT + total), accuracy (CAIP inbound calling prompts), and cost vs GPT-4o across all providers. `CostTracker` exists but has no pricing loaded. |
| 3 | **Pricing data** | `CostTracker.set_pricing()` must be called manually — no defaults. Need to pre-load pricing for GPT-4o, Claude, Gemini, Sarvam-30b, and any open-source models. |
| 4 | **Live MCP test with CAIP's tool definitions** | `MCPHandler` is built and can connect to MCP servers. Need a test script using CAIP's actual MCP tool definitions with a non-OpenAI provider (Sarvam/Anthropic/Gemini). |
| 5 | **Anthropic + OpenAI in quicktest.py** | Gemini is fully covered (7 tests incl. web search). Still need Anthropic + OpenAI live blocks. |

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

### Step 1b — Gemini native web search (Google Search grounding) — ✅ DONE
- [x] `GeminiProvider.supports_tool("web_search")` → `True`, so the client passes the tool to the provider instead of the OpenAI-backed client-side `WebSearchHandler` fallback
- [x] `_build_kwargs` maps `{"type": "web_search"}` → native `{"google_search": {}}`; coexists with `function_declarations`; `tool_config` only set for real function tools
- [x] `_convert_response` + streaming surface grounding as `web_search_call` items (one per query) + `url_citation` annotations on the answer text — parity with OpenAI's native web_search shape
- [x] New `_extract_grounding()` helper (defensive against non-list metadata)
- [x] Added 4 tests + updated `test_supports_tool`; full suite 676 pass, lint clean
- [x] Verified live in `quicktest.py` Test 7 (`gemini-2.5-flash`, Vertex): 2 search queries + 3 citations returned
- [ ] REMAINING: confirm `google_search` + `function_declarations` combined in one request works on the target model (supported on Gemini 2.0+; restricted on 1.5)

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
- [x] Added web search test (Test 7) — native Gemini grounding; 7/7 passing on `gemini-2.5-flash`
- [ ] Add Anthropic (Claude) test block with `anthropic_api_key`
- [ ] Add OpenAI test block with `openai_api_key`
- [ ] Add Ollama test block (no key needed, just local base_url)

---

## Key Files Reference

| File | Purpose |
|---|---|
| `src/caip_responses/providers/openai_compatible.py` | Shared OpenAI Chat Completions adapter — base for Sarvam + Ollama |
| `src/caip_responses/providers/ollama_provider.py` | Open-source model provider (Ollama/vLLM/LM Studio) |
| `src/caip_responses/providers/sarvam_provider.py` | Thin subclass of OpenAICompatibleProvider |
| `src/caip_responses/providers/registry.py` | Add prefix mappings for new providers here |
| `src/caip_responses/providers/gemini_provider.py` | Gemini adapter — native Google Search grounding for web_search (`_extract_grounding`) |
| `src/caip_responses/cost/tracker.py` | Add DEFAULT_PRICING here |
| `src/caip_responses/tool_handlers/mcp.py` | MCP handler — direct SDK + OpenAI fallback |
| `quicktest.py` | Live smoke test (Gemini, 7/7 passing — incl. web search grounding) |

---

## Notes
- `sarvam-m` is deprecated — use `sarvam-30b` or `sarvam-105b`
- `sarvam-30b` is a reasoning model — needs `max_output_tokens >= 2000` or reasoning tokens exhaust the budget before final answer
- Gemini daily free-tier quota exhausted on the test key — but Vertex AI via service account (`gemini-sa.json`) works; quicktest runs against `gemini-2.5-flash` there
- Gemini web search uses native Google Search grounding — citation URLs come back as `vertexaisearch.cloud.google.com/grounding-api-redirect/...` redirect links (not the raw source URLs)
- All quicktest fixes committed: `1e3d20c`
