# caip-responses-lib

**Multi-provider LLM library that implements the OpenAI Responses API interface for any LLM provider.**

Write your code once. Switch between OpenAI, Anthropic Claude, Google Gemini, and Sarvam AI with a single model name change. No provider-specific parameters, no conditional logic, no different response shapes.

```python
from caip_responses import AsyncClient

client = AsyncClient(
    openai_api_key="sk-...",
    anthropic_api_key="sk-ant-...",
    gemini_api_key="AIza...",
)

# Same code works with ANY provider - just change the model name
response = await client.responses.create(
    model="gpt-4.1",                    # OpenAI
    # model="claude-sonnet-4-20250514",  # Anthropic
    # model="gemini-2.0-flash",          # Google Gemini
    # model="sarvam-2b-v0.5",            # Sarvam AI
    input="Explain quantum computing in simple terms.",
    instructions="Be concise and use analogies.",
)
print(response.output_text)
```

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
   - [Environment Variables](#environment-variables)
   - [Constructor Parameters](#constructor-parameters)
4. [Providers](#providers)
   - [Model Routing](#model-routing)
   - [OpenAI](#openai-provider)
   - [Anthropic Claude](#anthropic-provider)
   - [Google Gemini](#gemini-provider)
   - [Sarvam AI](#sarvam-provider)
5. [API Reference: responses.create()](#api-reference-responsescreate)
6. [Tools](#tools)
   - [Function Tools](#function-tools)
   - [Web Search](#web-search)
   - [Code Interpreter](#code-interpreter)
   - [Shell](#shell)
   - [MCP (Model Context Protocol)](#mcp)
   - [File Search](#file-search)
   - [Image Generation](#image-generation)
   - [Computer Use](#computer-use)
   - [All Tools with All LLMs](#all-tools-with-all-llms)
7. [Streaming](#streaming)
8. [Multi-Turn Conversations](#multi-turn-conversations)
9. [Structured Output (JSON Schema)](#structured-output-json-schema)
10. [Reasoning / Thinking](#reasoning--thinking)
11. [Agentic Loop](#agentic-loop)
12. [Cost Tracking](#cost-tracking)
13. [Rate Limiting](#rate-limiting)
14. [Response Caching](#response-caching)
15. [Redis (Production Persistence)](#redis-production-persistence)
16. [Postman Collection](#postman-collection)
17. [Backend Integration Guide](#backend-integration-guide)
18. [Error Handling](#error-handling)
19. [Plugin System](#plugin-system)

---

## Installation

**Requirements:** Python 3.11+

**Core dependencies (auto-installed):** `pydantic>=2.8`, `pydantic-settings>=2.0`, `httpx>=0.27`

### Option 1: Install from wheel file (simplest)

Download the `.whl` file and install directly:

```bash
# Core only (no provider SDKs)
pip install caip_responses_lib-0.1.0-py3-none-any.whl

# With specific providers
pip install "caip_responses_lib-0.1.0-py3-none-any.whl[openai]"
pip install "caip_responses_lib-0.1.0-py3-none-any.whl[anthropic]"
pip install "caip_responses_lib-0.1.0-py3-none-any.whl[gemini]"
pip install "caip_responses_lib-0.1.0-py3-none-any.whl[redis]"

# All providers + Redis
pip install "caip_responses_lib-0.1.0-py3-none-any.whl[all]"

# Development (tests, linting)
pip install "caip_responses_lib-0.1.0-py3-none-any.whl[all,dev]"
```

### Option 2: Install from git repository

If the library is hosted in a git repository:

```bash
# Core
pip install git+https://your-org/caip-responses-lib.git

# With extras
pip install "caip-responses-lib[openai,anthropic] @ git+https://your-org/caip-responses-lib.git"

# All providers
pip install "caip-responses-lib[all] @ git+https://your-org/caip-responses-lib.git"
```

In your `requirements.txt`:
```
caip-responses-lib[all] @ git+https://your-org/caip-responses-lib.git@v0.1.0
```

Or in `pyproject.toml`:
```toml
dependencies = [
    "caip-responses-lib[all] @ git+https://your-org/caip-responses-lib.git@v0.1.0",
]
```

### Option 3: Install from private PyPI / Azure Artifacts

If your organization hosts a private package feed:

```bash
# One-time setup: upload the package
pip install twine
twine upload --repository-url https://pkgs.dev.azure.com/YOUR_ORG/_packaging/YOUR_FEED/pypi/upload/ dist/*

# Then anyone in the org can install:
pip install caip-responses-lib --index-url https://pkgs.dev.azure.com/YOUR_ORG/_packaging/YOUR_FEED/pypi/simple/

# With extras:
pip install "caip-responses-lib[all]" --index-url https://pkgs.dev.azure.com/YOUR_ORG/_packaging/YOUR_FEED/pypi/simple/
```

Configure pip globally so you don't need `--index-url` every time:
```ini
# ~/.pip/pip.conf (Linux/macOS) or %APPDATA%\pip\pip.ini (Windows)
[global]
extra-index-url = https://pkgs.dev.azure.com/YOUR_ORG/_packaging/YOUR_FEED/pypi/simple/
```

### Option 4: Install from source (editable / development)

```bash
git clone https://your-org/caip-responses-lib.git
cd caip-responses-lib
pip install -e ".[all,dev]"
```

### Optional dependency extras

| Extra | What it installs | When you need it |
|---|---|---|
| `openai` | `openai>=2.20` | Using GPT-4, O1, O3, O4 models; web_search/file_search/image_generation delegation |
| `anthropic` | `anthropic>=0.40` | Using Claude models |
| `gemini` | `google-genai>=1.0` | Using Gemini models |
| `redis` | `redis[hiredis]>=5.0` | Production persistence (conversations + cache) |
| `mcp` | `mcp>=1.0` | Direct MCP server connections (no OpenAI needed for MCP) |
| `all` | All of the above | Full multi-provider setup |
| `dev` | pytest, ruff, bandit | Running tests and linting |

### Verify installation

```python
python -c "from caip_responses import AsyncClient, __version__; print(f'caip-responses-lib v{__version__} installed successfully')"
```

### Building from source

To build the wheel and sdist yourself:

```bash
pip install build hatchling
python -m build
# Output: dist/caip_responses_lib-0.1.0-py3-none-any.whl
#         dist/caip_responses_lib-0.1.0.tar.gz
```

---

## Quick Start

### Async (recommended)

```python
import asyncio
from caip_responses import AsyncClient

async def main():
    client = AsyncClient(
        openai_api_key="sk-...",
        anthropic_api_key="sk-ant-...",
    )

    # Non-streaming
    response = await client.responses.create(
        model="claude-sonnet-4-20250514",
        input="What is the capital of France?",
    )
    print(response.output_text)  # "The capital of France is Paris."

    # Streaming
    stream = await client.responses.create(
        model="claude-sonnet-4-20250514",
        input="Write a haiku about coding.",
        stream=True,
    )
    async for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)

    await client.close()

asyncio.run(main())
```

### Sync

```python
from caip_responses import Client

client = Client(openai_api_key="sk-...")
response = client.responses.create(
    model="gpt-4.1",
    input="Hello!",
)
print(response.output_text)
client.close()
```

---

## Configuration

### Environment Variables

All environment variables are prefixed with `CAIP_RESPONSES_`. They serve as fallback defaults when constructor params are not provided.

| Environment Variable | Description | Default |
|---|---|---|
| `CAIP_RESPONSES_OPENAI_API_KEY` | OpenAI API key | `""` |
| `CAIP_RESPONSES_OPENAI_BASE_URL` | OpenAI base URL (for Azure, proxies) | `""` |
| `CAIP_RESPONSES_ANTHROPIC_API_KEY` | Anthropic API key | `""` |
| `CAIP_RESPONSES_ANTHROPIC_BASE_URL` | Anthropic base URL | `""` |
| `CAIP_RESPONSES_GEMINI_API_KEY` | Google Gemini API key | `""` |
| `CAIP_RESPONSES_SARVAM_API_KEY` | Sarvam AI API key | `""` |
| `CAIP_RESPONSES_SARVAM_BASE_URL` | Sarvam base URL | `https://api.sarvam.ai/v1` |
| `CAIP_RESPONSES_OPENAI_BASE_URL` | Azure OpenAI endpoint (also used for web_search/tool delegation) | `""` |
| `CAIP_RESPONSES_REDIS_URL` | Redis connection URL (enables persistent store + cache) | `""` |
| `CAIP_RESPONSES_DEFAULT_PROVIDER` | Default provider when model prefix is unknown | `""` |
| `CAIP_RESPONSES_AGENT_LOOP_MAX_STEPS` | Max agentic loop iterations | `10` |
| `CAIP_RESPONSES_CONVERSATION_TTL` | Conversation history TTL in seconds (Redis) | `86400` |

### Constructor Parameters

```python
client = AsyncClient(
    # Provider API keys (override env vars)
    openai_api_key="sk-...",
    openai_base_url="https://your-azure-endpoint.openai.azure.com/",
    anthropic_api_key="sk-ant-...",
    anthropic_base_url=None,
    gemini_api_key="AIza...",
    sarvam_api_key="sarvam-...",
    sarvam_base_url="https://api.sarvam.ai/v1",

    # Provider routing
    default_provider=None,       # "openai" | "anthropic" | "gemini" | "sarvam"
    providers=None,              # dict of custom BaseProvider instances

    # Conversation state
    max_conversation_history=1000,   # max stored responses for previous_response_id

    # Caching
    cache_max_size=500,          # max cached responses
    cache_ttl=3600,              # cache TTL in seconds
    enable_cache=True,           # enable response caching

    # Plugins
    discover_plugins=True,       # auto-discover entry-point plugins

    # Built-in tool handlers (for non-OpenAI providers)
    web_search_model="gpt-4.1-nano",       # model used for web search delegation
    web_search_callback=None,              # custom async search function
    code_interpreter_enabled=False,        # enable code execution
    code_interpreter_timeout=30,           # execution timeout (seconds)
    code_interpreter_working_dir=None,     # working directory
    code_interpreter_callback=None,        # custom async executor
    shell_enabled=False,                   # enable shell commands
    shell_timeout=30,                      # execution timeout (seconds)
    shell_working_dir=None,                # working directory
    shell_callback=None,                   # custom async executor
    builtin_registry=None,                 # override with custom BuiltinToolRegistry

    # Redis persistence (conversation store + response cache)
    redis_url=None,                        # e.g. "redis://localhost:6379/0"
    conversation_ttl=86400,                # conversation TTL in seconds (24h)
)
```

---

## Providers

### Model Routing

The provider is automatically detected from the model name prefix:

| Model Prefix | Provider | Examples |
|---|---|---|
| `gpt-` | OpenAI | `gpt-4.1`, `gpt-4.1-mini`, `gpt-4o` |
| `o1-`, `o3-`, `o4-` | OpenAI | `o1-pro`, `o3-mini`, `o4-mini` |
| `claude-` | Anthropic | `claude-sonnet-4-20250514`, `claude-haiku-4-5-20251001` |
| `gemini-` | Google Gemini | `gemini-2.0-flash`, `gemini-2.5-pro` |
| `sarvam-` | Sarvam AI | `sarvam-2b-v0.5` |

Override auto-detection with the `provider` parameter:

```python
response = await client.responses.create(
    model="my-custom-model",
    provider="anthropic",       # force Anthropic provider
    input="Hello",
)
```

### OpenAI Provider

Near pass-through to the OpenAI SDK. All features work natively: function tools, web_search, file_search, code_interpreter, MCP, shell, image_generation, computer use, previous_response_id.

```python
client = AsyncClient(openai_api_key="sk-...")

# For Azure OpenAI
client = AsyncClient(
    openai_api_key="your-azure-key",
    openai_base_url="https://your-resource.openai.azure.com/",
)
```

### Anthropic Provider

Translates to the Anthropic Messages API. Function tools work natively. All other tool types (web_search, code_interpreter, shell) are automatically emulated via synthetic function calls.

```python
client = AsyncClient(anthropic_api_key="sk-ant-...")
```

**Translation details:**
- `input` items -> `messages` array
- `instructions` -> `system` parameter
- `tools` (function) -> Anthropic `tools` with `input_schema`
- `reasoning.effort` -> `thinking.budget_tokens` (low=1024, medium=4096, high=16384)
- `stop_reason: "tool_use"` -> triggers client-side agentic loop

### Gemini Provider

Translates to the Google GenerateContent API. Function tools work natively.

```python
client = AsyncClient(gemini_api_key="AIza...")
```

**Translation details:**
- `input` items -> `contents` with `parts`
- `instructions` -> `system_instruction`
- `tools` (function) -> `function_declarations`
- JSON schema -> `response_mime_type: "application/json"` + `response_schema`

### Sarvam Provider

Translates to the standard Chat Completions API format.

```python
client = AsyncClient(
    sarvam_api_key="sarvam-...",
    sarvam_base_url="https://api.sarvam.ai/v1",  # default
)
```

---

## API Reference: responses.create()

```python
response = await client.responses.create(
    # Required
    model="claude-sonnet-4-20250514",

    # Input
    input="Hello",                       # str or list of input items
    instructions="Be helpful.",          # system instructions

    # Tools
    tools=[...],                         # list of tool definitions
    tool_choice="auto",                  # "auto" | "required" | "none" | {"type": "function", "name": "fn"}
    parallel_tool_calls=None,            # allow parallel tool calls

    # Output control
    stream=False,                        # True for streaming
    temperature=0.7,                     # 0.0 - 2.0
    top_p=None,                          # nucleus sampling
    max_output_tokens=4096,              # max output tokens
    text=None,                           # TextConfig for structured output

    # Conversation chaining
    previous_response_id=None,           # chain to previous response
    store=None,                          # persist for chaining (default: True)

    # Reasoning
    reasoning={"effort": "medium"},      # "low" | "medium" | "high"

    # Advanced
    prompt=None,                         # reusable prompt template
    metadata=None,                       # key-value metadata
    truncation=None,                     # "auto" | "disabled"
    user=None,                           # end-user identifier
    include=None,                        # additional data to include
    background=None,                     # background mode
    provider=None,                       # override auto-detected provider
)
```

### Response Object

```python
response.id                # "resp_abc123..." - unique response ID
response.model             # "claude-sonnet-4-20250514"
response.status            # "completed" | "failed" | "incomplete" | "in_progress"
response.output            # list of output items (messages, function calls, etc.)
response.output_text       # convenience: concatenated text from all messages
response.has_function_calls  # bool: any function calls in output?
response.function_calls    # list of FunctionCallItem objects
response.usage             # Usage object (input_tokens, output_tokens, total_tokens)
response.error             # error dict or None
response.metadata          # metadata dict or None
```

### Input Formats

```python
# Simple text
input="What is 2+2?"

# Conversation items
input=[
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "What's the weather?"},
]

# Rich content (text + images)
input=[
    {
        "role": "user",
        "content": [
            {"type": "input_text", "text": "What's in this image?"},
            {"type": "input_image", "image_url": "https://example.com/photo.jpg"},
        ],
    },
]

# With function call results
input=[
    {"role": "user", "content": "What's the weather in NYC?"},
    # ... previous response had a function_call ...
    {
        "type": "function_call_output",
        "call_id": "fc_abc123",
        "output": '{"temp": 72, "condition": "sunny"}',
    },
]
```

---

## Tools

### Function Tools

Custom functions the model can call. Works natively with all providers.

```python
# Define tools
tools = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
]

# Register handler
async def weather_handler(name: str, args: dict) -> str:
    city = args["city"]
    return json.dumps({"temp": 72, "condition": "sunny", "city": city})

client.tools.register("get_weather", weather_handler)

# Make request - agentic loop handles tool calls automatically
response = await client.responses.create(
    model="claude-sonnet-4-20250514",
    input="What's the weather in San Francisco?",
    tools=tools,
)
print(response.output_text)
# "The weather in San Francisco is 72 degrees and sunny."
```

### Web Search

Search the web for current information. **No extra API keys needed** — uses your existing Azure OpenAI deployment.

```python
# Recommended: uses your existing Azure OpenAI for web search
client = AsyncClient(
    openai_api_key="sk-...",
    openai_base_url="https://your-resource.openai.azure.com",
    anthropic_api_key="sk-ant-...",
)

# Or with custom search callback (fully independent)
async def my_search(query: str, num_results: int) -> list[dict]:
    # Your search implementation
    return [{"title": "...", "url": "...", "snippet": "..."}]

client = AsyncClient(
    anthropic_api_key="sk-ant-...",
    web_search_callback=my_search,
)

# Use in request — same code for any provider
response = await client.responses.create(
    model="claude-sonnet-4-20250514",
    input="What happened in the news today?",
    tools=[{"type": "web_search", "search_context_size": "medium"}],
)

# Track web search token usage for billing
print(client.web_search_metrics.total_tokens)
```

**How it works:**
- **OpenAI:** Native server-side web search (zero overhead)
- **Other providers:** Converted to synthetic function. Fallback chain:
  1. Custom callback (if `web_search_callback` provided) — no dependency
  2. Azure OpenAI `web_search` tool (if `openai_api_key` provided) — uses your existing deployment
  3. DuckDuckGo Instant Answer API (no key needed, best-effort)

### Code Interpreter

Execute Python code in a sandboxed environment.

```python
client = AsyncClient(
    anthropic_api_key="sk-ant-...",
    code_interpreter_enabled=True,      # MUST enable explicitly
    code_interpreter_timeout=60,        # seconds
)

response = await client.responses.create(
    model="claude-sonnet-4-20250514",
    input="Calculate the first 20 Fibonacci numbers",
    tools=[{"type": "code_interpreter"}],
)
```

**Security:** Disabled by default. Must set `code_interpreter_enabled=True`.

**Custom executor:**
```python
async def my_executor(code: str, language: str) -> dict:
    # Run in Docker, remote sandbox, etc.
    return {"stdout": "...", "stderr": "", "exit_code": 0}

client = AsyncClient(
    anthropic_api_key="sk-ant-...",
    code_interpreter_enabled=True,
    code_interpreter_callback=my_executor,
)
```

### Shell

Execute shell commands.

```python
client = AsyncClient(
    anthropic_api_key="sk-ant-...",
    shell_enabled=True,                 # MUST enable explicitly
    shell_timeout=30,
    shell_working_dir="/tmp/sandbox",
)

response = await client.responses.create(
    model="claude-sonnet-4-20250514",
    input="List all Python files in the current directory",
    tools=[{"type": "shell"}],
)
```

**Security:** Disabled by default. Must set `shell_enabled=True`.

**Custom executor:**
```python
async def my_shell(command: str) -> dict:
    # Run in container, remote server, etc.
    return {"stdout": "...", "stderr": "", "exit_code": 0}

client = AsyncClient(
    anthropic_api_key="sk-ant-...",
    shell_enabled=True,
    shell_callback=my_shell,
)
```

### MCP

Model Context Protocol — connect to external tool servers.

```python
response = await client.responses.create(
    model="gpt-4.1",
    input="Search for recent orders",
    tools=[
        {
            "type": "mcp",
            "server_label": "orders-api",
            "server_url": "https://mcp.example.com/orders",
            "require_approval": "never",
        },
    ],
)
```

> **Note:** MCP is currently supported natively by OpenAI only. Client-side MCP emulation for other providers is planned.

### File Search

Search across vector stores.

```python
response = await client.responses.create(
    model="gpt-4.1",
    input="What does the refund policy say?",
    tools=[
        {
            "type": "file_search",
            "vector_store_ids": ["vs_abc123"],
            "max_num_results": 5,
        },
    ],
)
```

> **Note:** File search is currently supported natively by OpenAI only.

### Image Generation

Generate images from text descriptions.

```python
response = await client.responses.create(
    model="gpt-4.1",
    input="Generate an image of a sunset over mountains",
    tools=[{"type": "image_generation"}],
)
```

### Computer Use

Control a virtual computer (screenshots, clicks, typing).

```python
response = await client.responses.create(
    model="gpt-4.1",
    input="Open the browser and go to example.com",
    tools=[
        {
            "type": "computer",
            "display_width": 1024,
            "display_height": 768,
            "environment": "browser",
        },
    ],
)
```

### All Tools with All LLMs

The library's **builtin tool handler system** makes every tool available with every provider. When a tool is not natively supported by a provider, the library:

1. **Converts** the tool definition into a synthetic function the model can call
2. **Injects** context into the system prompt so the model understands the tool
3. **Intercepts** the model's function call and executes it client-side
4. **Returns** the result to the model as a function call output

This is completely transparent to your code:

```python
# This works with Claude, Gemini, Sarvam — not just OpenAI
response = await client.responses.create(
    model="claude-sonnet-4-20250514",
    input="Search the web for the latest Python release and tell me about it",
    tools=[
        {"type": "web_search"},           # auto-emulated for non-OpenAI
        {"type": "function", "name": "save_note", ...},  # user function
    ],
)
```

#### Tool Support Matrix

Every tool works with every LLM. The table below shows *how* each tool is handled per provider:

| Tool Type | OpenAI | Anthropic Claude | Google Gemini | Sarvam AI | OpenAI Dependency? |
|---|---|---|---|---|---|
| `function` | Native | Native | Native | Native | No |
| `web_search` | Native | Client-side | Client-side | Client-side | **No** — uses OpenAI > DuckDuckGo > custom callback |
| `code_interpreter` | Native | Client-side | Client-side | Client-side | **No** — local Python subprocess |
| `shell` | Native | Client-side | Client-side | Client-side | **No** — local shell subprocess |
| `mcp` | Native | Client-side | Client-side | Client-side | **No** — direct MCP client (`mcp` SDK) with OpenAI fallback |
| `file_search` | Native | Via OpenAI | Via OpenAI | Via OpenAI | Yes — needs vector store infra |
| `image_generation` | Native | Via OpenAI | Via OpenAI | Via OpenAI | Yes — needs DALL-E |
| `computer_use` | Native | Via OpenAI | Via OpenAI | Via OpenAI | Yes — needs CUA infra |

**Legend:**
- **Native** — the provider handles the tool server-side, zero extra config
- **Client-side** — the library executes the tool locally, no external dependency
- **Via OpenAI** — delegated to Azure OpenAI which handles it server-side (requires `openai_api_key`)

#### Independent tools (no OpenAI needed)

These tools work with all LLMs without any OpenAI dependency:

| Tool | How it works | Fallback chain |
|---|---|---|
| `function` | Native on all providers | Direct |
| `web_search` | Custom callback > Azure OpenAI `web_search` > DuckDuckGo Instant Answer | 3 backends; callback and DuckDuckGo are free |
| `code_interpreter` | Local Python subprocess execution | Custom callback option |
| `shell` | Local shell subprocess execution | Custom callback option |
| `mcp` | Direct MCP client via `mcp` SDK (SSE/stdio transport) | Falls back to OpenAI delegation if SDK not installed |

#### OpenAI-delegated tools

These tools require Azure OpenAI because they depend on infrastructure only OpenAI provides:

| Tool | Why OpenAI is needed |
|---|---|
| `file_search` | Requires OpenAI vector stores for semantic search |
| `image_generation` | Requires DALL-E for image generation |
| `computer_use` | Requires OpenAI Computer Use Agent (CUA) infrastructure |

#### Billing and token tracking

Every delegated tool call tracks token usage separately so you can monitor costs:

```python
# Web search token usage
metrics = client.web_search_metrics
if metrics:
    print(f"Web search: {metrics.total_tokens} tokens, {metrics.total_search_calls} calls")

# All OpenAI-delegated tools (file_search, image_generation, computer_use, mcp fallback)
for tool_type, metrics in client.delegated_tool_metrics.items():
    print(f"{tool_type}: {metrics.total_tokens} tokens, {metrics.total_calls} calls")
```

Each output item from a delegated tool includes a `_delegated_usage` or `_web_search_usage` field with per-call token breakdown for precise billing attribution.

---

## Streaming

```python
stream = await client.responses.create(
    model="claude-sonnet-4-20250514",
    input="Tell me a story",
    stream=True,
)

async for event in stream:
    match event.type:
        # Response lifecycle
        case "response.created":
            print(f"Response started: {event.response['id']}")
        case "response.completed":
            print("\nResponse finished")
        case "response.failed":
            print(f"Error: {event.response}")

        # Text output
        case "response.output_text.delta":
            print(event.delta, end="", flush=True)
        case "response.output_text.done":
            pass  # full text available in event

        # Function calls
        case "response.output_item.added":
            if event.item and event.item.get("type") == "function_call":
                print(f"\nCalling: {event.item['name']}")
        case "response.function_call_arguments.delta":
            pass  # streaming function args
        case "response.function_call_arguments.done":
            pass  # complete function args

        # Reasoning
        case "response.reasoning_text.delta":
            pass  # thinking tokens (if reasoning enabled)

        # Content structure
        case "response.output_item.added":
            pass
        case "response.output_item.done":
            pass
        case "response.content_part.added":
            pass
        case "response.content_part.done":
            pass
```

### All Stream Event Types

| Event Type | Fields | Description |
|---|---|---|
| `response.created` | `response` | Response object created |
| `response.in_progress` | `response` | Processing started |
| `response.completed` | `response` | Successfully completed |
| `response.failed` | `response` | Failed with error |
| `response.incomplete` | `response` | Truncated (max tokens) |
| `response.output_item.added` | `output_index`, `item` | New output item |
| `response.output_item.done` | `output_index`, `item` | Item complete |
| `response.content_part.added` | `output_index`, `content_index`, `part` | New content part |
| `response.content_part.done` | `output_index`, `content_index`, `part` | Part complete |
| `response.output_text.delta` | `output_index`, `delta` | Text chunk |
| `response.output_text.done` | `output_index` | Text complete |
| `response.reasoning_text.delta` | `output_index`, `delta` | Thinking chunk |
| `response.reasoning_text.done` | `output_index` | Thinking complete |
| `response.function_call_arguments.delta` | `output_index`, `delta` | Arg chunk |
| `response.function_call_arguments.done` | `output_index`, `delta` | Args complete |

---

## Multi-Turn Conversations

### Using previous_response_id

Chain responses together for multi-turn conversations:

```python
# First turn
r1 = await client.responses.create(
    model="claude-sonnet-4-20250514",
    input="My name is Alice.",
    instructions="Remember the user's name.",
)

# Second turn - chains automatically
r2 = await client.responses.create(
    model="claude-sonnet-4-20250514",
    input="What's my name?",
    previous_response_id=r1.id,
)
print(r2.output_text)  # "Your name is Alice."
```

**How it works:**
- **OpenAI:** `previous_response_id` is passed to the API (server-side state)
- **Other providers:** The library stores conversation history in memory. When `previous_response_id` is used, the full history is reconstituted and sent to the provider.

### Using explicit input items

```python
response = await client.responses.create(
    model="claude-sonnet-4-20250514",
    input=[
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Nice to meet you, Alice!"},
        {"role": "user", "content": "What's my name?"},
    ],
)
```

---

## Structured Output (JSON Schema)

Force the model to output valid JSON matching a schema:

```python
response = await client.responses.create(
    model="claude-sonnet-4-20250514",
    input="Extract the person's name, age, and city from: 'John is 30 years old and lives in NYC'",
    text={
        "format": {
            "type": "json_schema",
            "name": "person_info",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "city": {"type": "string"},
                },
                "required": ["name", "age", "city"],
            },
        },
    },
)

import json
data = json.loads(response.output_text)
# {"name": "John", "age": 30, "city": "NYC"}
```

The library post-validates the output against the schema and sets `response.error` if validation fails.

---

## Reasoning / Thinking

Enable reasoning (chain-of-thought) tokens:

```python
response = await client.responses.create(
    model="claude-sonnet-4-20250514",
    input="What is 15% of 847?",
    reasoning={"effort": "high"},    # "low" | "medium" | "high"
)
```

**Provider mapping:**
| Effort | Anthropic `budget_tokens` | Gemini `budget_tokens` | OpenAI | Sarvam |
|---|---|---|---|---|
| `low` | 1024 | 1024 | Native `reasoning_effort` | Passed as-is |
| `medium` | 4096 | 4096 | Native | Passed as-is |
| `high` | 16384 | 16384 | Native | Passed as-is |

---

## Agentic Loop

For non-OpenAI providers, the library runs a client-side agentic loop when tools are present:

```
Request -> Provider -> Response with function_call(s)
                              |
                    Execute function handlers
                              |
                    Append results to input
                              |
                    Request -> Provider -> ... (repeat)
                              |
                    Response with text (done)
```

This is **automatic and transparent**. The loop runs up to `agent_loop_max_steps` times (default: 10, configurable via env var or config).

Two-tier dispatch:
1. **Builtin handlers** (web_search, code_interpreter, shell) — handled internally
2. **User handlers** — your registered function callbacks

---

## Cost Tracking

Track token usage and costs across all providers:

```python
from caip_responses import ModelPricing

# Set pricing (per million tokens)
client.cost_tracker.set_pricing("claude-sonnet-4-20250514", ModelPricing(
    input_cost_per_million=3.0,
    output_cost_per_million=15.0,
))
client.cost_tracker.set_pricing("gpt-4.1", ModelPricing(
    input_cost_per_million=2.0,
    output_cost_per_million=8.0,
))

# ... make requests ...

# Query usage
print(f"Total cost: ${client.cost_tracker.total_cost:.4f}")
print(f"Total tokens: {client.cost_tracker.total_tokens}")
print(f"Total requests: {client.cost_tracker.total_requests}")

# Per-model breakdown
for model, usage in client.cost_tracker.by_model().items():
    print(f"  {model}: {usage.requests} requests, ${usage.total_cost_usd:.4f}")

# Per-provider breakdown
for provider, cost in client.cost_tracker.by_provider().items():
    print(f"  {provider}: ${cost:.4f}")

# Full summary
print(client.cost_tracker.summary())

# Reset
client.cost_tracker.reset()
```

---

## Rate Limiting

Configure per-provider rate limits with token bucket algorithm:

```python
from caip_responses import RateLimitConfig

# Configure limits
client.rate_limiter.configure("anthropic", RateLimitConfig(
    requests_per_minute=60,        # 0 = unlimited
    tokens_per_minute=100_000,     # 0 = unlimited
    max_retries=3,
    retry_base_delay=1.0,          # seconds, exponential backoff
))

client.rate_limiter.configure("openai", RateLimitConfig(
    requests_per_minute=500,
    tokens_per_minute=1_000_000,
))

# Rate limiting is applied automatically before every API call
```

---

## Response Caching

Automatic in-memory LRU cache for deterministic requests:

```python
# Enabled by default for temperature=0 requests
client = AsyncClient(
    openai_api_key="sk-...",
    enable_cache=True,      # default
    cache_max_size=500,     # max entries
    cache_ttl=3600,         # TTL in seconds
)

# temperature=0 requests are cached automatically
r1 = await client.responses.create(model="gpt-4.1", input="Hello", temperature=0)
r2 = await client.responses.create(model="gpt-4.1", input="Hello", temperature=0)
# r2 is served from cache

# Check cache stats
print(client.cache.stats)
# {"hits": 1, "misses": 1, "total": 2, "hit_rate_pct": 50.0, "size": 1}

# Manual cache control
client.cache.clear()
client.cache.enabled = False
```

---

## Redis (Production Persistence)

By default, conversation history (`previous_response_id`) and response cache are stored **in-memory**. This works for single-process development but breaks in production when you have multiple workers, pods, or need data to survive restarts.

**Add `redis_url` to switch to Redis-backed persistence automatically:**

```python
client = AsyncClient(
    openai_api_key="sk-...",
    anthropic_api_key="sk-ant-...",
    redis_url="redis://localhost:6379/0",   # <-- that's it
    conversation_ttl=86400,                 # 24 hours (default)
)
```

Or via environment variable:
```bash
CAIP_RESPONSES_REDIS_URL=redis://your-redis-host:6379/0
```

### What gets stored in Redis

| Component | Redis Key Pattern | Default TTL | What's Stored |
|---|---|---|---|
| Conversation history | `caip:conv:{response_id}` | 24 hours | Input items + output items + instructions (for `previous_response_id` chaining) |
| Response cache | `caip:cache:{sha256_hash}` | 1 hour | Full Response objects (for `temperature=0` cache hits) |

### Install

```bash
pip install caip-responses-lib[redis]

# Or with all providers
pip install caip-responses-lib[all]   # includes redis
```

### When you need Redis

| Scenario | In-Memory | Redis |
|---|---|---|
| Single-process dev server | Works | Overkill |
| `uvicorn --workers 4` | Broken (each worker has separate memory) | Works |
| Kubernetes / multiple pods | Broken | Works |
| Server restarts mid-conversation | History lost | History preserved |
| Long-running conversations (hours) | LRU eviction at 1000 entries | TTL-based, no entry limit |

### When you DON'T need Redis

- You only use OpenAI (it has server-side `previous_response_id`)
- You always pass full conversation history in `input` (no `previous_response_id`)
- You don't need response caching
- Single-process, ephemeral sessions

### Azure Redis Cache Setup

```bash
# Azure CLI
az redis create \
    --name caip-redis \
    --resource-group your-rg \
    --location eastus \
    --sku Basic \
    --vm-size c0

# Get connection string
az redis show --name caip-redis --resource-group your-rg --query "hostName"
az redis list-keys --name caip-redis --resource-group your-rg
```

```python
# Azure Redis with SSL
client = AsyncClient(
    redis_url="rediss://:YOUR_ACCESS_KEY@caip-redis.redis.cache.windows.net:6380/0",
    ...
)
```

### Direct usage (advanced)

You can use the Redis stores directly without `AsyncClient`:

```python
from caip_responses.store.redis_store import RedisConversationStore
from caip_responses.cache.redis_cache import RedisResponseCache

# Custom conversation store
store = RedisConversationStore(
    redis_url="redis://localhost:6379/0",
    key_prefix="myapp:conv:",     # custom prefix
    ttl=7200,                     # 2 hours
)

# Custom cache
cache = RedisResponseCache(
    redis_url="redis://localhost:6379/1",   # separate DB
    key_prefix="myapp:cache:",
    default_ttl=1800,             # 30 minutes
)

# Health check
assert store.ping()
assert cache.ping()
```

### No other external services needed

| Feature | Storage | External Service Required? |
|---|---|---|
| Conversation history | In-memory or Redis | No (optional Redis) |
| Response cache | In-memory or Redis | No (optional Redis) |
| Cost tracking | In-memory | No |
| Rate limiting | In-memory | No |
| Provider API calls | HTTP | Just API keys |

The library is designed to work with **zero infrastructure** beyond API keys. Redis is purely optional for production scaling.

---

## Postman Collection

### Setting Up Postman to Test the Library

Since `caip-responses-lib` is a Python library (not an HTTP API), you need a thin wrapper server to test via Postman. Here's how:

### Step 1: Create a FastAPI Wrapper

```python
# server.py
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Any

from caip_responses import AsyncClient

app = FastAPI(title="caip-responses API", version="0.1.0")

# Initialize client with all providers
client = AsyncClient(
    openai_api_key="sk-...",
    anthropic_api_key="sk-ant-...",
    gemini_api_key="AIza...",
    sarvam_api_key="sarvam-...",
    code_interpreter_enabled=True,
    shell_enabled=True,
)


class CreateResponseRequest(BaseModel):
    model: str
    input: str | list[dict[str, Any]] = ""
    instructions: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, str] | None = "auto"
    stream: bool = False
    previous_response_id: str | None = None
    reasoning: dict[str, Any] | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    text: dict[str, Any] | None = None


@app.post("/v1/responses")
async def create_response(req: CreateResponseRequest):
    try:
        if req.stream:
            stream = await client.responses.create(
                model=req.model,
                input=req.input,
                instructions=req.instructions,
                tools=req.tools,
                tool_choice=req.tool_choice,
                stream=True,
                previous_response_id=req.previous_response_id,
                reasoning=req.reasoning,
                temperature=req.temperature,
                max_output_tokens=req.max_output_tokens,
                text=req.text,
            )

            async def event_generator():
                async for event in stream:
                    data = {
                        "type": event.type,
                        "sequence_number": event.sequence_number,
                    }
                    if event.delta:
                        data["delta"] = event.delta
                    if event.response:
                        data["response"] = event.response
                    if event.item:
                        data["item"] = event.item
                    yield f"data: {json.dumps(data)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
            )
        else:
            response = await client.responses.create(
                model=req.model,
                input=req.input,
                instructions=req.instructions,
                tools=req.tools,
                tool_choice=req.tool_choice,
                stream=False,
                previous_response_id=req.previous_response_id,
                reasoning=req.reasoning,
                temperature=req.temperature,
                max_output_tokens=req.max_output_tokens,
                text=req.text,
            )
            return response.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/cost")
async def get_cost():
    return client.cost_tracker.summary()


@app.get("/v1/cache/stats")
async def get_cache_stats():
    return client.cache.stats


@app.on_event("shutdown")
async def shutdown():
    await client.close()
```

### Step 2: Run the Server

```bash
pip install fastapi uvicorn
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### Step 3: Postman Requests

#### Basic Text Completion

```
POST http://localhost:8000/v1/responses
Content-Type: application/json

{
    "model": "claude-sonnet-4-20250514",
    "input": "Explain quantum computing in one paragraph.",
    "temperature": 0.7
}
```

**Expected Response:**
```json
{
    "id": "resp_abc123...",
    "object": "response",
    "model": "claude-sonnet-4-20250514",
    "status": "completed",
    "output": [
        {
            "type": "message",
            "id": "item_xyz...",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "Quantum computing harnesses...",
                    "annotations": []
                }
            ],
            "status": "completed"
        }
    ],
    "usage": {
        "input_tokens": 12,
        "output_tokens": 85,
        "total_tokens": 97
    }
}
```

#### Switch Provider (Same Request Shape)

```json
{
    "model": "gpt-4.1",
    "input": "Explain quantum computing in one paragraph.",
    "temperature": 0.7
}
```

```json
{
    "model": "gemini-2.0-flash",
    "input": "Explain quantum computing in one paragraph.",
    "temperature": 0.7
}
```

#### With System Instructions

```json
{
    "model": "claude-sonnet-4-20250514",
    "input": "What should I cook tonight?",
    "instructions": "You are a French chef. Always suggest French cuisine. Be enthusiastic."
}
```

#### Function Calling

```json
{
    "model": "claude-sonnet-4-20250514",
    "input": "What's the weather in Mumbai?",
    "tools": [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    ]
}
```

#### Web Search (Any Provider)

```json
{
    "model": "claude-sonnet-4-20250514",
    "input": "What were the top news headlines today?",
    "tools": [
        {
            "type": "web_search",
            "search_context_size": "medium"
        }
    ]
}
```

#### Code Interpreter (Any Provider)

```json
{
    "model": "gemini-2.0-flash",
    "input": "Calculate the compound interest on $10000 at 5% for 10 years",
    "tools": [
        {
            "type": "code_interpreter"
        }
    ]
}
```

#### Structured Output (JSON Schema)

```json
{
    "model": "claude-sonnet-4-20250514",
    "input": "Extract info: John Smith, 32, lives in Berlin, works as a software engineer",
    "text": {
        "format": {
            "type": "json_schema",
            "name": "person",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "city": {"type": "string"},
                    "occupation": {"type": "string"}
                },
                "required": ["name", "age", "city", "occupation"]
            }
        }
    }
}
```

#### Reasoning / Thinking

```json
{
    "model": "claude-sonnet-4-20250514",
    "input": "If a train leaves at 2pm going 60mph and another at 3pm going 90mph, when does the second catch up?",
    "reasoning": {
        "effort": "high"
    }
}
```

#### Multi-Turn Conversation

```
POST /v1/responses
{
    "model": "claude-sonnet-4-20250514",
    "input": "My name is Alice and I love Python programming."
}
```

Take the `id` from the response and use it:

```
POST /v1/responses
{
    "model": "claude-sonnet-4-20250514",
    "input": "What's my name and what do I love?",
    "previous_response_id": "resp_abc123..."
}
```

#### Streaming

```json
{
    "model": "claude-sonnet-4-20250514",
    "input": "Write a short story about a robot learning to cook.",
    "stream": true
}
```

Response is Server-Sent Events (SSE):
```
data: {"type": "response.created", "sequence_number": 0, "response": {"id": "resp_..."}}
data: {"type": "response.output_text.delta", "sequence_number": 5, "delta": "Once"}
data: {"type": "response.output_text.delta", "sequence_number": 6, "delta": " upon"}
data: {"type": "response.output_text.delta", "sequence_number": 7, "delta": " a time"}
...
data: {"type": "response.completed", "sequence_number": 100}
data: [DONE]
```

#### Check Cost

```
GET http://localhost:8000/v1/cost
```

#### Check Cache Stats

```
GET http://localhost:8000/v1/cache/stats
```

---

## Backend Integration Guide

### FastAPI Application

```python
# app/main.py
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from caip_responses import AsyncClient

# Global client instance
llm_client: AsyncClient | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_client
    llm_client = AsyncClient(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        sarvam_api_key=os.getenv("SARVAM_API_KEY"),
        code_interpreter_enabled=True,
        shell_enabled=False,       # keep disabled in production
    )
    yield
    await llm_client.close()

app = FastAPI(lifespan=lifespan)

@app.post("/chat")
async def chat(message: str, model: str = "claude-sonnet-4-20250514"):
    response = await llm_client.responses.create(
        model=model,
        input=message,
    )
    return {"reply": response.output_text}
```

### Django Application

```python
# views.py
import asyncio
from django.http import JsonResponse
from caip_responses import Client  # sync client for Django

# Initialize once
client = Client(
    openai_api_key=settings.OPENAI_API_KEY,
    anthropic_api_key=settings.ANTHROPIC_API_KEY,
)

def chat_view(request):
    message = request.POST.get("message", "")
    model = request.POST.get("model", "gpt-4.1")

    response = client.responses.create(
        model=model,
        input=message,
    )
    return JsonResponse({
        "reply": response.output_text,
        "model": response.model,
        "usage": response.usage.model_dump() if response.usage else None,
    })
```

### WebSocket Integration

```python
# For real-time chat with streaming
import json
from fastapi import WebSocket

@app.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket):
    await ws.accept()

    while True:
        data = json.loads(await ws.receive_text())
        model = data.get("model", "claude-sonnet-4-20250514")

        stream = await llm_client.responses.create(
            model=model,
            input=data["message"],
            instructions=data.get("instructions"),
            tools=data.get("tools"),
            stream=True,
        )

        async for event in stream:
            if event.type == "response.output_text.delta":
                await ws.send_json({
                    "type": "delta",
                    "text": event.delta,
                })
            elif event.type == "response.completed":
                await ws.send_json({
                    "type": "done",
                    "response_id": event.response.get("id") if event.response else None,
                })
```

### Multi-Provider Configuration

#### Development (.env)

```bash
# .env
CAIP_RESPONSES_OPENAI_API_KEY=sk-dev-...
CAIP_RESPONSES_ANTHROPIC_API_KEY=sk-ant-dev-...
CAIP_RESPONSES_GEMINI_API_KEY=AIza-dev-...
CAIP_RESPONSES_AGENT_LOOP_MAX_STEPS=5
```

#### Production (Azure App Service / Docker)

```bash
# Environment variables
CAIP_RESPONSES_OPENAI_API_KEY=sk-prod-...
CAIP_RESPONSES_OPENAI_BASE_URL=https://my-resource.openai.azure.com/
CAIP_RESPONSES_ANTHROPIC_API_KEY=sk-ant-prod-...
CAIP_RESPONSES_GEMINI_API_KEY=AIza-prod-...
CAIP_RESPONSES_SARVAM_API_KEY=sarvam-prod-...
CAIP_RESPONSES_REDIS_URL=rediss://:key@caip-redis.redis.cache.windows.net:6380/0
CAIP_RESPONSES_DEFAULT_PROVIDER=anthropic
CAIP_RESPONSES_AGENT_LOOP_MAX_STEPS=10
```

#### Docker

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# API keys via env vars at runtime
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  api:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - redis
    environment:
      CAIP_RESPONSES_OPENAI_API_KEY: ${OPENAI_API_KEY}
      CAIP_RESPONSES_ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      CAIP_RESPONSES_GEMINI_API_KEY: ${GEMINI_API_KEY}
      CAIP_RESPONSES_REDIS_URL: redis://redis:6379/0
```

### Migrating from OpenAI SDK

**Before:**
```python
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key="sk-...")
response = await client.responses.create(
    model="gpt-4.1",
    input=messages,
    tools=tools,
    stream=True,
)
async for event in response:
    ...
```

**After:**
```python
from caip_responses import AsyncClient

client = AsyncClient(
    openai_api_key="sk-...",
    anthropic_api_key="sk-ant-...",  # now supports multiple providers
)
response = await client.responses.create(
    model="gpt-4.1",                # same code, same result
    input=messages,
    tools=tools,
    stream=True,
)
async for event in response:
    ...

# Switch to Claude with ZERO code changes:
response = await client.responses.create(
    model="claude-sonnet-4-20250514",   # just change this line
    input=messages,
    tools=tools,
    stream=True,
)
```

### Production Checklist

- [ ] Set all provider API keys via environment variables (not hardcoded)
- [ ] Configure `CAIP_RESPONSES_REDIS_URL` for multi-worker/pod deployments
- [ ] Configure rate limiting per provider to stay within API limits
- [ ] Set up cost tracking with correct model pricing
- [ ] Keep `shell_enabled=False` unless explicitly needed
- [ ] Keep `code_interpreter_enabled=False` unless explicitly needed
- [ ] Use `async with AsyncClient() as client:` for proper cleanup
- [ ] Set `CAIP_RESPONSES_AGENT_LOOP_MAX_STEPS` appropriate to your use case
- [ ] Monitor `client.cost_tracker.summary()` in production
- [ ] Test failover: if one provider is down, switch model to another
- [ ] Use `previous_response_id` for multi-turn (saves tokens vs. sending full history)

---

## Error Handling

```python
from caip_responses import CaipResponsesError, ProviderError

try:
    response = await client.responses.create(
        model="claude-sonnet-4-20250514",
        input="Hello",
    )
except ProviderError as e:
    print(f"Provider: {e.provider}")
    print(f"Status code: {e.status_code}")
    print(f"Message: {e.message}")
except CaipResponsesError as e:
    print(f"Library error: {e}")
```

**Error hierarchy:**
```
CaipResponsesError (base)
  ProviderError          — API call failed (has provider, status_code, raw_error)
  ProviderNotFoundError  — no provider matched the model name
  ProviderNotConfiguredError — provider matched but no API key set
  MaxStepsExceededError  — agentic loop exceeded max_steps
```

---

## Plugin System

Register custom LLM providers as plugins:

```python
# In your package's pyproject.toml:
[project.entry-points."caip_responses.providers"]
my_llm = "my_package.provider:MyLLMProvider"
```

```python
# my_package/provider.py
from caip_responses.providers.base import BaseProvider

class MyLLMProvider(BaseProvider):
    def __init__(self, api_key: str, **kwargs):
        self._api_key = api_key

    @property
    def provider_name(self) -> str:
        return "my_llm"

    def supports_tool(self, tool_type: str) -> bool:
        return tool_type == "function"

    def supports_reasoning(self) -> bool:
        return False

    async def create_response(self, request):
        # Your implementation
        ...

    async def create_response_stream(self, request):
        # Your streaming implementation
        ...
```

```python
# Usage — plugin is auto-discovered
client = AsyncClient(discover_plugins=True)

# Or register manually
client.plugins.register_factory("my_llm", MyLLMProvider, prefixes=["my-"])

# Now use it
response = await client.responses.create(
    model="my-model-v1",    # routes to your provider
    input="Hello",
)
```
