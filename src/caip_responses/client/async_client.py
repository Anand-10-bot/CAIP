from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from caip_responses.cache.response_cache import ResponseCache
from caip_responses.client.config import CaipResponsesConfig
from caip_responses.cost.tracker import CostTracker
from caip_responses.loop.agent_loop import AgentLoop
from caip_responses.loop.tool_executor import ToolExecutor
from caip_responses.models.common import Reasoning, TextConfig
from caip_responses.models.request import CreateResponseRequest
from caip_responses.models.response import Response
from caip_responses.models.streaming import StreamEvent
from caip_responses.plugins.manager import PluginManager
from caip_responses.providers.base import BaseProvider
from caip_responses.providers.registry import ProviderRegistry
from caip_responses.ratelimit.limiter import RateLimiter
from caip_responses.store.conversation_store import ConversationStore
from caip_responses.tool_handlers.code_interpreter import (
    CodeExecutorCallback,
    CodeInterpreterHandler,
)
from caip_responses.tool_handlers.computer_use import ComputerUseHandler
from caip_responses.tool_handlers.file_search import FileSearchHandler
from caip_responses.tool_handlers.image_generation import ImageGenerationHandler
from caip_responses.tool_handlers.mcp import MCPHandler
from caip_responses.tool_handlers.openai_delegator import DelegatedToolMetrics
from caip_responses.tool_handlers.registry import BuiltinToolRegistry
from caip_responses.tool_handlers.shell import ShellExecutorCallback, ShellHandler
from caip_responses.tool_handlers.web_search import (
    SearchCallback,
    WebSearchHandler,
    WebSearchMetrics,
)
from caip_responses.utils.json_schema import validate_json_against_schema

logger = logging.getLogger(__name__)


class _ResponsesNamespace:
    """Mirrors `client.responses.create()` — the OpenAI SDK call pattern."""

    def __init__(
        self,
        registry: ProviderRegistry,
        store: ConversationStore,
        tool_executor: ToolExecutor,
        config: CaipResponsesConfig,
        rate_limiter: RateLimiter,
        cost_tracker: CostTracker,
        cache: ResponseCache,
        builtin_registry: BuiltinToolRegistry | None = None,
    ) -> None:
        self._registry = registry
        self._store = store
        self._executor = tool_executor
        self._config = config
        self._rate_limiter = rate_limiter
        self._cost_tracker = cost_tracker
        self._cache = cache
        self._builtin = builtin_registry or BuiltinToolRegistry()

    async def create(
        self,
        *,
        model: str,
        input: str | list[dict[str, Any]] = "",
        instructions: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, str] | None = "auto",
        parallel_tool_calls: bool | None = None,
        stream: bool = False,
        previous_response_id: str | None = None,
        reasoning: Reasoning | dict[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_output_tokens: int | None = None,
        text: TextConfig | dict[str, Any] | None = None,
        prompt: dict[str, Any] | None = None,
        metadata: dict[str, str] | None = None,
        store: bool | None = None,
        truncation: str | None = None,
        user: str | None = None,
        include: list[str] | None = None,
        background: bool | None = None,
        provider: str | None = None,
        **kwargs: Any,
    ) -> Response | AsyncIterator[StreamEvent]:
        """Create a model response — identical interface regardless of provider.

        Args:
            model: Model ID (e.g. "gpt-4.1", "claude-sonnet-4-20250514", "gemini-2.0-flash").
                   The provider is auto-detected from the model prefix.
            input: Text string or list of input items.
            instructions: System-level instructions.
            tools: List of tool definitions (function, web_search, mcp, etc.).
            tool_choice: "auto", "required", "none", or specific function.
            parallel_tool_calls: Allow multiple parallel tool calls.
            stream: If True, returns an async iterator of StreamEvents.
            previous_response_id: Chain to a previous response for multi-turn.
            reasoning: Reasoning/thinking configuration (effort: low/medium/high).
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            max_output_tokens: Maximum output tokens.
            text: Text format configuration (plain text or JSON schema).
            prompt: Reusable prompt template (id, version, variables).
            metadata: Key-value metadata.
            store: Whether to persist the response (enables previous_response_id).
            truncation: "auto" or "disabled".
            user: End-user identifier.
            include: Additional data to include in response.
            background: Run in background mode.
            provider: Override auto-detected provider.

        Returns:
            Response object (non-streaming) or AsyncIterator[StreamEvent] (streaming).
        """
        # Resolve provider
        resolved_provider = self._registry.resolve(model, explicit_provider=provider)

        # Reconstitute conversation history from previous_response_id
        effective_input = input
        effective_instructions = instructions
        if previous_response_id and resolved_provider.provider_name != "openai":
            history = self._store.get_history(previous_response_id)
            if history is not None:
                prior_items, prior_instructions = history
                new_input = self._ensure_input_list(effective_input)
                effective_input = prior_items + new_input
                if effective_instructions is None and prior_instructions:
                    effective_instructions = prior_instructions

        # MCP discovery pre-pass: for non-OpenAI providers that don't
        # support MCP natively, connect to each MCP server and expose its
        # tools as individual synthetic functions (with real schemas) so
        # the agent loop can call them directly. Without this, the handler
        # only produces a generic "request" function that can't invoke
        # specific MCP tools. Discovery failures are non-fatal — the
        # handler falls back to its generic function.
        if (
            tools
            and resolved_provider.provider_name != "openai"
            and self._builtin
            and not resolved_provider.supports_tool("mcp")
        ):
            mcp_handler = self._builtin.get("mcp")
            if mcp_handler is not None and hasattr(mcp_handler, "discover_tools"):
                for tool in tools:
                    if not isinstance(tool, dict) or tool.get("type") != "mcp":
                        continue
                    server_url = tool.get("server_url")
                    if not server_url:
                        continue
                    server_label = tool.get("server_label", "mcp_server")
                    try:
                        await mcp_handler.discover_tools(server_label, server_url)
                    except Exception as e:
                        # Server unreachable / discovery failed — fall back
                        # to the handler's generic request function.
                        logger.warning(
                            "MCP discovery failed for server %r (%s): %s",
                            server_label, server_url, e,
                        )

        # Preprocess tools for non-OpenAI providers: replace unsupported
        # built-in tools (web_search, code_interpreter, shell, etc.) with
        # synthetic function definitions the model can call.
        effective_tools = tools
        has_builtin_tools = False
        if tools and resolved_provider.provider_name != "openai" and self._builtin:
            provider_supports = {
                t for t in ("function", "web_search", "code_interpreter",
                            "shell", "file_search", "mcp", "computer",
                            "computer_use", "image_generation", "tool_search")
                if resolved_provider.supports_tool(t)
            }
            effective_tools, addenda, _tool_configs = (
                self._builtin.preprocess_tools(tools, provider_supports)
            )
            if addenda:
                addendum_text = "\n\n".join(addenda)
                if effective_instructions:
                    effective_instructions = (
                        f"{effective_instructions}\n\n{addendum_text}"
                    )
                else:
                    effective_instructions = addendum_text
            # Check if any builtin tools were activated (tools changed)
            has_builtin_tools = len(effective_tools) != len(tools) or any(
                t.get("type") != orig.get("type")
                for t, orig in zip(effective_tools, tools)
                if isinstance(t, dict) and isinstance(orig, dict)
            )

        request = CreateResponseRequest(
            model=model,
            input=effective_input,
            instructions=effective_instructions,
            tools=effective_tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            stream=stream,
            previous_response_id=previous_response_id if resolved_provider.provider_name == "openai" else None,
            reasoning=reasoning,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            text=text,
            prompt=prompt,
            metadata=metadata,
            store=store,
            truncation=truncation,
            user=user,
            include=include,
            background=background,
            provider=provider,
        )

        # Determine if we need the agentic loop (non-OpenAI with tools + handlers)
        needs_loop = (
            resolved_provider.provider_name != "openai"
            and effective_tools
            and (self._executor._handlers or has_builtin_tools)
        )

        provider_name = resolved_provider.provider_name

        if stream:
            # Rate limit before streaming
            await self._rate_limiter.acquire(provider_name)

            if needs_loop:
                loop = AgentLoop(
                    resolved_provider,
                    self._executor,
                    max_steps=self._config.agent_loop_max_steps,
                    builtin_registry=self._builtin,
                )
                return self._wrap_stream_with_store(
                    loop.run_stream(request),
                    request,
                    provider_name=provider_name,
                    store_enabled=store is not False,
                )
            return self._wrap_stream_with_store(
                resolved_provider.create_response_stream(request),
                request,
                provider_name=provider_name,
                store_enabled=store is not False,
            )
        else:
            # Check cache for non-streaming deterministic requests
            cache_key = None
            if self._cache.enabled and not needs_loop and temperature == 0:
                cache_key = self._cache.build_key(
                    model=model,
                    input=effective_input,
                    instructions=effective_instructions,
                    tools=tools,
                    tool_choice=tool_choice,
                    reasoning=reasoning,
                    max_output_tokens=max_output_tokens,
                    text=text,
                )
                cached = self._cache.get(cache_key)
                if cached is not None:
                    return cached

            # Rate limit
            await self._rate_limiter.acquire(provider_name)

            if needs_loop:
                loop = AgentLoop(
                    resolved_provider,
                    self._executor,
                    max_steps=self._config.agent_loop_max_steps,
                    builtin_registry=self._builtin,
                )
                response = await loop.run(request)
            else:
                response = await resolved_provider.create_response(request)

            # Post-validate structured output
            response = self._validate_structured_output(response, text)

            # Track cost
            if response.usage:
                self._cost_tracker.record(
                    model=model,
                    provider=provider_name,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )
                self._rate_limiter.record_tokens(
                    provider_name,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )

            # Store response for previous_response_id support
            if store is not False:
                input_list = self._ensure_input_list(effective_input)
                self._store.save(response, input_list, effective_instructions)

            # Cache the response
            if cache_key is not None and response.error is None:
                self._cache.set(cache_key, response)

            return response

    def _validate_structured_output(
        self, response: Response, text: TextConfig | dict[str, Any] | None
    ) -> Response:
        """Post-validate that output conforms to JSON schema if requested."""
        if text is None:
            return response

        schema = self._extract_json_schema(text)
        if schema is None:
            return response

        output_text = response.output_text
        if not output_text:
            return response

        if not validate_json_against_schema(output_text, schema):
            response.error = {
                "type": "json_schema_validation_error",
                "message": "Response output is not valid JSON matching the requested schema.",
            }

        return response

    @staticmethod
    def _extract_json_schema(
        text: TextConfig | dict[str, Any]
    ) -> dict[str, Any] | None:
        """Extract the JSON schema from a text config, if present."""
        if isinstance(text, dict):
            fmt = text.get("format", {})
            if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
                return fmt.get("schema", None)
        elif hasattr(text, "format") and text.format:
            fmt = text.format
            if hasattr(fmt, "type") and fmt.type == "json_schema":
                return getattr(fmt, "schema_", None)
        return None

    async def _wrap_stream_with_store(
        self,
        stream: AsyncIterator[StreamEvent],
        request: CreateResponseRequest,
        provider_name: str,
        store_enabled: bool,
    ) -> AsyncIterator[StreamEvent]:
        """Wrap a stream to capture the completed response and store it."""
        response_data: dict[str, Any] = {}

        async for event in stream:
            yield event

            if event.type == "response.completed" and event.response:
                response_data = event.response

        # Track cost from streaming usage
        usage = response_data.get("usage")
        if usage:
            self._cost_tracker.record(
                model=request.model,
                provider=provider_name,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            )

        # Store the response
        if store_enabled and response_data.get("id"):
            stored_response = Response(
                id=response_data["id"],
                model=response_data.get("model", request.model),
                status=response_data.get("status", "completed"),
            )
            input_list = self._ensure_input_list(request.input)
            self._store.save(stored_response, input_list, request.instructions)

    @staticmethod
    def _ensure_input_list(
        input_data: str | list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert input to a list of dicts."""
        if isinstance(input_data, str):
            return [{"role": "user", "content": input_data}]
        return list(input_data)


class AsyncClient:
    """Main entry point — drop-in replacement for OpenAI's AsyncOpenAI.

    Usage:
        client = AsyncClient(
            openai_api_key="...",
            anthropic_api_key="...",
        )
        response = await client.responses.create(
            model="claude-sonnet-4-20250514",
            input="Hello",
        )
        print(response.output_text)

    With agentic loop (auto tool calling):
        client = AsyncClient(anthropic_api_key="...")
        client.tools.register("get_weather", my_weather_handler)
        response = await client.responses.create(
            model="claude-sonnet-4-20250514",
            input="What's the weather?",
            tools=[{"type": "function", "name": "get_weather", ...}],
        )

    With cost tracking:
        from caip_responses.cost import ModelPricing
        client.cost_tracker.set_pricing("gpt-4.1", ModelPricing(
            input_cost_per_million=2.0, output_cost_per_million=8.0
        ))
        # ... make requests ...
        print(client.cost_tracker.total_cost)

    With rate limiting:
        from caip_responses.ratelimit import RateLimitConfig
        client.rate_limiter.configure("anthropic", RateLimitConfig(
            requests_per_minute=60
        ))

    With response caching:
        # Enabled by default for temperature=0 requests
        client.cache.enabled = True
    """

    def __init__(
        self,
        *,
        openai_api_key: str | None = None,
        openai_base_url: str | None = None,
        anthropic_api_key: str | None = None,
        anthropic_base_url: str | None = None,
        gemini_api_key: str | None = None,
        gemini_vertexai: bool | None = None,
        gemini_project: str | None = None,
        gemini_location: str | None = None,
        gemini_service_account_path: str | None = None,
        sarvam_api_key: str | None = None,
        sarvam_base_url: str | None = None,
        ollama_api_key: str | None = None,
        ollama_base_url: str | None = None,
        default_provider: str | None = None,
        providers: dict[str, BaseProvider] | None = None,
        max_conversation_history: int = 1000,
        cache_max_size: int = 500,
        cache_ttl: int = 3600,
        enable_cache: bool = True,
        discover_plugins: bool = True,
        # Built-in tool handler configuration
        web_search_model: str = "gpt-4.1-nano",
        web_search_callback: SearchCallback | None = None,
        code_interpreter_enabled: bool = False,
        code_interpreter_timeout: int = 30,
        code_interpreter_working_dir: str | None = None,
        code_interpreter_callback: CodeExecutorCallback | None = None,
        shell_enabled: bool = False,
        shell_timeout: int = 30,
        shell_working_dir: str | None = None,
        shell_callback: ShellExecutorCallback | None = None,
        shell_command_allowlist: set[str] | frozenset[str] | None = None,
        builtin_registry: BuiltinToolRegistry | None = None,
        # Redis (for persistent conversation store + cache)
        redis_url: str | None = None,
        conversation_ttl: int = 86400,
    ) -> None:
        # Load config from env vars as fallback
        config = CaipResponsesConfig()

        self._registry = ProviderRegistry(
            default_provider=default_provider or config.default_provider or None
        )

        # Resolve Redis URL (constructor param > env var)
        effective_redis_url = redis_url or config.redis_url or None

        # Conversation store for previous_response_id on non-OpenAI
        # Uses Redis when redis_url is provided, otherwise in-memory.
        if effective_redis_url:
            from caip_responses.store.redis_store import RedisConversationStore
            self._store: ConversationStore | RedisConversationStore = (
                RedisConversationStore(
                    redis_url=effective_redis_url,
                    ttl=conversation_ttl or config.conversation_ttl,
                )
            )
        else:
            self._store = ConversationStore(max_size=max_conversation_history)

        # Tool executor for agentic loop
        self._tool_executor = ToolExecutor()

        # Rate limiter
        self._rate_limiter = RateLimiter()

        # Cost tracker
        self._cost_tracker = CostTracker()

        # Response cache
        # Uses Redis when redis_url is provided, otherwise in-memory.
        if effective_redis_url:
            from caip_responses.cache.redis_cache import RedisResponseCache
            self._cache: ResponseCache | RedisResponseCache = RedisResponseCache(
                redis_url=effective_redis_url,
                max_size=cache_max_size,
                default_ttl=cache_ttl,
                enabled=enable_cache,
            )
        else:
            self._cache = ResponseCache(
                max_size=cache_max_size,
                default_ttl=cache_ttl,
                enabled=enable_cache,
            )

        # Plugin manager
        self._plugin_manager = PluginManager(self._registry)

        # Built-in tool handler registry — makes all tools available
        # with all LLMs by converting unsupported tool types to synthetic
        # function calls that any model can invoke.
        self._builtin_registry = builtin_registry if builtin_registry is not None else self._build_builtin_registry(
            config=config,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            web_search_model=web_search_model,
            web_search_callback=web_search_callback,
            code_interpreter_enabled=code_interpreter_enabled,
            code_interpreter_timeout=code_interpreter_timeout,
            code_interpreter_working_dir=code_interpreter_working_dir,
            code_interpreter_callback=code_interpreter_callback,
            shell_enabled=shell_enabled,
            shell_timeout=shell_timeout,
            shell_working_dir=shell_working_dir,
            shell_callback=shell_callback,
            shell_command_allowlist=shell_command_allowlist,
        )

        # Discover entry-point plugins
        if discover_plugins:
            self._plugin_manager.discover_entry_points()

        # Register custom providers first
        if providers:
            for name, prov in providers.items():
                self._registry.register(name, prov)

        # Initialize built-in providers based on supplied keys
        self._init_openai(
            api_key=openai_api_key or config.openai_api_key or None,
            base_url=openai_base_url or config.openai_base_url or None,
        )
        self._init_anthropic(
            api_key=anthropic_api_key or config.anthropic_api_key or None,
            base_url=anthropic_base_url or config.anthropic_base_url or None,
        )
        self._init_gemini(
            api_key=gemini_api_key or config.gemini_api_key or None,
            vertexai=gemini_vertexai if gemini_vertexai is not None else config.gemini_vertexai,
            project=gemini_project or config.gemini_project or None,
            location=gemini_location or config.gemini_location or "us-central1",
            service_account_path=(
                gemini_service_account_path
                or config.gemini_service_account_path
                or None
            ),
        )
        self._init_sarvam(
            api_key=sarvam_api_key or config.sarvam_api_key or None,
            base_url=sarvam_base_url or config.sarvam_base_url or None,
        )
        self._init_ollama(
            api_key=ollama_api_key or config.ollama_api_key or None,
            base_url=ollama_base_url or config.ollama_base_url or None,
        )

        self._responses = _ResponsesNamespace(
            self._registry,
            self._store,
            self._tool_executor,
            config,
            self._rate_limiter,
            self._cost_tracker,
            self._cache,
            builtin_registry=self._builtin_registry,
        )

    @property
    def responses(self) -> _ResponsesNamespace:
        """Access responses.create() — mirrors OpenAI SDK structure."""
        return self._responses

    @property
    def tools(self) -> ToolExecutor:
        """Access the tool executor to register function handlers."""
        return self._tool_executor

    @property
    def conversation_store(self) -> ConversationStore:
        """Access the conversation store (for previous_response_id support)."""
        return self._store

    @property
    def rate_limiter(self) -> RateLimiter:
        """Access the rate limiter to configure per-provider limits."""
        return self._rate_limiter

    @property
    def cost_tracker(self) -> CostTracker:
        """Access the cost tracker to set pricing and query usage."""
        return self._cost_tracker

    @property
    def cache(self) -> ResponseCache:
        """Access the response cache."""
        return self._cache

    @property
    def plugins(self) -> PluginManager:
        """Access the plugin manager to register custom providers."""
        return self._plugin_manager

    @property
    def builtin_tools(self) -> BuiltinToolRegistry:
        """Access the builtin tool registry to register/configure handlers."""
        return self._builtin_registry

    @property
    def web_search_metrics(self) -> WebSearchMetrics | None:
        """Token usage metrics from web search calls routed through OpenAI.

        Returns None if no web search handler is registered.
        Use this to track billing for web search operations separately
        from the primary LLM calls.

        Example::

            metrics = client.web_search_metrics
            if metrics:
                print(f"Web search tokens used: {metrics.total_tokens}")
                print(f"Web search calls: {metrics.total_search_calls}")
                print(f"Per-call breakdown: {metrics.to_dict()}")
        """
        handler = self._builtin_registry.get("web_search")
        if isinstance(handler, WebSearchHandler):
            return handler.metrics
        return None

    @property
    def delegated_tool_metrics(self) -> dict[str, DelegatedToolMetrics]:
        """Token usage metrics for all tools delegated to Azure OpenAI.

        Returns a dict mapping tool type names to their metrics.
        This covers MCP, file_search, image_generation, and computer_use.

        Example::

            for tool_type, metrics in client.delegated_tool_metrics.items():
                print(f"{tool_type}: {metrics.total_tokens} tokens, {metrics.total_calls} calls")
        """
        result: dict[str, DelegatedToolMetrics] = {}
        for tool_type in ("mcp", "file_search", "image_generation", "computer_use"):
            handler = self._builtin_registry.get(tool_type)
            if handler and hasattr(handler, "metrics"):
                result[tool_type] = handler.metrics
        return result

    async def close(self) -> None:
        """Cleanup all provider resources."""
        for name in self._registry.registered_providers:
            provider = self._registry.get(name)
            if provider:
                await provider.close()

    async def __aenter__(self) -> AsyncClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Built-in tool handler setup
    # ------------------------------------------------------------------

    @staticmethod
    def _build_builtin_registry(
        *,
        config: CaipResponsesConfig,
        openai_api_key: str | None,
        openai_base_url: str | None,
        web_search_model: str,
        web_search_callback: SearchCallback | None,
        code_interpreter_enabled: bool,
        code_interpreter_timeout: int,
        code_interpreter_working_dir: str | None,
        code_interpreter_callback: CodeExecutorCallback | None,
        shell_enabled: bool,
        shell_timeout: int,
        shell_working_dir: str | None,
        shell_callback: ShellExecutorCallback | None,
        shell_command_allowlist: set[str] | frozenset[str] | None,
    ) -> BuiltinToolRegistry:
        """Build the default builtin tool registry from config."""
        registry = BuiltinToolRegistry()

        # Web search — uses Azure OpenAI's built-in web_search tool.
        # Falls back to DuckDuckGo if no OpenAI key is configured.
        effective_openai_key = openai_api_key or config.openai_api_key or None
        effective_openai_base = openai_base_url or config.openai_base_url or None
        registry.register(WebSearchHandler(
            openai_api_key=effective_openai_key,
            openai_base_url=effective_openai_base,
            openai_model=web_search_model,
            search_callback=web_search_callback,
        ))

        # Code interpreter — registered but disabled by default
        registry.register(CodeInterpreterHandler(
            enabled=code_interpreter_enabled,
            timeout=code_interpreter_timeout,
            working_dir=code_interpreter_working_dir,
            executor_callback=code_interpreter_callback,
        ))

        # Shell — registered but disabled by default
        registry.register(ShellHandler(
            enabled=shell_enabled,
            timeout=shell_timeout,
            working_dir=shell_working_dir,
            executor_callback=shell_callback,
            command_allowlist=shell_command_allowlist,
        ))

        # MCP — delegates to Azure OpenAI for MCP server connections
        registry.register(MCPHandler(
            openai_api_key=effective_openai_key,
            openai_base_url=effective_openai_base,
        ))

        # File search — delegates to Azure OpenAI for vector store search
        registry.register(FileSearchHandler(
            openai_api_key=effective_openai_key,
            openai_base_url=effective_openai_base,
        ))

        # Image generation — delegates to Azure OpenAI for DALL-E
        registry.register(ImageGenerationHandler(
            openai_api_key=effective_openai_key,
            openai_base_url=effective_openai_base,
        ))

        # Computer use — delegates to Azure OpenAI for CUA
        registry.register(ComputerUseHandler(
            openai_api_key=effective_openai_key,
            openai_base_url=effective_openai_base,
        ))

        return registry

    # ------------------------------------------------------------------
    # Provider initialization (lazy — only if key is provided)
    # ------------------------------------------------------------------

    def _init_openai(self, api_key: str | None, base_url: str | None) -> None:
        if not api_key:
            return
        try:
            from caip_responses.providers.openai_provider import OpenAIProvider
            provider = OpenAIProvider(api_key=api_key, base_url=base_url)
            self._registry.register("openai", provider)
        except ImportError:
            pass

    def _init_anthropic(self, api_key: str | None, base_url: str | None) -> None:
        if not api_key:
            return
        try:
            from caip_responses.providers.anthropic_provider import AnthropicProvider
            provider = AnthropicProvider(api_key=api_key, base_url=base_url)
            self._registry.register("anthropic", provider)
        except ImportError:
            pass

    def _init_gemini(
        self,
        api_key: str | None,
        *,
        vertexai: bool = False,
        project: str | None = None,
        location: str = "us-central1",
        service_account_path: str | None = None,
    ) -> None:
        # Register if we have either a Developer API key or Vertex AI auth
        # (a service account path, or vertexai=True for default credentials).
        if not (api_key or service_account_path or vertexai):
            return
        try:
            from caip_responses.providers.gemini_provider import GeminiProvider
            provider = GeminiProvider(
                api_key=api_key,
                vertexai=vertexai,
                project=project,
                location=location,
                service_account_path=service_account_path,
            )
            self._registry.register("gemini", provider)
        except ImportError:
            pass

    def _init_sarvam(self, api_key: str | None, base_url: str | None) -> None:
        if not api_key:
            return
        try:
            from caip_responses.providers.sarvam_provider import SarvamProvider
            provider = SarvamProvider(
                api_key=api_key,
                base_url=base_url or "https://api.sarvam.ai/v1",
            )
            self._registry.register("sarvam", provider)
        except ImportError:
            pass

    def _init_ollama(self, api_key: str | None, base_url: str | None) -> None:
        # Ollama/vLLM/LM Studio rarely need an API key, so registration is
        # gated on base_url (the one setting that's always required).
        if not base_url:
            return
        try:
            from caip_responses.providers.ollama_provider import OllamaProvider
            provider = OllamaProvider(api_key=api_key, base_url=base_url)
            self._registry.register("ollama", provider)
        except ImportError:
            pass
