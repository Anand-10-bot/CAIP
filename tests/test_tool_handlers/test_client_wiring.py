"""Tests that AsyncClient correctly wires up builtin tool handlers.

Verifies:
- Default registry is created with all handlers (web_search, code_interpreter,
  shell, mcp, file_search, image_generation, computer_use)
- Custom config params are passed through
- Tool preprocessing replaces unsupported tools for non-OpenAI providers
- System prompt addenda are injected into instructions
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from caip_responses.client.async_client import AsyncClient, _ResponsesNamespace
from caip_responses.client.config import CaipResponsesConfig
from caip_responses.loop.tool_executor import ToolExecutor
from caip_responses.models.request import CreateResponseRequest
from caip_responses.models.response import Response
from caip_responses.models.streaming import StreamEvent
from caip_responses.providers.base import BaseProvider
from caip_responses.providers.registry import ProviderRegistry
from caip_responses.tool_handlers.code_interpreter import CodeInterpreterHandler
from caip_responses.tool_handlers.computer_use import ComputerUseHandler
from caip_responses.tool_handlers.file_search import FileSearchHandler
from caip_responses.tool_handlers.image_generation import ImageGenerationHandler
from caip_responses.tool_handlers.mcp import MCPHandler
from caip_responses.tool_handlers.registry import BuiltinToolRegistry
from caip_responses.tool_handlers.shell import ShellHandler
from caip_responses.tool_handlers.web_search import WebSearchHandler


class _MockAnthropicProvider(BaseProvider):
    """Mock non-OpenAI provider that only supports function tools."""

    def __init__(self) -> None:
        self.last_request: CreateResponseRequest | None = None

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def supports_tool(self, tool_type: str) -> bool:
        return tool_type == "function"

    def supports_reasoning(self) -> bool:
        return True

    async def create_response(self, request: CreateResponseRequest) -> Response:
        self.last_request = request
        return Response(
            id="resp_mock",
            model=request.model,
            output=[{
                "type": "message",
                "id": "item_1",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello", "annotations": []}],
                "status": "completed",
            }],
        )

    async def create_response_stream(
        self, request: CreateResponseRequest
    ) -> AsyncIterator[StreamEvent]:
        self.last_request = request
        yield StreamEvent(type="response.created", sequence_number=0)
        yield StreamEvent(type="response.completed", sequence_number=1)


class TestAsyncClientBuiltinRegistry:
    def test_default_registry_has_all_handlers(self):
        """Default client creates a registry with ALL tool handlers."""
        client = AsyncClient(discover_plugins=False)
        reg = client.builtin_tools
        assert reg.can_handle("web_search")
        assert reg.can_handle("code_interpreter")
        assert reg.can_handle("shell")
        assert reg.can_handle("mcp")
        assert reg.can_handle("file_search")
        assert reg.can_handle("image_generation")
        assert reg.can_handle("computer_use")

    def test_registered_types_complete(self):
        """All 7 tool types are registered."""
        client = AsyncClient(discover_plugins=False)
        types = sorted(client.builtin_tools.registered_types)
        assert types == [
            "code_interpreter",
            "computer_use",
            "file_search",
            "image_generation",
            "mcp",
            "shell",
            "web_search",
        ]

    def test_openai_key_passed_to_web_search(self):
        """OpenAI credentials are passed to WebSearchHandler."""
        client = AsyncClient(
            openai_api_key="sk-test-key",
            openai_base_url="https://my-resource.openai.azure.com",
            discover_plugins=False,
        )
        handler = client.builtin_tools.get("web_search")
        assert isinstance(handler, WebSearchHandler)
        assert handler._openai_api_key == "sk-test-key"
        assert handler._openai_base_url == "https://my-resource.openai.azure.com"

    def test_openai_key_passed_to_mcp(self):
        """OpenAI credentials are passed to MCPHandler."""
        client = AsyncClient(
            openai_api_key="sk-test-key",
            discover_plugins=False,
        )
        handler = client.builtin_tools.get("mcp")
        assert isinstance(handler, MCPHandler)
        assert handler._openai_api_key == "sk-test-key"

    def test_openai_key_passed_to_file_search(self):
        """OpenAI credentials are passed to FileSearchHandler."""
        client = AsyncClient(
            openai_api_key="sk-test-key",
            discover_plugins=False,
        )
        handler = client.builtin_tools.get("file_search")
        assert isinstance(handler, FileSearchHandler)
        assert handler._openai_api_key == "sk-test-key"

    def test_openai_key_passed_to_image_generation(self):
        """OpenAI credentials are passed to ImageGenerationHandler."""
        client = AsyncClient(
            openai_api_key="sk-test-key",
            discover_plugins=False,
        )
        handler = client.builtin_tools.get("image_generation")
        assert isinstance(handler, ImageGenerationHandler)
        assert handler._openai_api_key == "sk-test-key"

    def test_openai_key_passed_to_computer_use(self):
        """OpenAI credentials are passed to ComputerUseHandler."""
        client = AsyncClient(
            openai_api_key="sk-test-key",
            discover_plugins=False,
        )
        handler = client.builtin_tools.get("computer_use")
        assert isinstance(handler, ComputerUseHandler)
        assert handler._openai_api_key == "sk-test-key"

    def test_code_interpreter_enabled(self):
        """Code interpreter can be enabled via client param."""
        client = AsyncClient(
            code_interpreter_enabled=True,
            code_interpreter_timeout=60,
            discover_plugins=False,
        )
        handler = client.builtin_tools.get("code_interpreter")
        assert isinstance(handler, CodeInterpreterHandler)
        assert handler._enabled is True
        assert handler._timeout == 60

    def test_shell_enabled(self):
        """Shell can be enabled via client param."""
        client = AsyncClient(
            shell_enabled=True,
            shell_timeout=120,
            shell_working_dir="/tmp/work",
            discover_plugins=False,
        )
        handler = client.builtin_tools.get("shell")
        assert isinstance(handler, ShellHandler)
        assert handler._enabled is True
        assert handler._timeout == 120
        assert handler._working_dir == "/tmp/work"

    def test_custom_registry_overrides_default(self):
        """Passing a builtin_registry param uses it instead of building one."""
        custom_reg = BuiltinToolRegistry()
        client = AsyncClient(
            builtin_registry=custom_reg,
            discover_plugins=False,
        )
        assert client.builtin_tools is custom_reg

    def test_web_search_metrics_property(self):
        """Client exposes web search metrics."""
        client = AsyncClient(discover_plugins=False)
        metrics = client.web_search_metrics
        assert metrics is not None
        assert metrics.total_tokens == 0

    def test_delegated_tool_metrics_property(self):
        """Client exposes delegated tool metrics for all OpenAI-backed handlers."""
        client = AsyncClient(discover_plugins=False)
        metrics = client.delegated_tool_metrics
        assert "mcp" in metrics
        assert "file_search" in metrics
        assert "image_generation" in metrics
        assert "computer_use" in metrics
        for tool_type, m in metrics.items():
            assert m.total_calls == 0


class TestResponsesNamespaceToolPreprocessing:
    """Test that _ResponsesNamespace.create() preprocesses tools for non-OpenAI."""

    @pytest.mark.asyncio
    async def test_web_search_replaced_for_anthropic(self):
        """web_search tool is replaced with synthetic function for anthropic."""
        mock_provider = _MockAnthropicProvider()

        reg = ProviderRegistry()
        reg.register("anthropic", mock_provider)

        async def mock_search(query: str, num_results: int) -> list[dict]:
            return []

        builtin_reg = BuiltinToolRegistry()
        builtin_reg.register(WebSearchHandler(search_callback=mock_search))

        config = CaipResponsesConfig()
        ns = _ResponsesNamespace(
            registry=reg,
            store=_make_store(),
            tool_executor=ToolExecutor(),
            config=config,
            rate_limiter=_make_limiter(),
            cost_tracker=_make_cost_tracker(),
            cache=_make_cache(),
            builtin_registry=builtin_reg,
        )

        await ns.create(
            model="claude-sonnet-4-20250514",
            input="Search for news",
            tools=[
                {"type": "web_search", "search_context_size": "medium"},
                {"type": "function", "name": "my_fn", "parameters": {}},
            ],
        )

        # Verify the request sent to provider has synthetic function, not web_search
        req = mock_provider.last_request
        assert req is not None
        tool_types = [t.get("type") for t in req.tools]
        assert "web_search" not in tool_types
        assert all(t == "function" for t in tool_types)

        tool_names = [t.get("name") for t in req.tools]
        assert "_builtin_web_search_query" in tool_names
        assert "my_fn" in tool_names

    @pytest.mark.asyncio
    async def test_mcp_replaced_for_anthropic(self):
        """MCP tool is replaced with synthetic function for anthropic."""
        mock_provider = _MockAnthropicProvider()

        reg = ProviderRegistry()
        reg.register("anthropic", mock_provider)

        builtin_reg = BuiltinToolRegistry()
        builtin_reg.register(MCPHandler())

        config = CaipResponsesConfig()
        ns = _ResponsesNamespace(
            registry=reg,
            store=_make_store(),
            tool_executor=ToolExecutor(),
            config=config,
            rate_limiter=_make_limiter(),
            cost_tracker=_make_cost_tracker(),
            cache=_make_cache(),
            builtin_registry=builtin_reg,
        )

        await ns.create(
            model="claude-sonnet-4-20250514",
            input="Use MCP",
            tools=[{"type": "mcp", "server_label": "my_server", "server_url": "https://example.com"}],
        )

        req = mock_provider.last_request
        assert req is not None
        tool_types = [t.get("type") for t in req.tools]
        assert "mcp" not in tool_types
        assert all(t == "function" for t in tool_types)
        tool_names = [t.get("name") for t in req.tools]
        assert "_builtin_mcp_my_server" in tool_names

    @pytest.mark.asyncio
    async def test_file_search_replaced_for_anthropic(self):
        """file_search tool is replaced with synthetic function."""
        mock_provider = _MockAnthropicProvider()

        reg = ProviderRegistry()
        reg.register("anthropic", mock_provider)

        builtin_reg = BuiltinToolRegistry()
        builtin_reg.register(FileSearchHandler())

        config = CaipResponsesConfig()
        ns = _ResponsesNamespace(
            registry=reg,
            store=_make_store(),
            tool_executor=ToolExecutor(),
            config=config,
            rate_limiter=_make_limiter(),
            cost_tracker=_make_cost_tracker(),
            cache=_make_cache(),
            builtin_registry=builtin_reg,
        )

        await ns.create(
            model="claude-sonnet-4-20250514",
            input="Search files",
            tools=[{"type": "file_search", "vector_store_ids": ["vs_123"]}],
        )

        req = mock_provider.last_request
        assert req is not None
        tool_types = [t.get("type") for t in req.tools]
        assert "file_search" not in tool_types
        tool_names = [t.get("name") for t in req.tools]
        assert "_builtin_file_search_query" in tool_names

    @pytest.mark.asyncio
    async def test_image_generation_replaced_for_anthropic(self):
        """image_generation tool is replaced with synthetic function."""
        mock_provider = _MockAnthropicProvider()

        reg = ProviderRegistry()
        reg.register("anthropic", mock_provider)

        builtin_reg = BuiltinToolRegistry()
        builtin_reg.register(ImageGenerationHandler())

        config = CaipResponsesConfig()
        ns = _ResponsesNamespace(
            registry=reg,
            store=_make_store(),
            tool_executor=ToolExecutor(),
            config=config,
            rate_limiter=_make_limiter(),
            cost_tracker=_make_cost_tracker(),
            cache=_make_cache(),
            builtin_registry=builtin_reg,
        )

        await ns.create(
            model="claude-sonnet-4-20250514",
            input="Generate an image",
            tools=[{"type": "image_generation"}],
        )

        req = mock_provider.last_request
        assert req is not None
        tool_types = [t.get("type") for t in req.tools]
        assert "image_generation" not in tool_types
        tool_names = [t.get("name") for t in req.tools]
        assert "_builtin_image_generation_create" in tool_names

    @pytest.mark.asyncio
    async def test_computer_use_replaced_for_anthropic(self):
        """computer_use tool is replaced with synthetic function."""
        mock_provider = _MockAnthropicProvider()

        reg = ProviderRegistry()
        reg.register("anthropic", mock_provider)

        builtin_reg = BuiltinToolRegistry()
        builtin_reg.register(ComputerUseHandler())

        config = CaipResponsesConfig()
        ns = _ResponsesNamespace(
            registry=reg,
            store=_make_store(),
            tool_executor=ToolExecutor(),
            config=config,
            rate_limiter=_make_limiter(),
            cost_tracker=_make_cost_tracker(),
            cache=_make_cache(),
            builtin_registry=builtin_reg,
        )

        await ns.create(
            model="claude-sonnet-4-20250514",
            input="Click the button",
            tools=[{"type": "computer_use"}],
        )

        req = mock_provider.last_request
        assert req is not None
        tool_types = [t.get("type") for t in req.tools]
        assert "computer_use" not in tool_types
        tool_names = [t.get("name") for t in req.tools]
        assert "_builtin_computer_use_action" in tool_names

    @pytest.mark.asyncio
    async def test_all_tools_replaced_for_anthropic(self):
        """ALL non-function tools replaced with synthetics for anthropic."""
        mock_provider = _MockAnthropicProvider()

        reg = ProviderRegistry()
        reg.register("anthropic", mock_provider)

        builtin_reg = BuiltinToolRegistry()
        builtin_reg.register(WebSearchHandler())
        builtin_reg.register(MCPHandler())
        builtin_reg.register(FileSearchHandler())
        builtin_reg.register(ImageGenerationHandler())
        builtin_reg.register(ComputerUseHandler())
        builtin_reg.register(ShellHandler(enabled=True))
        builtin_reg.register(CodeInterpreterHandler(enabled=True))

        config = CaipResponsesConfig()
        ns = _ResponsesNamespace(
            registry=reg,
            store=_make_store(),
            tool_executor=ToolExecutor(),
            config=config,
            rate_limiter=_make_limiter(),
            cost_tracker=_make_cost_tracker(),
            cache=_make_cache(),
            builtin_registry=builtin_reg,
        )

        await ns.create(
            model="claude-sonnet-4-20250514",
            input="Do everything",
            tools=[
                {"type": "web_search"},
                {"type": "mcp", "server_label": "srv"},
                {"type": "file_search"},
                {"type": "image_generation"},
                {"type": "computer_use"},
                {"type": "shell"},
                {"type": "code_interpreter"},
                {"type": "function", "name": "my_fn", "parameters": {}},
            ],
        )

        req = mock_provider.last_request
        assert req is not None
        # Every tool should be type=function now
        for tool in req.tools:
            assert tool.get("type") == "function", f"Expected function, got {tool}"

    @pytest.mark.asyncio
    async def test_instructions_get_addenda(self):
        """System prompt addenda from handlers are appended to instructions."""
        mock_provider = _MockAnthropicProvider()

        reg = ProviderRegistry()
        reg.register("anthropic", mock_provider)

        builtin_reg = BuiltinToolRegistry()
        builtin_reg.register(ShellHandler(enabled=True))

        config = CaipResponsesConfig()
        ns = _ResponsesNamespace(
            registry=reg,
            store=_make_store(),
            tool_executor=ToolExecutor(),
            config=config,
            rate_limiter=_make_limiter(),
            cost_tracker=_make_cost_tracker(),
            cache=_make_cache(),
            builtin_registry=builtin_reg,
        )

        await ns.create(
            model="claude-sonnet-4-20250514",
            input="Run a command",
            instructions="Be helpful.",
            tools=[{"type": "shell"}],
        )

        req = mock_provider.last_request
        assert req is not None
        # Instructions should have the shell addendum appended
        assert "Be helpful." in req.instructions
        assert "shell" in req.instructions.lower()

    @pytest.mark.asyncio
    async def test_addenda_used_as_instructions_when_none(self):
        """When instructions is None, addenda become the instructions."""
        mock_provider = _MockAnthropicProvider()

        reg = ProviderRegistry()
        reg.register("anthropic", mock_provider)

        builtin_reg = BuiltinToolRegistry()
        builtin_reg.register(CodeInterpreterHandler(enabled=True))

        config = CaipResponsesConfig()
        ns = _ResponsesNamespace(
            registry=reg,
            store=_make_store(),
            tool_executor=ToolExecutor(),
            config=config,
            rate_limiter=_make_limiter(),
            cost_tracker=_make_cost_tracker(),
            cache=_make_cache(),
            builtin_registry=builtin_reg,
        )

        await ns.create(
            model="claude-sonnet-4-20250514",
            input="Run some code",
            tools=[{"type": "code_interpreter"}],
        )

        req = mock_provider.last_request
        assert req is not None
        assert req.instructions is not None
        assert "Python" in req.instructions

    @pytest.mark.asyncio
    async def test_no_preprocessing_for_openai(self):
        """OpenAI provider should NOT have tools preprocessed."""

        class _MockOpenAIProvider(BaseProvider):
            def __init__(self) -> None:
                self.last_request: CreateResponseRequest | None = None

            @property
            def provider_name(self) -> str:
                return "openai"

            def supports_tool(self, tool_type: str) -> bool:
                return True

            def supports_reasoning(self) -> bool:
                return True

            async def create_response(self, request: CreateResponseRequest) -> Response:
                self.last_request = request
                return Response(
                    id="resp_oai",
                    model=request.model,
                    output=[{
                        "type": "message",
                        "id": "item_1",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "OK", "annotations": []}],
                        "status": "completed",
                    }],
                )

            async def create_response_stream(
                self, request: CreateResponseRequest
            ) -> AsyncIterator[StreamEvent]:
                yield StreamEvent(type="response.created", sequence_number=0)

        mock_provider = _MockOpenAIProvider()
        reg = ProviderRegistry()
        reg.register("openai", mock_provider)

        builtin_reg = BuiltinToolRegistry()
        builtin_reg.register(WebSearchHandler())

        config = CaipResponsesConfig()
        ns = _ResponsesNamespace(
            registry=reg,
            store=_make_store(),
            tool_executor=ToolExecutor(),
            config=config,
            rate_limiter=_make_limiter(),
            cost_tracker=_make_cost_tracker(),
            cache=_make_cache(),
            builtin_registry=builtin_reg,
        )

        await ns.create(
            model="gpt-4.1",
            input="Search for news",
            tools=[{"type": "web_search"}],
        )

        req = mock_provider.last_request
        assert req is not None
        # Tools should be unchanged — no preprocessing for OpenAI
        assert req.tools[0]["type"] == "web_search"

    @pytest.mark.asyncio
    async def test_function_only_tools_no_preprocessing(self):
        """If all tools are functions (provider-supported), no preprocessing occurs."""
        mock_provider = _MockAnthropicProvider()

        reg = ProviderRegistry()
        reg.register("anthropic", mock_provider)

        builtin_reg = BuiltinToolRegistry()
        builtin_reg.register(WebSearchHandler())

        config = CaipResponsesConfig()
        ns = _ResponsesNamespace(
            registry=reg,
            store=_make_store(),
            tool_executor=ToolExecutor(),
            config=config,
            rate_limiter=_make_limiter(),
            cost_tracker=_make_cost_tracker(),
            cache=_make_cache(),
            builtin_registry=builtin_reg,
        )

        tools = [{"type": "function", "name": "my_fn", "parameters": {}}]
        await ns.create(
            model="claude-sonnet-4-20250514",
            input="Hello",
            tools=tools,
        )

        req = mock_provider.last_request
        assert req is not None
        assert req.tools == tools


# ------------------------------------------------------------------
# Helper factories for creating test dependencies
# ------------------------------------------------------------------

def _make_store():
    from caip_responses.store.conversation_store import ConversationStore
    return ConversationStore(max_size=10)


def _make_limiter():
    from caip_responses.ratelimit.limiter import RateLimiter
    return RateLimiter()


def _make_cost_tracker():
    from caip_responses.cost.tracker import CostTracker
    return CostTracker()


def _make_cache():
    from caip_responses.cache.response_cache import ResponseCache
    return ResponseCache(max_size=10, default_ttl=60, enabled=False)
