from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from caip_responses.client.async_client import AsyncClient
from caip_responses.models.errors import ProviderNotFoundError
from caip_responses.models.request import CreateResponseRequest
from caip_responses.models.response import Response
from caip_responses.models.streaming import StreamEvent
from caip_responses.providers.base import BaseProvider


class MockProvider(BaseProvider):
    def __init__(self, name: str = "mock") -> None:
        self._name = name
        self.last_request: CreateResponseRequest | None = None

    @property
    def provider_name(self) -> str:
        return self._name

    def supports_tool(self, tool_type: str) -> bool:
        return True

    def supports_reasoning(self) -> bool:
        return True

    async def create_response(self, request: CreateResponseRequest) -> Response:
        self.last_request = request
        return Response(
            id="resp_mock",
            model=request.model,
            output=[
                {
                    "type": "message",
                    "id": "item_1",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Mock response", "annotations": []}],
                    "status": "completed",
                }
            ],
        )

    async def create_response_stream(self, request: CreateResponseRequest) -> AsyncIterator[StreamEvent]:
        self.last_request = request
        yield StreamEvent(type="response.created", response={"id": "resp_mock", "model": request.model})
        yield StreamEvent(type="response.output_text.delta", delta="Mock ")
        yield StreamEvent(type="response.output_text.delta", delta="response")
        yield StreamEvent(type="response.completed")


class TestAsyncClient:
    @pytest.mark.asyncio
    async def test_create_non_streaming(self):
        mock_provider = MockProvider("openai")
        client = AsyncClient(providers={"openai": mock_provider})
        client._registry.add_prefix_mapping("gpt-", "openai")

        response = await client.responses.create(
            model="gpt-4.1",
            input="Hello",
            instructions="Be helpful",
        )

        assert isinstance(response, Response)
        assert response.id == "resp_mock"
        assert response.output_text == "Mock response"
        assert mock_provider.last_request.model == "gpt-4.1"
        assert mock_provider.last_request.input == "Hello"
        assert mock_provider.last_request.instructions == "Be helpful"

    @pytest.mark.asyncio
    async def test_create_streaming(self):
        mock_provider = MockProvider("openai")
        client = AsyncClient(providers={"openai": mock_provider})
        client._registry.add_prefix_mapping("gpt-", "openai")

        stream = await client.responses.create(
            model="gpt-4.1",
            input="Hello",
            stream=True,
        )

        events = []
        async for event in stream:
            events.append(event)

        assert len(events) == 4
        assert events[0].type == "response.created"
        assert events[1].type == "response.output_text.delta"
        assert events[1].delta == "Mock "
        assert events[2].delta == "response"
        assert events[3].type == "response.completed"

    @pytest.mark.asyncio
    async def test_provider_auto_routing_claude(self):
        mock_anthropic = MockProvider("anthropic")
        client = AsyncClient(providers={"anthropic": mock_anthropic})
        client._registry.add_prefix_mapping("claude-", "anthropic")

        await client.responses.create(
            model="claude-sonnet-4-20250514",
            input="Hello",
        )
        assert mock_anthropic.last_request.model == "claude-sonnet-4-20250514"

    @pytest.mark.asyncio
    async def test_explicit_provider_override(self):
        mock_openai = MockProvider("openai")
        mock_anthropic = MockProvider("anthropic")
        client = AsyncClient(providers={"openai": mock_openai, "anthropic": mock_anthropic})

        # Use a gpt model name but force anthropic provider
        await client.responses.create(
            model="gpt-4.1",
            input="Hello",
            provider="anthropic",
        )
        assert mock_anthropic.last_request is not None
        assert mock_openai.last_request is None

    @pytest.mark.asyncio
    async def test_no_provider_raises(self):
        client = AsyncClient()
        with pytest.raises(ProviderNotFoundError):
            await client.responses.create(model="unknown-model", input="Hello")

    @pytest.mark.asyncio
    async def test_all_params_passed_through(self):
        mock_provider = MockProvider("openai")
        client = AsyncClient(providers={"openai": mock_provider})
        client._registry.add_prefix_mapping("gpt-", "openai")

        await client.responses.create(
            model="gpt-4.1",
            input=[{"role": "user", "content": "Hi"}],
            instructions="Be helpful",
            tools=[{"type": "function", "name": "test", "parameters": {}}],
            tool_choice="required",
            parallel_tool_calls=True,
            temperature=0.5,
            top_p=0.9,
            max_output_tokens=512,
            reasoning={"effort": "high"},
            metadata={"session": "123"},
            store=True,
            user="user_1",
            previous_response_id="resp_prev",
        )

        req = mock_provider.last_request
        assert req.instructions == "Be helpful"
        assert req.tool_choice == "required"
        assert req.parallel_tool_calls is True
        assert req.temperature == 0.5
        assert req.max_output_tokens == 512
        assert req.reasoning.effort == "high"
        assert req.metadata == {"session": "123"}
        assert req.store is True
        assert req.user == "user_1"
        assert req.previous_response_id == "resp_prev"

    @pytest.mark.asyncio
    async def test_context_manager(self):
        mock_provider = MockProvider("openai")
        async with AsyncClient(providers={"openai": mock_provider}) as client:
            client._registry.add_prefix_mapping("gpt-", "openai")
            response = await client.responses.create(model="gpt-4.1", input="Hi")
            assert response.id == "resp_mock"

    @pytest.mark.asyncio
    async def test_prompt_param_passed_through(self):
        mock_provider = MockProvider("openai")
        client = AsyncClient(providers={"openai": mock_provider})
        client._registry.add_prefix_mapping("gpt-", "openai")

        await client.responses.create(
            model="gpt-5",
            prompt={
                "id": "pmpt_abc123",
                "version": "2",
                "variables": {"customer_name": "Jane"},
            },
        )

        req = mock_provider.last_request
        assert req.prompt is not None
        # Pydantic coerces the dict into a PromptConfig model
        assert req.prompt.id == "pmpt_abc123"
        assert req.prompt.variables["customer_name"] == "Jane"

    @pytest.mark.asyncio
    async def test_file_search_tools_passed_through(self):
        """User example: file_search with vector_store_ids."""
        mock_provider = MockProvider("openai")
        client = AsyncClient(providers={"openai": mock_provider})
        client._registry.add_prefix_mapping("gpt-", "openai")

        await client.responses.create(
            model="gpt-4.1",
            input="What is in my files?",
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": ["vs_abc123"],
                    "max_num_results": 5,
                },
            ],
        )

        req = mock_provider.last_request
        assert req.tools is not None
        assert len(req.tools) == 1
        assert req.tools[0]["type"] == "file_search"
        assert req.tools[0]["vector_store_ids"] == ["vs_abc123"]

    @pytest.mark.asyncio
    async def test_namespace_tool_search_passed_through(self):
        """User example: namespace + tool_search + defer_loading + gpt-5.4."""
        mock_provider = MockProvider("openai")
        client = AsyncClient(providers={"openai": mock_provider})
        client._registry.add_prefix_mapping("gpt-", "openai")

        await client.responses.create(
            model="gpt-5.4",
            input="What is 2 + 2?",
            tools=[
                {
                    "type": "namespace",
                    "name": "math",
                    "description": "A collection of math tools",
                    "tools": [
                        {
                            "type": "function",
                            "name": "add",
                            "description": "Add two numbers together",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "a": {"type": "number"},
                                    "b": {"type": "number"},
                                },
                            },
                            "defer_loading": True,
                        },
                        {
                            "type": "function",
                            "name": "multiply",
                            "description": "Multiply two numbers together",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "a": {"type": "number"},
                                    "b": {"type": "number"},
                                },
                            },
                            "defer_loading": True,
                        },
                    ],
                },
                {"type": "tool_search"},
            ],
        )

        req = mock_provider.last_request
        assert req.model == "gpt-5.4"
        assert req.tools is not None
        assert len(req.tools) == 2
        assert req.tools[0]["type"] == "namespace"
        assert req.tools[0]["name"] == "math"
        assert len(req.tools[0]["tools"]) == 2
        assert req.tools[0]["tools"][0]["defer_loading"] is True
        assert req.tools[1]["type"] == "tool_search"

    @pytest.mark.asyncio
    async def test_mcp_with_server_description_passed_through(self):
        """User example: MCP with server_description."""
        mock_provider = MockProvider("openai")
        client = AsyncClient(providers={"openai": mock_provider})
        client._registry.add_prefix_mapping("gpt-", "openai")

        await client.responses.create(
            model="gpt-4.1",
            input="Roll a d20 for my character",
            tools=[
                {
                    "type": "mcp",
                    "server_label": "dmcp",
                    "server_url": "https://dmcp-server.deno.dev/sse",
                    "server_description": "A D&D MCP server for dice rolling and character management.",
                    "require_approval": "never",
                },
            ],
        )

        req = mock_provider.last_request
        assert req.tools is not None
        assert len(req.tools) == 1
        assert req.tools[0]["type"] == "mcp"
        assert req.tools[0]["server_label"] == "dmcp"
        assert req.tools[0]["server_description"] == "A D&D MCP server for dice rolling and character management."

    @pytest.mark.asyncio
    async def test_mcp_discovery_prepass_for_non_openai(self):
        """For a provider that doesn't support MCP natively, the client
        discovers the server's tools and exposes each as a synthetic
        function the agent loop can call directly."""
        from unittest.mock import AsyncMock

        class GeminiLikeProvider(MockProvider):
            def supports_tool(self, tool_type: str) -> bool:
                return tool_type in {"function", "web_search"}

        mock_provider = GeminiLikeProvider("gemini")
        client = AsyncClient(providers={"gemini": mock_provider})
        client._registry.add_prefix_mapping("gemini-", "gemini")

        mcp_handler = client.builtin_tools.get("mcp")

        async def fake_discover(label: str, url: str):
            mcp_handler._server_tools[label] = [
                {
                    "type": "function",
                    "name": "_builtin_mcp_dmcp_roll_dice",
                    "description": "Roll dice",
                    "parameters": {"type": "object", "properties": {}},
                    "_mcp_server_label": label,
                    "_mcp_server_url": url,
                    "_mcp_tool_name": "roll_dice",
                }
            ]
            return mcp_handler._server_tools[label]

        mcp_handler.discover_tools = AsyncMock(side_effect=fake_discover)

        await client.responses.create(
            model="gemini-2.5-flash",
            input="Roll a d20",
            tools=[
                {
                    "type": "mcp",
                    "server_label": "dmcp",
                    "server_url": "https://dmcp-server.deno.dev/sse",
                },
            ],
        )

        mcp_handler.discover_tools.assert_awaited_once_with(
            "dmcp", "https://dmcp-server.deno.dev/sse"
        )
        req = mock_provider.last_request
        assert req.tools is not None
        names = [t.get("name") for t in req.tools]
        assert "_builtin_mcp_dmcp_roll_dice" in names

    @pytest.mark.asyncio
    async def test_custom_tool_passed_through(self):
        """Custom tool with grammar passed through to provider."""
        mock_provider = MockProvider("openai")
        client = AsyncClient(providers={"openai": mock_provider})
        client._registry.add_prefix_mapping("gpt-", "openai")

        await client.responses.create(
            model="gpt-4.1",
            input="Write a Python hello world",
            tools=[
                {
                    "type": "custom_tool",
                    "name": "code_exec",
                    "description": "Execute Python code",
                },
            ],
        )

        req = mock_provider.last_request
        assert req.tools[0]["type"] == "custom_tool"
        assert req.tools[0]["name"] == "code_exec"

    @pytest.mark.asyncio
    async def test_allowed_tools_tool_choice_passed_through(self):
        """allowed_tools tool_choice passes through to provider."""
        mock_provider = MockProvider("openai")
        client = AsyncClient(providers={"openai": mock_provider})
        client._registry.add_prefix_mapping("gpt-", "openai")

        await client.responses.create(
            model="gpt-4.1",
            input="What's the weather?",
            tools=[
                {"type": "function", "name": "get_weather", "parameters": {}},
                {"type": "function", "name": "send_email", "parameters": {}},
            ],
            tool_choice={
                "type": "allowed_tools",
                "mode": "auto",
                "tools": [
                    {"type": "function", "name": "get_weather"},
                ],
            },
        )

        req = mock_provider.last_request
        assert req.tool_choice["type"] == "allowed_tools"
        assert len(req.tool_choice["tools"]) == 1

    @pytest.mark.asyncio
    async def test_web_search_with_filters_and_location(self):
        """Web search tool with domain filtering and user location."""
        mock_provider = MockProvider("openai")
        client = AsyncClient(providers={"openai": mock_provider})
        client._registry.add_prefix_mapping("gpt-", "openai")

        await client.responses.create(
            model="gpt-4.1",
            input="What are the latest updates from OpenAI?",
            tools=[
                {
                    "type": "web_search",
                    "search_context_size": "high",
                    "user_location": {
                        "type": "approximate",
                        "country": "US",
                        "city": "San Francisco",
                        "region": "California",
                        "timezone": "America/Los_Angeles",
                    },
                    "filters": {
                        "allowed_domains": ["openai.com", "blog.openai.com"],
                    },
                },
            ],
        )

        req = mock_provider.last_request
        assert req.tools[0]["type"] == "web_search"
        assert req.tools[0]["search_context_size"] == "high"
        assert req.tools[0]["user_location"]["country"] == "US"
        assert req.tools[0]["filters"]["allowed_domains"] == ["openai.com", "blog.openai.com"]

    @pytest.mark.asyncio
    async def test_web_search_preview_tool(self):
        """web_search_preview tool variant."""
        mock_provider = MockProvider("openai")
        client = AsyncClient(providers={"openai": mock_provider})
        client._registry.add_prefix_mapping("gpt-", "openai")

        await client.responses.create(
            model="gpt-4.1",
            input="Search for something",
            tools=[{"type": "web_search_preview"}],
        )

        req = mock_provider.last_request
        assert req.tools[0]["type"] == "web_search_preview"

    @pytest.mark.asyncio
    async def test_mcp_connector_passed_through(self):
        """Connector (Dropbox) with connector_id instead of server_url."""
        mock_provider = MockProvider("openai")
        client = AsyncClient(providers={"openai": mock_provider})
        client._registry.add_prefix_mapping("gpt-", "openai")

        await client.responses.create(
            model="gpt-5",
            input="Summarize the Q2 earnings report.",
            tools=[
                {
                    "type": "mcp",
                    "server_label": "Dropbox",
                    "connector_id": "connector_dropbox",
                    "authorization": "<oauth access token>",
                    "require_approval": "never",
                },
            ],
        )

        req = mock_provider.last_request
        assert req.tools[0]["type"] == "mcp"
        assert req.tools[0]["connector_id"] == "connector_dropbox"
        assert "server_url" not in req.tools[0] or req.tools[0].get("server_url") is None

    @pytest.mark.asyncio
    async def test_mcp_with_allowed_tools_filter(self):
        """MCP with allowed_tools to filter exposed tools."""
        mock_provider = MockProvider("openai")
        client = AsyncClient(providers={"openai": mock_provider})
        client._registry.add_prefix_mapping("gpt-", "openai")

        await client.responses.create(
            model="gpt-5",
            input="Roll 2d4+1",
            tools=[
                {
                    "type": "mcp",
                    "server_label": "dmcp",
                    "server_url": "https://dmcp-server.deno.dev/sse",
                    "require_approval": "never",
                    "allowed_tools": ["roll"],
                },
            ],
        )

        req = mock_provider.last_request
        assert req.tools[0]["allowed_tools"] == ["roll"]

    @pytest.mark.asyncio
    async def test_mcp_require_approval_dict(self):
        """MCP with dict-style require_approval for per-tool approvals."""
        mock_provider = MockProvider("openai")
        client = AsyncClient(providers={"openai": mock_provider})
        client._registry.add_prefix_mapping("gpt-", "openai")

        await client.responses.create(
            model="gpt-5",
            input="What transport protocols does MCP support?",
            tools=[
                {
                    "type": "mcp",
                    "server_label": "deepwiki",
                    "server_url": "https://mcp.deepwiki.com/mcp",
                    "require_approval": {
                        "never": {
                            "tool_names": ["ask_question", "read_wiki_structure"],
                        },
                    },
                },
            ],
        )

        req = mock_provider.last_request
        assert req.tools[0]["require_approval"]["never"]["tool_names"] == [
            "ask_question",
            "read_wiki_structure",
        ]

    @pytest.mark.asyncio
    async def test_mcp_approval_response_as_input(self):
        """MCP approval response sent back as input item."""
        mock_provider = MockProvider("openai")
        client = AsyncClient(providers={"openai": mock_provider})
        client._registry.add_prefix_mapping("gpt-", "openai")

        await client.responses.create(
            model="gpt-5",
            input=[
                {
                    "type": "mcp_approval_response",
                    "approve": True,
                    "approval_request_id": "mcpr_682d498e3bd4819196a0ce1664f8e77b04ad1e533afccbfa",
                },
            ],
            tools=[
                {
                    "type": "mcp",
                    "server_label": "dmcp",
                    "server_url": "https://dmcp-server.deno.dev/sse",
                    "require_approval": "always",
                },
            ],
            previous_response_id="resp_682d498bdefc81918b4a6aa477bfafd904ad1e533afccbfa",
        )

        req = mock_provider.last_request
        assert req.input[0]["type"] == "mcp_approval_response"
        assert req.input[0]["approve"] is True
        assert req.previous_response_id == "resp_682d498bdefc81918b4a6aa477bfafd904ad1e533afccbfa"

    @pytest.mark.asyncio
    async def test_shell_with_hosted_skills(self):
        """Shell tool with hosted skill_reference skills (container_auto)."""
        mock_provider = MockProvider("openai")
        client = AsyncClient(providers={"openai": mock_provider})
        client._registry.add_prefix_mapping("gpt-", "openai")

        await client.responses.create(
            model="gpt-5.4",
            input="Use the skills to add 144 and 377.",
            tools=[
                {
                    "type": "shell",
                    "environment": {
                        "type": "container_auto",
                        "skills": [
                            {"type": "skill_reference", "skill_id": "skill_abc"},
                            {"type": "skill_reference", "skill_id": "skill_def", "version": 2},
                        ],
                    },
                },
            ],
        )

        req = mock_provider.last_request
        assert req.tools[0]["type"] == "shell"
        assert req.tools[0]["environment"]["type"] == "container_auto"
        assert len(req.tools[0]["environment"]["skills"]) == 2
        assert req.tools[0]["environment"]["skills"][1]["version"] == 2

    @pytest.mark.asyncio
    async def test_shell_with_local_skills(self):
        """Shell tool with local skill paths."""
        mock_provider = MockProvider("openai")
        client = AsyncClient(providers={"openai": mock_provider})
        client._registry.add_prefix_mapping("gpt-", "openai")

        await client.responses.create(
            model="gpt-5.4",
            input="Use the csv-insights skill to summarize reports.",
            tools=[
                {
                    "type": "shell",
                    "environment": {
                        "type": "local",
                        "skills": [
                            {
                                "name": "csv-insights",
                                "description": "Summarize CSV files and produce a markdown report.",
                                "path": "/path/to/skill",
                            },
                        ],
                    },
                },
            ],
        )

        req = mock_provider.last_request
        assert req.tools[0]["environment"]["type"] == "local"
        assert req.tools[0]["environment"]["skills"][0]["name"] == "csv-insights"

    @pytest.mark.asyncio
    async def test_tool_search_hosted_with_namespace(self):
        """Hosted tool search with namespace + deferred functions."""
        mock_provider = MockProvider("openai")
        client = AsyncClient(providers={"openai": mock_provider})
        client._registry.add_prefix_mapping("gpt-", "openai")

        await client.responses.create(
            model="gpt-5.4",
            input="What is 2 + 2?",
            tools=[
                {
                    "type": "namespace",
                    "name": "math",
                    "description": "A collection of math tools",
                    "tools": [
                        {
                            "type": "function",
                            "name": "add",
                            "description": "Add two numbers",
                            "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}},
                            "defer_loading": True,
                        },
                    ],
                },
                {"type": "tool_search"},
            ],
        )

        req = mock_provider.last_request
        assert req.tools[0]["type"] == "namespace"
        assert req.tools[0]["tools"][0]["defer_loading"] is True
        assert req.tools[1]["type"] == "tool_search"

    @pytest.mark.asyncio
    async def test_tool_search_client_mode(self):
        """Client-executed tool search with execution and schema."""
        mock_provider = MockProvider("openai")
        client = AsyncClient(providers={"openai": mock_provider})
        client._registry.add_prefix_mapping("gpt-", "openai")

        await client.responses.create(
            model="gpt-5.4",
            input="Find a tool to add numbers",
            tools=[
                {
                    "type": "tool_search",
                    "execution": "client",
                    "schema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            ],
        )

        req = mock_provider.last_request
        assert req.tools[0]["type"] == "tool_search"
        assert req.tools[0]["execution"] == "client"
        assert req.tools[0]["schema"]["properties"]["query"]["type"] == "string"

    @pytest.mark.asyncio
    async def test_tool_search_output_as_input(self):
        """Client-executed: send tool_search_output back as input."""
        mock_provider = MockProvider("openai")
        client = AsyncClient(providers={"openai": mock_provider})
        client._registry.add_prefix_mapping("gpt-", "openai")

        await client.responses.create(
            model="gpt-5.4",
            input=[
                {
                    "type": "tool_search_output",
                    "id": "tso_abc",
                    "call_id": "call_ts_123",
                    "tools": [
                        {
                            "type": "function",
                            "name": "add",
                            "description": "Add two numbers",
                            "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}},
                        },
                    ],
                },
            ],
            tools=[
                {"type": "tool_search", "execution": "client"},
            ],
            previous_response_id="resp_prev_123",
        )

        req = mock_provider.last_request
        assert req.input[0]["type"] == "tool_search_output"
        assert req.input[0]["call_id"] == "call_ts_123"
        assert len(req.input[0]["tools"]) == 1

    @pytest.mark.asyncio
    async def test_file_search_with_metadata_filters(self):
        """File search with metadata filtering and max_num_results."""
        mock_provider = MockProvider("openai")
        client = AsyncClient(providers={"openai": mock_provider})
        client._registry.add_prefix_mapping("gpt-", "openai")

        await client.responses.create(
            model="gpt-4.1",
            input="Find Q4 revenue data",
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": ["vs_abc123"],
                    "max_num_results": 10,
                    "filters": {
                        "type": "eq",
                        "key": "year",
                        "value": "2025",
                    },
                },
            ],
        )

        req = mock_provider.last_request
        assert req.tools is not None
        assert req.tools[0]["type"] == "file_search"
        assert req.tools[0]["max_num_results"] == 10
        assert req.tools[0]["filters"]["type"] == "eq"
        assert req.tools[0]["filters"]["key"] == "year"

    @pytest.mark.asyncio
    async def test_file_search_with_compound_filter(self):
        """File search with compound (and/or) metadata filter."""
        mock_provider = MockProvider("openai")
        client = AsyncClient(providers={"openai": mock_provider})
        client._registry.add_prefix_mapping("gpt-", "openai")

        await client.responses.create(
            model="gpt-4.1",
            input="Find recent financial reports",
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": ["vs_abc123"],
                    "filters": {
                        "type": "and",
                        "filters": [
                            {"type": "eq", "key": "category", "value": "financial"},
                            {"type": "eq", "key": "year", "value": "2025"},
                        ],
                    },
                },
            ],
        )

        req = mock_provider.last_request
        assert req.tools[0]["filters"]["type"] == "and"
        assert len(req.tools[0]["filters"]["filters"]) == 2

    @pytest.mark.asyncio
    async def test_file_search_include_results(self):
        """include parameter passes through for file_search_call.results."""
        mock_provider = MockProvider("openai")
        client = AsyncClient(providers={"openai": mock_provider})
        client._registry.add_prefix_mapping("gpt-", "openai")

        await client.responses.create(
            model="gpt-4.1",
            input="Search my files",
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": ["vs_abc123"],
                },
            ],
            include=["file_search_call.results"],
        )

        req = mock_provider.last_request
        assert req.include == ["file_search_call.results"]
        assert req.tools[0]["type"] == "file_search"

    @pytest.mark.asyncio
    async def test_image_generation_basic(self):
        """Basic image generation tool passes through."""
        mock_provider = MockProvider("openai")
        client = AsyncClient(providers={"openai": mock_provider})
        client._registry.add_prefix_mapping("gpt-", "openai")

        await client.responses.create(
            model="gpt-5.4",
            input="Generate an image of gray tabby cat hugging an otter",
            tools=[{"type": "image_generation"}],
        )

        req = mock_provider.last_request
        assert req.tools is not None
        assert len(req.tools) == 1
        assert req.tools[0]["type"] == "image_generation"

    @pytest.mark.asyncio
    async def test_image_generation_with_options(self):
        """Image generation with size, quality, format, background, action options."""
        mock_provider = MockProvider("openai")
        client = AsyncClient(providers={"openai": mock_provider})
        client._registry.add_prefix_mapping("gpt-", "openai")

        await client.responses.create(
            model="gpt-5.4",
            input="Draw a landscape",
            tools=[
                {
                    "type": "image_generation",
                    "size": "1024x1536",
                    "quality": "high",
                    "output_format": "png",
                    "background": "opaque",
                    "action": "generate",
                },
            ],
        )

        req = mock_provider.last_request
        tool = req.tools[0]
        assert tool["type"] == "image_generation"
        assert tool["size"] == "1024x1536"
        assert tool["quality"] == "high"
        assert tool["output_format"] == "png"
        assert tool["background"] == "opaque"
        assert tool["action"] == "generate"

    @pytest.mark.asyncio
    async def test_image_generation_with_partial_images(self):
        """partial_images streaming option passes through."""
        mock_provider = MockProvider("openai")
        client = AsyncClient(providers={"openai": mock_provider})
        client._registry.add_prefix_mapping("gpt-", "openai")

        await client.responses.create(
            model="gpt-5.4",
            input="Draw a river of owl feathers",
            tools=[
                {
                    "type": "image_generation",
                    "partial_images": 2,
                },
            ],
        )

        req = mock_provider.last_request
        assert req.tools[0]["partial_images"] == 2

    @pytest.mark.asyncio
    async def test_image_generation_force_tool_choice(self):
        """tool_choice can force image_generation."""
        mock_provider = MockProvider("openai")
        client = AsyncClient(providers={"openai": mock_provider})
        client._registry.add_prefix_mapping("gpt-", "openai")

        await client.responses.create(
            model="gpt-5.4",
            input="A cat",
            tools=[{"type": "image_generation"}],
            tool_choice={"type": "image_generation"},
        )

        req = mock_provider.last_request
        assert req.tool_choice == {"type": "image_generation"}

    @pytest.mark.asyncio
    async def test_image_generation_multi_turn_previous_response(self):
        """Multi-turn editing via previous_response_id."""
        mock_provider = MockProvider("openai")
        client = AsyncClient(providers={"openai": mock_provider})
        client._registry.add_prefix_mapping("gpt-", "openai")

        await client.responses.create(
            model="gpt-5.4",
            previous_response_id="resp_img_1",
            input="Now make it look realistic",
            tools=[{"type": "image_generation"}],
        )

        req = mock_provider.last_request
        assert req.previous_response_id == "resp_img_1"
        assert req.tools[0]["type"] == "image_generation"

    @pytest.mark.asyncio
    async def test_image_generation_multi_turn_image_id(self):
        """Multi-turn editing by referencing image_generation_call id in input."""
        mock_provider = MockProvider("openai")
        client = AsyncClient(providers={"openai": mock_provider})
        client._registry.add_prefix_mapping("gpt-", "openai")

        await client.responses.create(
            model="gpt-5.4",
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Make it realistic"}],
                },
                {
                    "type": "image_generation_call",
                    "id": "ig_abc123",
                },
            ],
            tools=[{"type": "image_generation"}],
        )

        req = mock_provider.last_request
        assert req.input[0]["role"] == "user"
        assert req.input[1]["type"] == "image_generation_call"
        assert req.input[1]["id"] == "ig_abc123"

    @pytest.mark.asyncio
    async def test_computer_tool_ga(self):
        """GA computer tool passes through."""
        mock_provider = MockProvider("openai")
        client = AsyncClient(providers={"openai": mock_provider})
        client._registry.add_prefix_mapping("gpt-", "openai")

        await client.responses.create(
            model="gpt-5.4",
            input="Go to example.com and click the login button",
            tools=[
                {
                    "type": "computer",
                    "display_width": 1440,
                    "display_height": 900,
                    "environment": "browser",
                },
            ],
        )

        req = mock_provider.last_request
        assert req.tools[0]["type"] == "computer"
        assert req.tools[0]["display_width"] == 1440
        assert req.tools[0]["display_height"] == 900
        assert req.tools[0]["environment"] == "browser"

    @pytest.mark.asyncio
    async def test_computer_tool_preview(self):
        """Preview computer_use tool still works (backward compat)."""
        mock_provider = MockProvider("openai")
        client = AsyncClient(providers={"openai": mock_provider})
        client._registry.add_prefix_mapping("computer-", "openai")

        await client.responses.create(
            model="computer-use-preview",
            input="Click the button",
            tools=[
                {
                    "type": "computer_use",
                    "display_width": 1024,
                    "display_height": 768,
                },
            ],
            truncation="auto",
        )

        req = mock_provider.last_request
        assert req.tools[0]["type"] == "computer_use"
        assert req.truncation == "auto"

    @pytest.mark.asyncio
    async def test_computer_call_output_as_input(self):
        """Send computer_call_output screenshot back as input."""
        mock_provider = MockProvider("openai")
        client = AsyncClient(providers={"openai": mock_provider})
        client._registry.add_prefix_mapping("gpt-", "openai")

        await client.responses.create(
            model="gpt-5.4",
            input=[
                {
                    "type": "computer_call_output",
                    "call_id": "call_cua_1",
                    "output": {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,iVBORw0KGgo...",
                    },
                },
            ],
            tools=[
                {"type": "computer", "display_width": 1024, "display_height": 768},
            ],
            previous_response_id="resp_cua_1",
        )

        req = mock_provider.last_request
        assert req.input[0]["type"] == "computer_call_output"
        assert req.input[0]["call_id"] == "call_cua_1"
        assert req.input[0]["output"]["type"] == "input_image"
        assert req.previous_response_id == "resp_cua_1"

    @pytest.mark.asyncio
    async def test_computer_call_output_with_safety_checks(self):
        """Acknowledged safety checks pass through in computer_call_output."""
        mock_provider = MockProvider("openai")
        client = AsyncClient(providers={"openai": mock_provider})
        client._registry.add_prefix_mapping("gpt-", "openai")

        await client.responses.create(
            model="gpt-5.4",
            input=[
                {
                    "type": "computer_call_output",
                    "call_id": "call_safe_1",
                    "output": {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,abc...",
                    },
                    "acknowledged_safety_checks": [
                        {"id": "sc_1", "code": "sensitive_action", "message": "Form submission"},
                    ],
                },
            ],
            tools=[
                {"type": "computer", "display_width": 1024, "display_height": 768},
            ],
            previous_response_id="resp_safe_1",
        )

        req = mock_provider.last_request
        assert req.input[0]["acknowledged_safety_checks"][0]["id"] == "sc_1"

    @pytest.mark.asyncio
    async def test_shell_container_reference(self):
        """Shell with container_reference for reusable containers."""
        mock_provider = MockProvider("openai")
        client = AsyncClient(providers={"openai": mock_provider})
        client._registry.add_prefix_mapping("gpt-", "openai")

        await client.responses.create(
            model="gpt-5.4",
            input="List files in the container and show disk usage.",
            tools=[
                {
                    "type": "shell",
                    "environment": {
                        "type": "container_reference",
                        "container_id": "cntr_08f3d96c87a585390069118b594f7481a088b16cda7d9415fe",
                    },
                },
            ],
        )

        req = mock_provider.last_request
        assert req.tools[0]["environment"]["type"] == "container_reference"
        assert req.tools[0]["environment"]["container_id"].startswith("cntr_")

    @pytest.mark.asyncio
    async def test_shell_network_policy(self):
        """Shell with network allowlist."""
        mock_provider = MockProvider("openai")
        client = AsyncClient(providers={"openai": mock_provider})
        client._registry.add_prefix_mapping("gpt-", "openai")

        await client.responses.create(
            model="gpt-5.4",
            input="pip install httpx and fetch data",
            tool_choice="required",
            tools=[
                {
                    "type": "shell",
                    "environment": {
                        "type": "container_auto",
                        "network_policy": {
                            "type": "allowlist",
                            "allowed_domains": ["pypi.org", "files.pythonhosted.org"],
                        },
                    },
                },
            ],
        )

        req = mock_provider.last_request
        policy = req.tools[0]["environment"]["network_policy"]
        assert policy["type"] == "allowlist"
        assert "pypi.org" in policy["allowed_domains"]
        assert req.tool_choice == "required"

    @pytest.mark.asyncio
    async def test_shell_domain_secrets(self):
        """Shell with domain_secrets for private auth headers."""
        mock_provider = MockProvider("openai")
        client = AsyncClient(providers={"openai": mock_provider})
        client._registry.add_prefix_mapping("gpt-", "openai")

        await client.responses.create(
            model="gpt-5.4",
            input="Curl httpbin with auth",
            tools=[
                {
                    "type": "shell",
                    "environment": {
                        "type": "container_auto",
                        "network_policy": {
                            "type": "allowlist",
                            "allowed_domains": ["httpbin.org"],
                            "domain_secrets": [
                                {
                                    "domain": "httpbin.org",
                                    "name": "API_KEY",
                                    "value": "debug-secret-123",
                                },
                            ],
                        },
                    },
                },
            ],
        )

        req = mock_provider.last_request
        secrets = req.tools[0]["environment"]["network_policy"]["domain_secrets"]
        assert secrets[0]["name"] == "API_KEY"

    @pytest.mark.asyncio
    async def test_shell_call_output_as_input_local(self):
        """Local shell: send shell_call_output back as input."""
        mock_provider = MockProvider("openai")
        client = AsyncClient(providers={"openai": mock_provider})
        client._registry.add_prefix_mapping("gpt-", "openai")

        await client.responses.create(
            model="gpt-5.4",
            input=[
                {
                    "type": "shell_call_output",
                    "call_id": "call_local_1",
                    "output": "README.md\nsetup.py\nsrc/\n",
                },
            ],
            tools=[{"type": "shell", "environment": {"type": "local"}}],
            previous_response_id="resp_shell_local_1",
        )

        req = mock_provider.last_request
        assert req.input[0]["type"] == "shell_call_output"
        assert req.input[0]["call_id"] == "call_local_1"
        assert req.previous_response_id == "resp_shell_local_1"

    @pytest.mark.asyncio
    async def test_shell_multi_turn_continuation(self):
        """Continue work in same container with previous_response_id."""
        mock_provider = MockProvider("openai")
        client = AsyncClient(providers={"openai": mock_provider})
        client._registry.add_prefix_mapping("gpt-", "openai")

        await client.responses.create(
            model="gpt-5.4",
            previous_response_id="resp_prev_shell",
            input="Read /mnt/data/top5.csv and report the top candidate.",
            tools=[
                {
                    "type": "shell",
                    "environment": {
                        "type": "container_reference",
                        "container_id": "cntr_f19c2b51e4a06793d82d54a7be0fc9154d3361ab28ce7f6041",
                    },
                },
            ],
        )

        req = mock_provider.last_request
        assert req.previous_response_id == "resp_prev_shell"
        assert req.tools[0]["environment"]["type"] == "container_reference"
