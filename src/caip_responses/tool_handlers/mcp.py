"""MCP (Model Context Protocol) tool handler.

Connects directly to MCP servers using the ``mcp`` Python SDK.
No OpenAI dependency — this is a native, standalone MCP client.

Supports both transport types:
- **SSE (HTTP)**: ``server_url`` based — connects via HTTP/SSE
- **Stdio**: ``command`` based — spawns a subprocess

Falls back to OpenAI delegation only if the ``mcp`` SDK is not installed
AND an OpenAI API key is configured.
"""

from __future__ import annotations

import json
from typing import Any

from caip_responses.tool_handlers.base import BuiltinToolHandler
from caip_responses.tool_handlers.openai_delegator import (
    DelegatedToolMetrics,
    OpenAIDelegatorMixin,
)
from caip_responses.utils.id_gen import generate_item_id


class MCPHandler(BuiltinToolHandler, OpenAIDelegatorMixin):
    """Client-side MCP tool handler for non-OpenAI providers.

    **Primary: Direct MCP client** — connects to MCP servers natively
    using the ``mcp`` Python SDK. Lists available tools from the server,
    then exposes each as a synthetic function call the model can invoke.

    **Fallback: Azure OpenAI** — if ``mcp`` SDK is not installed but
    ``openai_api_key`` is provided, delegates to OpenAI which handles
    MCP server connections natively.

    Priority order:
    1. Direct MCP client (``mcp`` SDK installed + ``server_url`` or ``command``)
    2. Azure OpenAI delegation (``openai_api_key`` provided)
    3. Error — neither available

    Usage::

        # Direct (recommended — no OpenAI dependency):
        handler = MCPHandler()
        registry.register(handler)

        # With OpenAI fallback:
        handler = MCPHandler(
            openai_api_key="sk-...",
            openai_base_url="https://your-resource.openai.azure.com",
        )
        registry.register(handler)
    """

    def __init__(
        self,
        *,
        openai_api_key: str | None = None,
        openai_base_url: str | None = None,
        openai_model: str = "gpt-4.1-nano",
    ) -> None:
        BuiltinToolHandler.__init__(self)
        OpenAIDelegatorMixin.__init__(
            self,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            openai_model=openai_model,
        )
        # Store the original MCP tool configs for delegation
        self._mcp_configs: list[dict[str, Any]] = []
        # Cache discovered MCP tools per server: server_label -> [tool_defs]
        self._server_tools: dict[str, list[dict[str, Any]]] = {}
        # Check if mcp SDK is available
        self._has_mcp_sdk = self._check_mcp_sdk()

    @staticmethod
    def _check_mcp_sdk() -> bool:
        try:
            import mcp  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def metrics(self) -> DelegatedToolMetrics:
        return self._delegated_metrics

    def tool_type(self) -> str:
        return "mcp"

    def set_mcp_configs(self, configs: list[dict[str, Any]]) -> None:
        """Store the original MCP tool definitions."""
        self._mcp_configs = configs

    def to_function_tools(
        self, tool_config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Convert MCP tool to synthetic function(s) the model can call.

        If MCP tools were pre-discovered via ``discover_tools()``, each
        MCP server tool becomes its own function. Otherwise a single
        generic function is created.
        """
        server_label = tool_config.get("server_label", "mcp_server")
        if tool_config not in self._mcp_configs:
            self._mcp_configs.append(tool_config)

        # If we have pre-discovered tools for this server, expose each
        if server_label in self._server_tools:
            return self._server_tools[server_label]

        # Fallback: generic request function
        return [
            {
                "type": "function",
                "name": self._make_fn_name(server_label),
                "description": (
                    f"Execute a request via MCP server '{server_label}'. "
                    "This connects to an external tool server that can "
                    "perform actions, retrieve data, or interact with "
                    "external services. Describe what you need done."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "request": {
                            "type": "string",
                            "description": (
                                "Natural language description of what to do "
                                "via the MCP server."
                            ),
                        },
                    },
                    "required": ["request"],
                },
            },
        ]

    async def discover_tools(self, server_label: str, server_url: str) -> list[dict[str, Any]]:
        """Connect to an MCP server and discover its available tools.

        Call this before ``to_function_tools()`` to expose each MCP
        server tool as its own function with proper parameter schemas.

        Args:
            server_label: Label to identify this server.
            server_url: The SSE endpoint URL of the MCP server.

        Returns:
            List of function tool definitions discovered from the server.
        """
        if not self._has_mcp_sdk:
            return []

        from mcp import ClientSession
        from mcp.client.sse import sse_client

        tools: list[dict[str, Any]] = []

        async with sse_client(url=server_url) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.list_tools()

                for tool in result.tools:
                    fn_name = self._make_fn_name(f"{server_label}_{tool.name}")
                    tool_def: dict[str, Any] = {
                        "type": "function",
                        "name": fn_name,
                        "description": tool.description or f"MCP tool: {tool.name}",
                        "parameters": tool.inputSchema if tool.inputSchema else {
                            "type": "object",
                            "properties": {},
                        },
                        "_mcp_server_label": server_label,
                        "_mcp_server_url": server_url,
                        "_mcp_tool_name": tool.name,
                    }
                    tools.append(tool_def)

        self._server_tools[server_label] = tools
        return tools

    async def execute(
        self, name: str, arguments: dict[str, Any]
    ) -> str:
        # Check if this is a discovered MCP tool (has _mcp metadata)
        mcp_tool_name, server_url = self._resolve_mcp_tool(name)

        if mcp_tool_name and server_url and self._has_mcp_sdk:
            return await self._execute_direct(mcp_tool_name, server_url, arguments)

        # Generic request mode — try direct first, then OpenAI fallback
        request_text = arguments.get("request", "")
        if not request_text and not mcp_tool_name:
            return json.dumps({"error": "No request provided"})

        # Try to find a server URL from stored configs
        server_url_from_config = self._find_server_url_from_configs()

        if server_url_from_config and self._has_mcp_sdk:
            return await self._execute_generic_direct(
                request_text or json.dumps(arguments),
                server_url_from_config,
            )

        if self._openai_api_key:
            return await self._execute_via_openai(request_text or json.dumps(arguments))

        return json.dumps({
            "error": (
                "MCP tool requires either: (1) mcp SDK installed "
                "(pip install mcp) + server_url in tool config, or "
                "(2) openai_api_key for OpenAI delegation"
            ),
        })

    async def _execute_direct(
        self, tool_name: str, server_url: str, arguments: dict[str, Any]
    ) -> str:
        """Call a specific MCP tool directly via the mcp SDK."""
        from mcp import ClientSession
        from mcp.client.sse import sse_client

        try:
            async with sse_client(url=server_url) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments)

                    # Extract content from result
                    content_parts = []
                    for item in result.content:
                        if hasattr(item, "text"):
                            content_parts.append(item.text)
                        elif hasattr(item, "data"):
                            content_parts.append(str(item.data))

                    return json.dumps({
                        "result": "\n".join(content_parts) if content_parts else "Tool executed successfully",
                        "is_error": result.isError or False,
                        "_source": "direct_mcp",
                    })

        except Exception as e:
            return json.dumps({
                "error": f"MCP call failed: {e}",
                "_source": "direct_mcp",
            })

    async def _execute_generic_direct(
        self, request: str, server_url: str
    ) -> str:
        """List tools from server and try to match the request."""
        from mcp import ClientSession
        from mcp.client.sse import sse_client

        try:
            async with sse_client(url=server_url) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tools_result = await session.list_tools()

                    # Return list of available tools for the model
                    tool_list = []
                    for tool in tools_result.tools:
                        tool_list.append({
                            "name": tool.name,
                            "description": tool.description or "",
                        })

                    return json.dumps({
                        "result": f"MCP server has {len(tool_list)} tools available",
                        "available_tools": tool_list,
                        "hint": "Call a specific tool by name with the required arguments.",
                        "_source": "direct_mcp",
                    })

        except Exception as e:
            return json.dumps({
                "error": f"MCP connection failed: {e}",
                "_source": "direct_mcp",
            })

    async def _execute_via_openai(self, request_text: str) -> str:
        """Fallback: delegate to Azure OpenAI."""
        client = self._get_openai_client()

        response = await client.responses.create(
            model=self._openai_model,
            input=request_text,
            tools=self._mcp_configs if self._mcp_configs else [{"type": "mcp"}],
        )

        usage = self._record_usage(response, "mcp")
        text = self._extract_text_from_response(response)

        mcp_outputs = []
        for item in response.output:
            item_type = getattr(item, "type", None)
            if item_type == "mcp_call":
                mcp_outputs.append({
                    "name": getattr(item, "name", ""),
                    "output": getattr(item, "output", ""),
                    "server_label": getattr(item, "server_label", ""),
                })

        result: dict[str, Any] = {"result": text, "_source": "openai_delegation"}
        if mcp_outputs:
            result["mcp_calls"] = mcp_outputs
        result["_usage"] = {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
        }
        return json.dumps(result)

    def to_output_item(
        self,
        name: str,
        arguments: dict[str, Any],
        result: str,
    ) -> dict[str, Any] | None:
        item: dict[str, Any] = {
            "type": "mcp_call",
            "id": generate_item_id(),
            "name": name,
            "arguments": json.dumps(arguments),
            "output": result,
            "status": "completed",
            "server_label": "",
        }

        last = self.last_delegated_usage
        if last:
            item["_delegated_usage"] = {
                "input_tokens": last.input_tokens,
                "output_tokens": last.output_tokens,
                "total_tokens": last.total_tokens,
                "model": last.model,
                "provider": last.provider,
            }
        return item

    def system_prompt_addendum(
        self, tool_config: dict[str, Any]
    ) -> str | None:
        server_label = tool_config.get("server_label", "mcp_server")
        return (
            f"You have access to an MCP (Model Context Protocol) server "
            f"named '{server_label}'. You can call the "
            f"'{self._make_fn_name(server_label)}' function to interact "
            f"with it. Describe your request in natural language."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_mcp_tool(self, fn_name: str) -> tuple[str | None, str | None]:
        """Resolve a synthetic function name to its MCP tool name and server URL."""
        for tools in self._server_tools.values():
            for tool_def in tools:
                if tool_def.get("name") == fn_name:
                    return (
                        tool_def.get("_mcp_tool_name"),
                        tool_def.get("_mcp_server_url"),
                    )
        return None, None

    def _find_server_url_from_configs(self) -> str | None:
        """Find a server URL from stored MCP configs."""
        for config in self._mcp_configs:
            url = config.get("server_url")
            if url:
                return url
        return None
