from __future__ import annotations

from caip_responses.models.tools import (
    CodeInterpreterTool,
    ComputerTool,
    ComputerUseTool,
    CustomTool,
    CustomToolGrammar,
    FileSearchTool,
    FunctionTool,
    ImageGenerationTool,
    MCPTool,
    NamespaceTool,
    ShellTool,
    ToolSearchTool,
    WebSearchPreviewTool,
    WebSearchTool,
)


class TestFunctionTool:
    def test_basic(self):
        tool = FunctionTool(
            name="get_weather",
            description="Get weather",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        )
        assert tool.type == "function"
        assert tool.name == "get_weather"

    def test_with_strict(self):
        tool = FunctionTool(name="test", strict=True)
        assert tool.strict is True

    def test_defaults(self):
        tool = FunctionTool(name="test")
        assert tool.description == ""
        assert tool.parameters == {}
        assert tool.strict is None


class TestWebSearchTool:
    def test_basic(self):
        tool = WebSearchTool()
        assert tool.type == "web_search"
        assert tool.search_context_size == "medium"

    def test_with_filters(self):
        tool = WebSearchTool(
            search_context_size="high",
            user_location={"type": "approximate", "country": "IN"},
            filters={"allowed_domains": ["example.com"]},
        )
        assert tool.search_context_size == "high"
        assert tool.filters["allowed_domains"] == ["example.com"]

    def test_with_external_web_access(self):
        tool = WebSearchTool(external_web_access=True)
        assert tool.external_web_access is True

    def test_external_web_access_false(self):
        """Offline/cache-only mode per docs."""
        tool = WebSearchTool(external_web_access=False)
        assert tool.external_web_access is False

    def test_domain_filtering_allowed(self):
        """Domain filtering with allowed_domains — up to 100 domains."""
        tool = WebSearchTool(
            filters={"allowed_domains": ["openai.com", "github.com"]},
        )
        assert tool.filters["allowed_domains"] == ["openai.com", "github.com"]

    def test_domain_filtering_blocked(self):
        """Domain filtering with blocked_domains."""
        tool = WebSearchTool(
            filters={"blocked_domains": ["spam.com"]},
        )
        assert tool.filters["blocked_domains"] == ["spam.com"]

    def test_user_location_full(self):
        """User location with all fields per docs."""
        tool = WebSearchTool(
            user_location={
                "type": "approximate",
                "country": "US",
                "city": "Minneapolis",
                "region": "Minnesota",
                "timezone": "America/Chicago",
            },
        )
        loc = tool.user_location
        assert loc["country"] == "US"
        assert loc["city"] == "Minneapolis"
        assert loc["region"] == "Minnesota"
        assert loc["timezone"] == "America/Chicago"


class TestWebSearchPreviewTool:
    def test_basic(self):
        tool = WebSearchPreviewTool()
        assert tool.type == "web_search_preview"
        assert tool.search_context_size == "medium"

    def test_with_filters(self):
        tool = WebSearchPreviewTool(
            search_context_size="high",
            filters={"allowed_domains": ["docs.python.org"]},
        )
        assert tool.search_context_size == "high"
        assert tool.filters["allowed_domains"] == ["docs.python.org"]

    def test_no_external_web_access_field(self):
        """Preview tool ignores external_web_access — always live."""
        tool = WebSearchPreviewTool(user_location={"type": "approximate", "country": "GB"})
        assert not hasattr(tool, "external_web_access") or tool.external_web_access is None


class TestMCPTool:
    def test_basic(self):
        tool = MCPTool(
            server_label="RAG-Server",
            server_url="http://localhost:8080",
        )
        assert tool.type == "mcp"
        assert tool.server_label == "RAG-Server"
        assert tool.require_approval == "never"

    def test_with_auth(self):
        tool = MCPTool(
            server_label="test",
            server_url="http://localhost",
            authorization="Bearer token123",
        )
        assert tool.authorization == "Bearer token123"

    def test_remote_mcp_server(self):
        """Exact pattern from docs: remote MCP server."""
        tool = MCPTool(
            server_label="dmcp",
            server_description="A Dungeons and Dragons MCP server to assist with dice rolling.",
            server_url="https://dmcp-server.deno.dev/sse",
            require_approval="never",
        )
        assert tool.server_url == "https://dmcp-server.deno.dev/sse"
        assert tool.server_description is not None
        assert tool.connector_id is None

    def test_connector_dropbox(self):
        """Exact pattern from docs: Dropbox connector."""
        tool = MCPTool(
            server_label="Dropbox",
            connector_id="connector_dropbox",
            authorization="<oauth access token>",
            require_approval="never",
        )
        assert tool.connector_id == "connector_dropbox"
        assert tool.server_url is None

    def test_connector_google_calendar(self):
        """Exact pattern from docs: Google Calendar connector."""
        tool = MCPTool(
            server_label="google_calendar",
            connector_id="connector_googlecalendar",
            authorization="ya29.A0AS3H6...",
            require_approval="never",
        )
        assert tool.connector_id == "connector_googlecalendar"

    def test_allowed_tools_filter(self):
        """Filter which tools are exposed from an MCP server."""
        tool = MCPTool(
            server_label="dmcp",
            server_url="https://dmcp-server.deno.dev/sse",
            require_approval="never",
            allowed_tools=["roll"],
        )
        assert tool.allowed_tools == ["roll"]

    def test_require_approval_dict(self):
        """require_approval as dict — skip approvals for specific tools."""
        tool = MCPTool(
            server_label="deepwiki",
            server_url="https://mcp.deepwiki.com/mcp",
            require_approval={
                "never": {
                    "tool_names": ["ask_question", "read_wiki_structure"],
                },
            },
        )
        assert tool.require_approval["never"]["tool_names"] == ["ask_question", "read_wiki_structure"]

    def test_require_approval_always(self):
        tool = MCPTool(
            server_label="dmcp",
            server_url="https://dmcp-server.deno.dev/sse",
            require_approval="always",
        )
        assert tool.require_approval == "always"

    def test_defer_loading(self):
        """Defer loading MCP server tools for use with tool_search."""
        tool = MCPTool(
            server_label="large-server",
            server_url="https://example.com/mcp",
            defer_loading=True,
        )
        assert tool.defer_loading is True


class TestFileSearchTool:
    def test_basic(self):
        tool = FileSearchTool(vector_store_ids=["vs_123"])
        assert tool.type == "file_search"
        assert tool.vector_store_ids == ["vs_123"]

    def test_with_max_results(self):
        tool = FileSearchTool(
            vector_store_ids=["vs_abc123"],
            max_num_results=5,
        )
        assert tool.max_num_results == 5

    def test_with_metadata_filters(self):
        """Metadata filtering on file search results."""
        tool = FileSearchTool(
            vector_store_ids=["vs_abc123"],
            filters={
                "type": "eq",
                "key": "category",
                "value": "finance",
            },
        )
        assert tool.filters["key"] == "category"
        assert tool.filters["value"] == "finance"

    def test_with_compound_filter(self):
        """Compound metadata filter with AND/OR."""
        tool = FileSearchTool(
            vector_store_ids=["vs_abc123"],
            filters={
                "type": "and",
                "filters": [
                    {"type": "eq", "key": "department", "value": "engineering"},
                    {"type": "eq", "key": "year", "value": "2025"},
                ],
            },
        )
        assert tool.filters["type"] == "and"
        assert len(tool.filters["filters"]) == 2

    def test_multiple_vector_stores(self):
        tool = FileSearchTool(vector_store_ids=["vs_123", "vs_456"])
        assert len(tool.vector_store_ids) == 2

    def test_extra_fields_allowed(self):
        """Forward-compat: extra fields pass through."""
        tool = FileSearchTool(vector_store_ids=["vs_123"], ranking_options={"ranker": "auto"})
        assert tool.ranking_options == {"ranker": "auto"}


class TestCodeInterpreterTool:
    def test_basic(self):
        tool = CodeInterpreterTool()
        assert tool.type == "code_interpreter"


class TestComputerTool:
    """GA computer tool (type='computer')."""

    def test_basic(self):
        tool = ComputerTool()
        assert tool.type == "computer"
        assert tool.display_width == 1024
        assert tool.display_height == 768
        assert tool.environment is None

    def test_custom_resolution(self):
        tool = ComputerTool(display_width=1440, display_height=900)
        assert tool.display_width == 1440
        assert tool.display_height == 900

    def test_with_environment(self):
        tool = ComputerTool(environment="browser")
        assert tool.environment == "browser"

    def test_extra_fields_allowed(self):
        """Forward-compat: extra fields pass through."""
        tool = ComputerTool(display_width=1024, display_height=768, sandbox="isolated")
        assert tool.sandbox == "isolated"

    def test_model_dump_roundtrip(self):
        data = {"type": "computer", "display_width": 1600, "display_height": 900}
        tool = ComputerTool(**data)
        dumped = tool.model_dump()
        assert dumped["type"] == "computer"
        assert dumped["display_width"] == 1600


class TestComputerUseTool:
    """Preview/deprecated tool (type='computer_use')."""

    def test_basic(self):
        tool = ComputerUseTool()
        assert tool.type == "computer_use"
        assert tool.display_width == 1024
        assert tool.display_height == 768

    def test_extra_fields_allowed(self):
        tool = ComputerUseTool(truncation="auto")
        data = tool.model_dump()
        assert data["truncation"] == "auto"


class TestMCPToolWithDescription:
    def test_server_description(self):
        tool = MCPTool(
            server_label="dmcp",
            server_url="https://dmcp-server.deno.dev/sse",
            server_description="A D&D MCP server for dice rolling.",
            require_approval="never",
        )
        assert tool.server_description == "A D&D MCP server for dice rolling."


class TestToolSearchTool:
    def test_basic(self):
        tool = ToolSearchTool()
        assert tool.type == "tool_search"
        assert tool.execution is None

    def test_hosted_mode(self):
        """Hosted tool search (default) — execution=server."""
        tool = ToolSearchTool(execution="server")
        assert tool.execution == "server"

    def test_client_mode(self):
        """Client-executed tool search with schema for search arguments.

        Note: 'schema' is stored as extra field (clashes with BaseModel.schema method).
        Access via model_extra or model_dump(); it serializes correctly to dict/JSON.
        """
        tool = ToolSearchTool(
            execution="client",
            schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        )
        assert tool.execution == "client"
        dumped = tool.model_dump()
        assert dumped["schema"]["properties"]["query"]["type"] == "string"

    def test_client_mode_serializes_schema(self):
        """Verify schema round-trips correctly through model_dump."""
        tool = ToolSearchTool(
            execution="client",
            schema={"type": "object", "properties": {"q": {"type": "string"}}},
        )
        dumped = tool.model_dump()
        assert "schema" in dumped
        assert dumped["schema"]["properties"]["q"]["type"] == "string"


class TestShellTool:
    def test_basic(self):
        tool = ShellTool()
        assert tool.type == "shell"
        assert tool.environment is None

    def test_hosted_shell_with_skill_references(self):
        """Exact pattern from docs: hosted shell with skill_reference skills."""
        tool = ShellTool(
            environment={
                "type": "container_auto",
                "skills": [
                    {"type": "skill_reference", "skill_id": "<skill_id>"},
                    {"type": "skill_reference", "skill_id": "<skill_id>", "version": 2},
                ],
            },
        )
        assert tool.environment["type"] == "container_auto"
        assert len(tool.environment["skills"]) == 2
        assert tool.environment["skills"][0]["type"] == "skill_reference"
        assert tool.environment["skills"][1]["version"] == 2

    def test_local_shell_with_skills(self):
        """Exact pattern from docs: local shell with inline skill paths."""
        tool = ShellTool(
            environment={
                "type": "local",
                "skills": [
                    {
                        "name": "csv-insights",
                        "description": "Summarize CSV files and produce a markdown report.",
                        "path": "<path-to-skill-folder>",
                    },
                ],
            },
        )
        assert tool.environment["type"] == "local"
        assert tool.environment["skills"][0]["name"] == "csv-insights"
        assert tool.environment["skills"][0]["path"] == "<path-to-skill-folder>"

    def test_hosted_shell_no_skills(self):
        """Hosted shell without skills."""
        tool = ShellTool(
            environment={"type": "container_auto"},
        )
        assert tool.environment["type"] == "container_auto"
        assert "skills" not in tool.environment

    def test_container_reference(self):
        """Reuse a container across requests via container_reference."""
        tool = ShellTool(
            environment={
                "type": "container_reference",
                "container_id": "cntr_08f3d96c87a585390069118b594f7481a088b16cda7d9415fe",
            },
        )
        assert tool.environment["type"] == "container_reference"
        assert tool.environment["container_id"].startswith("cntr_")

    def test_network_policy_allowlist(self):
        """Network access with domain allowlist."""
        tool = ShellTool(
            environment={
                "type": "container_auto",
                "network_policy": {
                    "type": "allowlist",
                    "allowed_domains": ["pypi.org", "files.pythonhosted.org", "github.com"],
                },
            },
        )
        policy = tool.environment["network_policy"]
        assert policy["type"] == "allowlist"
        assert len(policy["allowed_domains"]) == 3
        assert "pypi.org" in policy["allowed_domains"]

    def test_domain_secrets(self):
        """Domain secrets for private authorization headers."""
        tool = ShellTool(
            environment={
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
        )
        secrets = tool.environment["network_policy"]["domain_secrets"]
        assert len(secrets) == 1
        assert secrets[0]["domain"] == "httpbin.org"
        assert secrets[0]["name"] == "API_KEY"

    def test_extra_fields_allowed(self):
        tool = ShellTool(
            environment={"type": "container_auto"},
            timeout=30000,
        )
        data = tool.model_dump()
        assert data["timeout"] == 30000


class TestImageGenerationTool:
    def test_basic(self):
        tool = ImageGenerationTool()
        assert tool.type == "image_generation"

    def test_with_output_options(self):
        """All configurable output options from the docs: size, quality, format, etc."""
        tool = ImageGenerationTool(
            size="1024x1536",
            quality="high",
            output_format="png",
            background="transparent",
            action="generate",
        )
        assert tool.type == "image_generation"
        data = tool.model_dump()
        assert data["size"] == "1024x1536"
        assert data["quality"] == "high"
        assert data["output_format"] == "png"
        assert data["background"] == "transparent"
        assert data["action"] == "generate"

    def test_with_auto_options(self):
        """size, quality, and background support 'auto' for model selection."""
        tool = ImageGenerationTool(
            size="auto",
            quality="auto",
            background="auto",
        )
        data = tool.model_dump()
        assert data["size"] == "auto"
        assert data["quality"] == "auto"
        assert data["background"] == "auto"

    def test_with_compression(self):
        """compression for JPEG/WebP formats (0-100)."""
        tool = ImageGenerationTool(
            output_format="webp",
            compression=75,
        )
        data = tool.model_dump()
        assert data["output_format"] == "webp"
        assert data["compression"] == 75

    def test_with_partial_images(self):
        """partial_images for streaming (1-3)."""
        tool = ImageGenerationTool(partial_images=2)
        data = tool.model_dump()
        assert data["partial_images"] == 2

    def test_action_edit(self):
        """action='edit' forces editing an existing image."""
        tool = ImageGenerationTool(action="edit")
        data = tool.model_dump()
        assert data["action"] == "edit"

    def test_action_auto(self):
        """action='auto' lets model choose generate vs edit."""
        tool = ImageGenerationTool(action="auto")
        data = tool.model_dump()
        assert data["action"] == "auto"


class TestNamespaceTool:
    def test_basic(self):
        tool = NamespaceTool(name="math_tools", description="Math utilities")
        assert tool.type == "namespace"
        assert tool.name == "math_tools"
        assert tool.description == "Math utilities"
        assert tool.tools == []

    def test_with_nested_functions(self):
        tool = NamespaceTool(
            name="math",
            description="Math tools",
            tools=[
                {
                    "type": "function",
                    "name": "add",
                    "description": "Add two numbers",
                    "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}},
                },
                {
                    "type": "function",
                    "name": "multiply",
                    "description": "Multiply two numbers",
                    "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}},
                },
            ],
        )
        assert len(tool.tools) == 2
        assert tool.tools[0]["name"] == "add"
        assert tool.tools[1]["name"] == "multiply"

    def test_extra_fields_allowed(self):
        tool = NamespaceTool(name="test", custom_field="value")
        assert tool.custom_field == "value"


class TestFunctionToolDeferLoading:
    def test_defer_loading_default_none(self):
        tool = FunctionTool(name="test")
        assert tool.defer_loading is None

    def test_defer_loading_true(self):
        tool = FunctionTool(name="test", defer_loading=True)
        assert tool.defer_loading is True

    def test_extra_fields_allowed(self):
        """FunctionTool uses extra='allow' for forward-compatibility."""
        tool = FunctionTool(name="test", some_future_field="value")
        assert tool.some_future_field == "value"


class TestUserExample_FileSearch:
    """Exact pattern from user's file_search example."""

    def test_file_search_with_vector_store_ids(self):
        tool = FileSearchTool(
            vector_store_ids=["vs_abc123"],
            max_num_results=5,
        )
        assert tool.type == "file_search"
        assert tool.vector_store_ids == ["vs_abc123"]
        assert tool.max_num_results == 5


class TestUserExample_NamespaceToolSearch:
    """Exact pattern from user's namespace + tool_search + defer_loading example."""

    def test_namespace_with_deferred_functions_and_tool_search(self):
        # Build the exact tools list from the user's example
        tools = [
            NamespaceTool(
                name="math",
                description="A collection of math tools",
                tools=[
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
            ),
            ToolSearchTool(),
        ]

        assert tools[0].type == "namespace"
        assert tools[0].name == "math"
        assert len(tools[0].tools) == 2
        assert tools[0].tools[0]["defer_loading"] is True
        assert tools[0].tools[1]["defer_loading"] is True
        assert tools[1].type == "tool_search"

    def test_deferred_function_as_model(self):
        """Verify FunctionTool model also supports defer_loading."""
        func = FunctionTool(
            name="add",
            description="Add two numbers together",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
            },
            defer_loading=True,
        )
        assert func.defer_loading is True
        assert func.type == "function"


class TestUserExample_MCPWithDescription:
    """Exact pattern from user's MCP with server_description example."""

    def test_mcp_with_server_description(self):
        tool = MCPTool(
            server_label="dmcp",
            server_url="https://dmcp-server.deno.dev/sse",
            server_description="A D&D MCP server for dice rolling and character management.",
            require_approval="never",
        )
        assert tool.type == "mcp"
        assert tool.server_label == "dmcp"
        assert tool.server_url == "https://dmcp-server.deno.dev/sse"
        assert tool.server_description == "A D&D MCP server for dice rolling and character management."
        assert tool.require_approval == "never"


class TestCustomToolGrammar:
    def test_lark_grammar(self):
        grammar = CustomToolGrammar(type="lark", value='start: expr\nexpr: NUMBER\nNUMBER: /[0-9]+/')
        assert grammar.type == "lark"
        assert "start:" in grammar.value

    def test_regex_grammar(self):
        grammar = CustomToolGrammar(type="regex", value=r"[A-Za-z]+\s\d{1,2}(st|nd|rd|th)\s\d{4}")
        assert grammar.type == "regex"


class TestCustomTool:
    def test_basic(self):
        tool = CustomTool(name="code_exec", description="Execute Python code")
        assert tool.type == "custom_tool"
        assert tool.name == "code_exec"
        assert tool.description == "Execute Python code"
        assert tool.grammar is None

    def test_with_lark_grammar(self):
        tool = CustomTool(
            name="math_exp",
            description="A mathematical expression",
            grammar=CustomToolGrammar(
                type="lark",
                value='start: expr\nexpr: term (("+"|"-") term)*\nterm: NUMBER\nNUMBER: /[0-9]+/',
            ),
        )
        assert tool.grammar.type == "lark"

    def test_with_regex_grammar(self):
        tool = CustomTool(
            name="timestamp",
            description="A human-readable date-time string",
            grammar={"type": "regex", "value": r"[A-Za-z]+ \d{1,2}(st|nd|rd|th) \d{4} at \d{1,2}(AM|PM)"},
        )
        assert tool.type == "custom_tool"
        # Dict is coerced to CustomToolGrammar
        assert tool.grammar.type == "regex"

    def test_without_grammar(self):
        """Custom tool with no grammar — free-text input from model."""
        tool = CustomTool(
            name="code_exec",
            description="Execute the given Python code",
        )
        assert tool.grammar is None

    def test_extra_fields_allowed(self):
        tool = CustomTool(name="test", output_format="json")
        assert tool.output_format == "json"
