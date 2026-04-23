from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FunctionTool(BaseModel):
    """A user-defined function tool."""

    model_config = ConfigDict(extra="allow")

    type: Literal["function"] = "function"
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    strict: bool | None = None
    defer_loading: bool | None = None


class WebSearchTool(BaseModel):
    """Built-in web search tool."""

    model_config = ConfigDict(extra="allow")

    type: Literal["web_search"] = "web_search"
    search_context_size: Literal["low", "medium", "high"] | None = "medium"
    user_location: dict[str, Any] | None = None
    external_web_access: bool | None = None
    filters: dict[str, Any] | None = None


class WebSearchPreviewTool(BaseModel):
    """Earlier preview version of the web search tool.

    Accepts the same parameters as WebSearchTool but uses type
    "web_search_preview". Ignores external_web_access (always live).
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["web_search_preview"] = "web_search_preview"
    search_context_size: Literal["low", "medium", "high"] | None = "medium"
    user_location: dict[str, Any] | None = None
    filters: dict[str, Any] | None = None


class FileSearchTool(BaseModel):
    """Built-in file search tool — semantic and keyword search over vector stores."""

    model_config = ConfigDict(extra="allow")

    type: Literal["file_search"] = "file_search"
    vector_store_ids: list[str] = Field(default_factory=list)
    max_num_results: int | None = None
    filters: dict[str, Any] | None = None


class CodeInterpreterTool(StrictBase):
    """Built-in code interpreter tool."""

    type: Literal["code_interpreter"] = "code_interpreter"
    container: str | None = None


class ComputerTool(BaseModel):
    """Built-in computer use tool (GA).

    The model returns structured UI actions (click, type, scroll, etc.)
    in a batched ``actions[]`` array on each ``computer_call``.
    Your harness executes the actions and returns a screenshot as
    ``computer_call_output``.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["computer"] = "computer"
    display_width: int = 1024
    display_height: int = 768
    environment: str | None = None


class ComputerUseTool(BaseModel):
    """Built-in computer use tool (preview / deprecated).

    Use ``ComputerTool`` (type="computer") for new integrations.
    This model is kept for backward compatibility with the
    ``computer-use-preview`` model.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["computer_use"] = "computer_use"
    display_width: int = 1024
    display_height: int = 768
    environment: str | None = None


class MCPTool(BaseModel):
    """MCP tool — connects to remote MCP servers or OpenAI connectors.

    For remote MCP servers, provide ``server_url``.
    For connectors (Dropbox, Gmail, etc.), provide ``connector_id`` instead.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["mcp"] = "mcp"
    server_label: str
    server_url: str | None = None
    connector_id: str | None = None
    authorization: str | None = None
    require_approval: str | dict[str, Any] | None = "never"
    server_description: str | None = None
    allowed_tools: list[str] | None = None
    defer_loading: bool | None = None


class NamespaceTool(BaseModel):
    """Namespace tool — groups related tools under a named scope.

    Used with tool_search to organize large tool sets. The model can
    selectively load tools from a namespace when needed.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["namespace"] = "namespace"
    name: str
    description: str = ""
    tools: list[dict[str, Any]] = Field(default_factory=list)


class ToolSearchTool(BaseModel):
    """Tool search — dynamically loads relevant tool definitions at runtime.

    Defers tool loading until the model decides they're needed,
    optimizing token usage for large tool sets.

    Modes:
    - Hosted (default): ``execution="server"`` — OpenAI searches deferred tools
      and returns the loaded subset in the same response.
    - Client-executed: ``execution="client"`` — the model emits a
      ``tool_search_call``, your application performs the lookup, and returns
      a ``tool_search_output`` with matched tools.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["tool_search"] = "tool_search"
    execution: Literal["server", "client"] | None = None
    # Note: "schema" for client-mode search arguments passes through via
    # extra="allow" to avoid shadowing BaseModel.schema.


class ShellTool(BaseModel):
    """Shell tool — run shell commands in hosted containers or local runtime.

    Use ``environment`` to configure the execution mode and attach skills:
    - ``container_auto``: hosted container with optional skill_reference skills
    - ``local``: local execution with skills specified by name/description/path
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["shell"] = "shell"
    environment: dict[str, Any] | None = None


class ImageGenerationTool(BaseModel):
    """Image generation tool — generate or edit images using GPT Image."""

    model_config = ConfigDict(extra="allow")

    type: Literal["image_generation"] = "image_generation"


class CustomToolGrammar(BaseModel):
    """Grammar definition for constraining custom tool input."""

    model_config = ConfigDict(extra="allow")

    type: Literal["lark", "regex"]
    value: str


class CustomTool(BaseModel):
    """Custom tool — free-text input/output, optionally constrained by a grammar.

    Unlike function tools which use JSON schema, custom tools receive
    plain text input from the model. A grammar (Lark CFG or regex) can
    optionally constrain the model's output.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["custom_tool"] = "custom_tool"
    name: str
    description: str = ""
    grammar: CustomToolGrammar | dict[str, Any] | None = None


ToolDefinition = Annotated[
    FunctionTool
    | WebSearchTool
    | WebSearchPreviewTool
    | FileSearchTool
    | CodeInterpreterTool
    | ComputerTool
    | ComputerUseTool
    | MCPTool
    | NamespaceTool
    | ToolSearchTool
    | ShellTool
    | ImageGenerationTool
    | CustomTool,
    Field(discriminator="type"),
]
