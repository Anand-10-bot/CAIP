from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Content types
# ---------------------------------------------------------------------------

class InputTextContent(StrictBase):
    """Text content within an input message."""

    type: Literal["input_text"] = "input_text"
    text: str


class InputImageContent(StrictBase):
    """Image content within an input message."""

    type: Literal["input_image"] = "input_image"
    image_url: str | None = None
    detail: Literal["auto", "low", "high"] | None = None


class InputFileContent(StrictBase):
    """File content within an input message or prompt variable."""

    type: Literal["input_file"] = "input_file"
    file_id: str | None = None
    filename: str | None = None
    file_data: str | None = None


class OutputTextContent(StrictBase):
    """Text content within an output message."""

    type: Literal["output_text"] = "output_text"
    text: str
    annotations: list[dict[str, Any]] = Field(default_factory=list)


class OutputRefusalContent(StrictBase):
    """Refusal content within an output message."""

    type: Literal["refusal"] = "refusal"
    refusal: str


# ---------------------------------------------------------------------------
# Input items
# ---------------------------------------------------------------------------

class InputMessage(BaseModel):
    """A message in the input conversation."""

    model_config = ConfigDict(extra="allow")

    role: Literal["user", "assistant", "developer", "system"]
    content: str | list[InputTextContent | InputImageContent | InputFileContent | OutputTextContent | dict[str, Any]]


class FunctionCallOutputItem(StrictBase):
    """Result of executing a function call — sent back as input."""

    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    output: str


class MCPApprovalResponseItem(BaseModel):
    """MCP approval response — developer approves or denies an MCP tool call."""

    model_config = ConfigDict(extra="allow")

    type: Literal["mcp_approval_response"] = "mcp_approval_response"
    approve: bool
    approval_request_id: str


class ComputerCallOutputItem(BaseModel):
    """Screenshot sent back after executing computer_call actions.

    The ``output`` field contains a screenshot as an image object with
    ``type: "input_image"`` and ``image_url`` (base64 data-URI or URL).
    Use ``detail: "original"`` for best click accuracy.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["computer_call_output"] = "computer_call_output"
    call_id: str
    output: dict[str, Any]
    acknowledged_safety_checks: list[dict[str, Any]] = Field(default_factory=list)


class ItemReference(StrictBase):
    """Reference to a previous response item by ID."""

    type: Literal["item_reference"] = "item_reference"
    id: str


# Union of all input item types (includes tool outputs sent back in local/client modes)
InputItem = (
    InputMessage
    | FunctionCallOutputItem
    | ComputerCallOutputItem
    | MCPApprovalResponseItem
    | ItemReference
)
# Note: ShellCallOutputItem and ToolSearchOutputItem are also valid input items
# but are defined below alongside their output counterparts to keep related
# types together.  They pass through as dicts in the request's ``input`` field.


# ---------------------------------------------------------------------------
# Output items
# ---------------------------------------------------------------------------

class MessageOutputItem(StrictBase):
    """Assistant text message output item."""

    type: Literal["message"] = "message"
    id: str
    role: Literal["assistant"] = "assistant"
    content: list[OutputTextContent | OutputRefusalContent] = Field(default_factory=list)
    status: Literal["completed", "incomplete"] = "completed"


class FunctionCallItem(StrictBase):
    """Model wants to call a user-defined function."""

    type: Literal["function_call"] = "function_call"
    id: str
    call_id: str
    name: str
    arguments: str  # JSON string


class ReasoningItem(BaseModel):
    """Reasoning/thinking tokens from the model."""

    model_config = ConfigDict(extra="allow")

    type: Literal["reasoning"] = "reasoning"
    id: str
    content: list[dict[str, Any]] = Field(default_factory=list)
    summary: list[dict[str, str]] = Field(default_factory=list)
    encrypted_content: str | None = None


class WebSearchCallItem(BaseModel):
    """Web search tool invocation result.

    The action field describes what the model did:
    - "search": a web search (may include queries list)
    - "open_page": opened a specific page (reasoning models)
    - "find_in_page": searched within a page (reasoning models)
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["web_search_call"] = "web_search_call"
    id: str
    status: str = "completed"
    action: dict[str, Any] | None = None


class FileSearchCallItem(BaseModel):
    """File search tool invocation result.

    When ``include=["file_search_call.results"]`` is set on the request,
    the ``results`` field contains the search results with scores and content.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["file_search_call"] = "file_search_call"
    id: str
    status: str = "completed"
    queries: list[str] = Field(default_factory=list)
    results: list[dict[str, Any]] | None = None


class ComputerCallItem(BaseModel):
    """Computer use tool invocation result.

    The ``actions`` array contains batched UI actions (click, type, scroll,
    keypress, drag, move, wait, screenshot, double_click) that your harness
    must execute in order.  After executing, send back a screenshot as
    ``computer_call_output``.

    ``call_id`` ties the call to its corresponding output item.
    ``pending_safety_checks`` may contain safety check objects that the
    application should surface to the user before executing.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["computer_call"] = "computer_call"
    id: str
    call_id: str = ""
    actions: list[dict[str, Any]] = Field(default_factory=list)
    status: str = "completed"
    pending_safety_checks: list[dict[str, Any]] = Field(default_factory=list)


class CodeInterpreterCallItem(StrictBase):
    """Code interpreter tool invocation result."""

    type: Literal["code_interpreter_call"] = "code_interpreter_call"
    id: str
    status: str = "completed"
    code: str = ""


class MCPCallItem(BaseModel):
    """MCP tool call — shows what the model sent and what the server returned."""

    model_config = ConfigDict(extra="allow")

    type: Literal["mcp_call"] = "mcp_call"
    id: str
    name: str = ""
    arguments: str = ""
    output: str | None = None
    error: str | None = None
    server_label: str = ""
    approval_request_id: str | None = None
    status: str = "completed"


class MCPListToolsItem(BaseModel):
    """MCP list tools — tools discovered from an MCP server or connector."""

    model_config = ConfigDict(extra="allow")

    type: Literal["mcp_list_tools"] = "mcp_list_tools"
    id: str
    server_label: str = ""
    tools: list[dict[str, Any]] = Field(default_factory=list)


class MCPApprovalRequestItem(BaseModel):
    """MCP approval request — model wants permission to call an MCP tool."""

    model_config = ConfigDict(extra="allow")

    type: Literal["mcp_approval_request"] = "mcp_approval_request"
    id: str
    name: str = ""
    arguments: str = ""
    server_label: str = ""


class ShellCallItem(BaseModel):
    """Shell tool call — commands requested by the model.

    In hosted mode, actions execute server-side and results appear
    as ``shell_call_output`` in the same response.
    In local mode, your harness executes the commands and returns
    ``ShellCallOutputItem`` as input in the next request.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["shell_call"] = "shell_call"
    id: str
    status: str = "completed"
    call_id: str = ""
    action: dict[str, Any] = Field(default_factory=dict)
    max_output_length: int | None = None


class ShellCallOutputItem(BaseModel):
    """Shell call output — result of executing shell commands.

    In hosted mode this appears as an output item alongside ``shell_call``.
    In local mode, send this as an input item with stdout/stderr/outcome
    after executing the ``shell_call`` actions.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["shell_call_output"] = "shell_call_output"
    id: str = ""
    call_id: str = ""
    output: str = ""
    error: str = ""
    status: str = "completed"


class ImageGenerationCallItem(BaseModel):
    """Image generation tool call result.

    The ``result`` field contains the base64-encoded generated image.
    When the mainline model revises the prompt for better image quality,
    the revised text appears in ``revised_prompt``.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["image_generation_call"] = "image_generation_call"
    id: str
    status: str = "completed"
    result: str | None = None
    revised_prompt: str | None = None


class CustomToolCallItem(BaseModel):
    """Custom tool call — the model returns free-text input instead of JSON arguments."""

    model_config = ConfigDict(extra="allow")

    type: Literal["custom_tool_call"] = "custom_tool_call"
    id: str
    call_id: str
    name: str
    input: str  # plain text, not JSON
    status: str = "completed"


class ToolSearchCallItem(BaseModel):
    """Tool search invocation — model searched for relevant tools.

    In hosted mode: ``execution="server"``, ``call_id=None``.
    In client mode: ``execution="client"``, ``call_id`` is set (echo it back
    in the ``tool_search_output``).
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["tool_search_call"] = "tool_search_call"
    id: str
    call_id: str | None = None
    execution: str | None = None
    status: str = "completed"


class ToolSearchOutputItem(BaseModel):
    """Tool search output — tools found and loaded by tool_search.

    The ``tools`` list contains the tool definitions that become callable.
    In client mode, ``call_id`` must echo the ``tool_search_call``'s call_id.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["tool_search_output"] = "tool_search_output"
    id: str
    call_id: str | None = None
    tools: list[dict[str, Any]] = Field(default_factory=list)
    output: list[dict[str, Any]] = Field(default_factory=list)


OutputItem = Annotated[
    MessageOutputItem
    | FunctionCallItem
    | ReasoningItem
    | WebSearchCallItem
    | FileSearchCallItem
    | ComputerCallItem
    | CodeInterpreterCallItem
    | MCPCallItem
    | MCPListToolsItem
    | MCPApprovalRequestItem
    | ShellCallItem
    | ShellCallOutputItem
    | ImageGenerationCallItem
    | CustomToolCallItem
    | ToolSearchCallItem
    | ToolSearchOutputItem,
    Field(discriminator="type"),
]
