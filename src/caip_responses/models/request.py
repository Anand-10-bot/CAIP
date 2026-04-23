from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from caip_responses.models.common import Reasoning, TextConfig


class PromptConfig(BaseModel):
    """Reusable prompt template configuration.

    Used with OpenAI's stored prompts feature — reference a prompt by ID
    and optionally override version and substitute variables.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    version: str | None = None
    variables: dict[str, Any] | None = None


class CreateResponseRequest(BaseModel):
    """Unified request model for creating a response — identical for all providers."""

    model_config = ConfigDict(extra="allow")

    # Required
    model: str

    # Input
    input: str | list[dict[str, Any]] = ""
    instructions: str | None = None

    # Reusable prompt template (OpenAI feature — ignored by other providers)
    prompt: PromptConfig | dict[str, Any] | None = None

    # Tools
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = "auto"
    parallel_tool_calls: bool | None = None

    # Generation parameters
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None

    # Reasoning
    reasoning: Reasoning | dict[str, Any] | None = None

    # Output format
    text: TextConfig | dict[str, Any] | None = None

    # Conversation state
    previous_response_id: str | None = None
    store: bool | None = None

    # Metadata
    metadata: dict[str, str] | None = None
    user: str | None = None

    # Truncation
    truncation: Literal["auto", "disabled"] | None = None

    # Streaming
    stream: bool = False

    # Background execution
    background: bool | None = None

    # Include additional data in response
    include: list[str] | None = None

    # Provider override (our extension — not in OpenAI spec)
    provider: str | None = Field(default=None, exclude=True)
