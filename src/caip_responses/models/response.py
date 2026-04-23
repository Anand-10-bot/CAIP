from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from caip_responses.models.common import Usage
from caip_responses.models.items import (
    FunctionCallItem,
    MessageOutputItem,
    OutputTextContent,
)


class Response(BaseModel):
    """Unified response object — identical structure regardless of provider."""

    model_config = ConfigDict(extra="allow")

    id: str
    object: Literal["response"] = "response"
    created_at: int = 0
    model: str = ""
    status: Literal["completed", "failed", "incomplete", "in_progress"] = "completed"
    output: list[dict[str, Any] | MessageOutputItem | FunctionCallItem] = Field(
        default_factory=list
    )
    usage: Usage | None = None
    metadata: dict[str, str] | None = None
    error: dict[str, Any] | None = None
    incomplete_details: dict[str, Any] | None = None

    @property
    def output_text(self) -> str:
        """Extract concatenated text from all message output items.

        Matches the OpenAI SDK's `response.output_text` convenience property.
        """
        parts: list[str] = []
        for item in self.output:
            # Handle both Pydantic model instances and raw dicts
            if isinstance(item, MessageOutputItem):
                for block in item.content:
                    if isinstance(block, OutputTextContent):
                        parts.append(block.text)
            elif isinstance(item, dict):
                if item.get("type") == "message":
                    for block in item.get("content", []):
                        if isinstance(block, dict) and block.get("type") == "output_text":
                            parts.append(block.get("text", ""))
                        elif isinstance(block, OutputTextContent):
                            parts.append(block.text)
        return "".join(parts)

    @property
    def has_function_calls(self) -> bool:
        """Check if the response contains any function call items."""
        for item in self.output:
            if isinstance(item, FunctionCallItem):
                return True
            if isinstance(item, dict) and item.get("type") == "function_call":
                return True
        return False

    @property
    def function_calls(self) -> list[FunctionCallItem]:
        """Extract all function call items from the output."""
        calls: list[FunctionCallItem] = []
        for item in self.output:
            if isinstance(item, FunctionCallItem):
                calls.append(item)
            elif isinstance(item, dict) and item.get("type") == "function_call":
                calls.append(FunctionCallItem(**item))
        return calls
