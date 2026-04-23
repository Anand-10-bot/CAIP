from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictBase(BaseModel):
    """Base model with strict validation — rejects unknown fields."""

    model_config = ConfigDict(extra="forbid")


class Usage(StrictBase):
    """Token usage breakdown for a response."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_tokens_details: dict[str, int] | None = None
    output_tokens_details: dict[str, int] | None = None


class Reasoning(StrictBase):
    """Configuration for reasoning/thinking models."""

    effort: Literal["low", "medium", "high"] | None = None
    summary: Literal["auto", "concise", "detailed"] | None = None


class TextFormat(StrictBase):
    """Plain text format configuration."""

    type: Literal["text"] = "text"


class JsonSchemaFormat(StrictBase):
    """JSON schema format for structured outputs."""

    type: Literal["json_schema"] = "json_schema"
    name: str = "response"
    description: str = ""
    schema_: dict[str, Any] = Field(default_factory=dict, alias="schema")
    strict: bool | None = None

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class TextConfig(StrictBase):
    """Configuration for text output format."""

    format: TextFormat | JsonSchemaFormat | None = None


class Metadata(BaseModel):
    """Arbitrary key-value metadata (max 16 pairs)."""

    model_config = ConfigDict(extra="allow")
