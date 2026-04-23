"""Base class for tool handlers that delegate execution to Azure OpenAI.

When a non-OpenAI provider (Claude, Gemini, Sarvam) doesn't support a tool
natively, the handler intercepts the call and routes it through Azure OpenAI's
Responses API — which supports the tool server-side. The results (and token
usage for billing) are then returned to the caller.

This keeps a single OpenAI client instance and tracks cumulative token usage
across all delegated tool calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DelegatedToolUsage:
    """Token usage from a single delegated OpenAI call."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    provider: str = "openai"
    tool_type: str = ""


@dataclass
class DelegatedToolMetrics:
    """Accumulated metrics for all delegated tool calls."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_calls: int = 0
    calls: list[DelegatedToolUsage] = field(default_factory=list)

    def record(self, usage: DelegatedToolUsage) -> None:
        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens
        self.total_tokens += usage.total_tokens
        self.total_calls += 1
        self.calls.append(usage)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_calls": self.total_calls,
            "calls": [
                {
                    "input_tokens": c.input_tokens,
                    "output_tokens": c.output_tokens,
                    "total_tokens": c.total_tokens,
                    "model": c.model,
                    "provider": c.provider,
                    "tool_type": c.tool_type,
                }
                for c in self.calls
            ],
        }


class OpenAIDelegatorMixin:
    """Mixin for handlers that delegate tool execution to Azure OpenAI.

    Provides a shared, lazily-initialised OpenAI client and usage tracking.
    """

    def __init__(
        self,
        *,
        openai_api_key: str | None = None,
        openai_base_url: str | None = None,
        openai_model: str = "gpt-4.1-nano",
    ) -> None:
        self._openai_api_key = openai_api_key
        self._openai_base_url = openai_base_url
        self._openai_model = openai_model
        self._openai_client: Any = None
        self._delegated_metrics = DelegatedToolMetrics()

    @property
    def delegated_metrics(self) -> DelegatedToolMetrics:
        """Accumulated token usage from OpenAI-delegated calls."""
        return self._delegated_metrics

    def reset_delegated_metrics(self) -> None:
        self._delegated_metrics = DelegatedToolMetrics()

    @property
    def last_delegated_usage(self) -> DelegatedToolUsage | None:
        if self._delegated_metrics.calls:
            return self._delegated_metrics.calls[-1]
        return None

    def _get_openai_client(self) -> Any:
        """Get or create the OpenAI async client."""
        if self._openai_client is not None:
            return self._openai_client

        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for delegating tool calls to "
                "Azure OpenAI. Install with: pip install caip-responses-lib[openai]"
            )

        kwargs: dict[str, Any] = {"api_key": self._openai_api_key}
        if self._openai_base_url:
            kwargs["base_url"] = self._openai_base_url
        self._openai_client = AsyncOpenAI(**kwargs)
        return self._openai_client

    def _record_usage(
        self, response: Any, tool_type: str
    ) -> DelegatedToolUsage:
        """Extract and record token usage from an OpenAI response."""
        usage = DelegatedToolUsage(
            model=self._openai_model,
            provider="openai",
            tool_type=tool_type,
        )
        if response.usage:
            usage.input_tokens = response.usage.input_tokens
            usage.output_tokens = response.usage.output_tokens
            usage.total_tokens = response.usage.total_tokens
        self._delegated_metrics.record(usage)
        return usage

    def _extract_text_from_response(self, response: Any) -> str:
        """Extract all text output from an OpenAI response."""
        parts: list[str] = []
        for item in response.output:
            item_type = getattr(item, "type", None)
            if item_type == "message":
                for block in getattr(item, "content", []):
                    if getattr(block, "type", None) == "output_text":
                        parts.append(getattr(block, "text", ""))
        return "".join(parts)
