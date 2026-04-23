from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from caip_responses.models.request import CreateResponseRequest
from caip_responses.models.response import Response
from caip_responses.models.streaming import StreamEvent


class BaseProvider(ABC):
    """Abstract base class for LLM provider adapters.

    Each provider translates the unified Responses API format
    to/from its native API format. The caller never sees
    provider-specific details — the interface is identical.
    """

    @abstractmethod
    async def create_response(
        self, request: CreateResponseRequest
    ) -> Response:
        """Non-streaming response creation."""
        ...

    @abstractmethod
    async def create_response_stream(
        self, request: CreateResponseRequest
    ) -> AsyncIterator[StreamEvent]:
        """Streaming response creation. Yields unified StreamEvents."""
        ...

    @abstractmethod
    def supports_tool(self, tool_type: str) -> bool:
        """Whether this provider natively supports a given tool type."""
        ...

    @abstractmethod
    def supports_reasoning(self) -> bool:
        """Whether this provider supports reasoning/thinking tokens."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Unique provider identifier (e.g., 'openai', 'anthropic')."""
        ...

    async def close(self) -> None:
        """Cleanup resources (HTTP clients, etc.). Override if needed."""
