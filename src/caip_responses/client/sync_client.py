from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

from caip_responses.cache.response_cache import ResponseCache
from caip_responses.client.async_client import AsyncClient
from caip_responses.cost.tracker import CostTracker
from caip_responses.loop.tool_executor import ToolExecutor
from caip_responses.models.common import Reasoning, TextConfig
from caip_responses.models.response import Response
from caip_responses.models.streaming import StreamEvent
from caip_responses.plugins.manager import PluginManager
from caip_responses.providers.base import BaseProvider
from caip_responses.ratelimit.limiter import RateLimiter
from caip_responses.store.conversation_store import ConversationStore


class _SyncResponsesNamespace:
    """Sync wrapper for responses.create()."""

    def __init__(self, async_client: AsyncClient) -> None:
        self._async_client = async_client

    def create(
        self,
        *,
        model: str,
        input: str | list[dict[str, Any]] = "",
        instructions: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, str] | None = "auto",
        parallel_tool_calls: bool | None = None,
        stream: bool = False,
        previous_response_id: str | None = None,
        reasoning: Reasoning | dict[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_output_tokens: int | None = None,
        text: TextConfig | dict[str, Any] | None = None,
        prompt: dict[str, Any] | None = None,
        metadata: dict[str, str] | None = None,
        store: bool | None = None,
        truncation: str | None = None,
        user: str | None = None,
        include: list[str] | None = None,
        background: bool | None = None,
        provider: str | None = None,
        **kwargs: Any,
    ) -> Response | Iterator[StreamEvent]:
        """Synchronous version of responses.create()."""
        loop = self._get_loop()

        coro = self._async_client.responses.create(
            model=model,
            input=input,
            instructions=instructions,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            stream=stream,
            previous_response_id=previous_response_id,
            reasoning=reasoning,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            text=text,
            prompt=prompt,
            metadata=metadata,
            store=store,
            truncation=truncation,
            user=user,
            include=include,
            background=background,
            provider=provider,
            **kwargs,
        )

        result = loop.run_until_complete(coro)

        if stream:
            # Wrap async iterator into sync iterator
            return self._sync_stream(result, loop)
        return result

    def _sync_stream(
        self, async_iter: Any, loop: asyncio.AbstractEventLoop
    ) -> Iterator[StreamEvent]:
        """Convert an async iterator to a sync iterator."""
        try:
            while True:
                event = loop.run_until_complete(async_iter.__anext__())
                yield event
        except StopAsyncIteration:
            pass

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError(
                    "Cannot use sync Client inside an async context. "
                    "Use AsyncClient instead."
                )
            return loop
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop


class Client:
    """Synchronous client — wraps AsyncClient for non-async code.

    Usage:
        client = Client(openai_api_key="...")
        response = client.responses.create(model="gpt-4.1", input="Hello")
        print(response.output_text)
    """

    def __init__(
        self,
        *,
        openai_api_key: str | None = None,
        openai_base_url: str | None = None,
        anthropic_api_key: str | None = None,
        anthropic_base_url: str | None = None,
        gemini_api_key: str | None = None,
        sarvam_api_key: str | None = None,
        sarvam_base_url: str | None = None,
        default_provider: str | None = None,
        providers: dict[str, BaseProvider] | None = None,
        max_conversation_history: int = 1000,
        cache_max_size: int = 500,
        cache_ttl: int = 3600,
        enable_cache: bool = True,
        discover_plugins: bool = True,
    ) -> None:
        self._async_client = AsyncClient(
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            anthropic_api_key=anthropic_api_key,
            anthropic_base_url=anthropic_base_url,
            gemini_api_key=gemini_api_key,
            sarvam_api_key=sarvam_api_key,
            sarvam_base_url=sarvam_base_url,
            default_provider=default_provider,
            providers=providers,
            max_conversation_history=max_conversation_history,
            cache_max_size=cache_max_size,
            cache_ttl=cache_ttl,
            enable_cache=enable_cache,
            discover_plugins=discover_plugins,
        )
        self._responses = _SyncResponsesNamespace(self._async_client)

    @property
    def responses(self) -> _SyncResponsesNamespace:
        return self._responses

    @property
    def tools(self) -> ToolExecutor:
        """Access the tool executor to register function handlers."""
        return self._async_client.tools

    @property
    def conversation_store(self) -> ConversationStore:
        """Access the conversation store."""
        return self._async_client.conversation_store

    @property
    def rate_limiter(self) -> RateLimiter:
        """Access the rate limiter to configure per-provider limits."""
        return self._async_client.rate_limiter

    @property
    def cost_tracker(self) -> CostTracker:
        """Access the cost tracker to set pricing and query usage."""
        return self._async_client.cost_tracker

    @property
    def cache(self) -> ResponseCache:
        """Access the response cache."""
        return self._async_client.cache

    @property
    def plugins(self) -> PluginManager:
        """Access the plugin manager to register custom providers."""
        return self._async_client.plugins

    def close(self) -> None:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._async_client.close())
        finally:
            loop.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
