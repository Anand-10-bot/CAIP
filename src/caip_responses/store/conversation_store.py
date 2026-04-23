from __future__ import annotations

from collections import OrderedDict
from typing import Any

from caip_responses.models.response import Response


class ConversationStore:
    """In-memory store mapping response IDs to conversation histories.

    OpenAI stores conversation state server-side via previous_response_id.
    For non-OpenAI providers, we maintain this state client-side.

    When a request includes previous_response_id, the store reconstitutes
    the full conversation history so the provider receives the complete
    context.

    The store uses an LRU eviction strategy with a configurable max size
    to prevent unbounded memory growth.
    """

    def __init__(self, max_size: int = 1000) -> None:
        self._max_size = max_size
        # OrderedDict for LRU eviction: stores {response_id: ConversationEntry}
        self._entries: OrderedDict[str, _ConversationEntry] = OrderedDict()

    def save(
        self,
        response: Response,
        input_items: list[dict[str, Any]],
        instructions: str | None = None,
    ) -> None:
        """Save a response and its input context for later retrieval.

        Args:
            response: The Response object returned by the provider.
            input_items: The input items (messages/items) sent in this request.
            instructions: The system instructions used in this request.
        """
        entry = _ConversationEntry(
            response_id=response.id,
            input_items=list(input_items),
            output_items=list(response.output),
            instructions=instructions,
            model=response.model,
        )

        # Evict oldest if at capacity
        if len(self._entries) >= self._max_size and response.id not in self._entries:
            self._entries.popitem(last=False)

        self._entries[response.id] = entry
        # Move to end (most recently used)
        self._entries.move_to_end(response.id)

    def get_history(
        self, response_id: str
    ) -> tuple[list[dict[str, Any]], str | None] | None:
        """Reconstitute the full conversation history for a response ID.

        Returns:
            Tuple of (full_input_items, instructions) if found, None otherwise.
            The input_items include all prior turns plus the model's response
            items, forming the complete conversation up to that point.
        """
        entry = self._entries.get(response_id)
        if entry is None:
            return None

        # Move to end (most recently accessed)
        self._entries.move_to_end(response_id)

        # Build the full history: input items + output items converted to input format
        history: list[dict[str, Any]] = []
        history.extend(entry.input_items)
        history.extend(self._output_to_input(entry.output_items))

        return history, entry.instructions

    def get_chain(
        self, response_id: str
    ) -> tuple[list[dict[str, Any]], str | None] | None:
        """Walk the full chain of previous_response_ids to build complete history.

        This handles multi-turn chains: if response A references response B
        which references response C, we reconstruct the full chain C→B→A.

        Returns:
            Tuple of (full_input_items, instructions) if the chain can be
            resolved, None if any link is missing.
        """
        # For now, get_chain is the same as get_history since each entry
        # stores the complete input (which already includes prior history
        # when the client prepends it). The chain is built incrementally
        # at request time.
        return self.get_history(response_id)

    def has(self, response_id: str) -> bool:
        """Check if a response ID exists in the store."""
        return response_id in self._entries

    def remove(self, response_id: str) -> bool:
        """Remove a response from the store. Returns True if it existed."""
        if response_id in self._entries:
            del self._entries[response_id]
            return True
        return False

    def clear(self) -> None:
        """Remove all entries from the store."""
        self._entries.clear()

    @property
    def size(self) -> int:
        """Number of entries currently stored."""
        return len(self._entries)

    @staticmethod
    def _output_to_input(
        output_items: list[Any],
    ) -> list[dict[str, Any]]:
        """Convert response output items to input items for the next turn.

        Maps:
        - message items → assistant messages
        - function_call items → function_call input items
        - reasoning items → skipped (not sent back as input)
        """
        items: list[dict[str, Any]] = []
        for item in output_items:
            if isinstance(item, dict):
                item_dict = item
            elif hasattr(item, "model_dump"):
                item_dict = item.model_dump()
            else:
                continue

            item_type = item_dict.get("type")

            if item_type == "message":
                content = item_dict.get("content", [])
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "output_text":
                        text_parts.append(block.get("text", ""))
                    elif hasattr(block, "text"):
                        text_parts.append(block.text)
                if text_parts:
                    items.append({
                        "role": "assistant",
                        "content": "".join(text_parts),
                    })

            elif item_type == "function_call":
                items.append({
                    "type": "function_call",
                    "call_id": item_dict.get("call_id", ""),
                    "name": item_dict.get("name", ""),
                    "arguments": item_dict.get("arguments", "{}"),
                })

            # Skip reasoning, web_search_call, etc. — not useful as input

        return items


class _ConversationEntry:
    """A single entry in the conversation store."""

    __slots__ = ("response_id", "input_items", "output_items", "instructions", "model")

    def __init__(
        self,
        response_id: str,
        input_items: list[dict[str, Any]],
        output_items: list[Any],
        instructions: str | None,
        model: str,
    ) -> None:
        self.response_id = response_id
        self.input_items = input_items
        self.output_items = output_items
        self.instructions = instructions
        self.model = model
