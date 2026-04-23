"""Conversation state store — enables previous_response_id for non-OpenAI providers."""

from caip_responses.store.conversation_store import ConversationStore

__all__ = ["ConversationStore"]

# RedisConversationStore is imported lazily to avoid requiring redis
# at import time.  Use: from caip_responses.store.redis_store import RedisConversationStore
