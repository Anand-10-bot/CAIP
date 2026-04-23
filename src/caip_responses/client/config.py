from __future__ import annotations

from pydantic_settings import BaseSettings


class CaipResponsesConfig(BaseSettings):
    """Configuration via environment variables or explicit init.

    All env vars are prefixed with CAIP_RESPONSES_.
    Example: CAIP_RESPONSES_OPENAI_API_KEY=sk-...
    """

    # Provider API keys
    openai_api_key: str = ""
    openai_base_url: str = ""
    anthropic_api_key: str = ""
    anthropic_base_url: str = ""
    gemini_api_key: str = ""
    sarvam_api_key: str = ""
    sarvam_base_url: str = "https://api.sarvam.ai/v1"

    # Redis (for persistent conversation store + cache)
    redis_url: str = ""

    # Defaults
    default_provider: str = ""
    agent_loop_max_steps: int = 10
    conversation_ttl: int = 86400

    model_config = {"env_prefix": "CAIP_RESPONSES_"}
