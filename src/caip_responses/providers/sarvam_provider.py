from __future__ import annotations

from caip_responses.providers.openai_compatible import OpenAICompatibleProvider


class SarvamProvider(OpenAICompatibleProvider):
    """Sarvam AI provider — translates Responses API to Chat Completions API.

    Sarvam exposes a standard OpenAI-compatible Chat Completions endpoint
    (``/chat/completions``), so all translation logic is inherited from
    :class:`OpenAICompatibleProvider`. The caller's code is identical to
    calling any other provider.
    """

    PROVIDER_NAME = "sarvam"
    DEFAULT_BASE_URL = "https://api.sarvam.ai/v1"
    SUPPORTS_REASONING = True
