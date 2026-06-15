from __future__ import annotations

from caip_responses.providers.openai_compatible import OpenAICompatibleProvider


class OllamaProvider(OpenAICompatibleProvider):
    """Provider for open-source models served via an OpenAI-compatible endpoint.

    Works with any server that speaks the OpenAI ``/chat/completions`` API:

    - **Ollama** — ``http://localhost:11434/v1`` (Llama, Mistral, Qwen, Gemma, Phi)
    - **vLLM** — ``http://localhost:8000/v1``
    - **LM Studio** — ``http://localhost:1234/v1``
    - Any other self-hosted or hosted OpenAI-compatible gateway

    This is what lets CAIP swap GPT-4o for a self-hosted open-source model
    with a one-line model name change and no integration rewrite.

    Most local servers don't require an API key; pass one only if your
    gateway enforces auth. Point ``base_url`` at a non-default server to
    use vLLM or LM Studio instead of Ollama.

    Usage::

        client = AsyncClient(ollama_base_url="http://localhost:11434/v1")
        resp = await client.responses.create(
            model="llama3.1",
            input="Hello",
        )
    """

    PROVIDER_NAME = "ollama"
    DEFAULT_BASE_URL = "http://localhost:11434/v1"
    SUPPORTS_REASONING = True
