from __future__ import annotations


class CaipResponsesError(Exception):
    """Base exception for caip-responses-lib."""


class ProviderError(CaipResponsesError):
    """Raised when an LLM provider returns an error."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: int | None = None,
        raw_error: Exception | None = None,
    ) -> None:
        self.provider = provider
        self.status_code = status_code
        self.raw_error = raw_error
        super().__init__(f"[{provider}] {message}")


class ProviderNotFoundError(CaipResponsesError):
    """Raised when no provider is registered for a model."""

    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(
            f"No provider found for model '{model}'. "
            "Register a provider or pass provider= explicitly."
        )


class ProviderNotConfiguredError(CaipResponsesError):
    """Raised when a provider exists but is missing credentials."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        super().__init__(
            f"Provider '{provider}' is not configured. "
            "Supply the API key when creating the client."
        )


class MaxStepsExceededError(CaipResponsesError):
    """Raised when the agentic loop exceeds max_steps."""

    def __init__(self, max_steps: int) -> None:
        self.max_steps = max_steps
        super().__init__(
            f"Agentic loop exceeded {max_steps} steps without completing."
        )
