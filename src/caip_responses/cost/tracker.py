from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelPricing:
    """Pricing per million tokens for a specific model.

    Costs are in USD. Set to 0.0 if unknown/free.
    """

    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
    cached_input_cost_per_million: float = 0.0


@dataclass
class UsageRecord:
    """Accumulated usage for a single model."""

    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0
    total_cost_usd: float = 0.0


class CostTracker:
    """Tracks token usage and estimated costs across providers and models.

    Register pricing for models, then call `record()` after each response.
    Query accumulated costs per model, per provider, or total.

    Usage:
        tracker = CostTracker()
        tracker.set_pricing("gpt-4.1", ModelPricing(
            input_cost_per_million=2.0,
            output_cost_per_million=8.0,
        ))
        tracker.set_pricing("claude-sonnet-4-20250514", ModelPricing(
            input_cost_per_million=3.0,
            output_cost_per_million=15.0,
        ))

        # After each response:
        tracker.record(
            model="gpt-4.1",
            provider="openai",
            input_tokens=500,
            output_tokens=200,
        )

        print(tracker.total_cost)      # Total USD across all models
        print(tracker.by_provider())   # {"openai": 0.0026, "anthropic": 0.0}
        print(tracker.by_model())      # {"gpt-4.1": UsageRecord(...)}
    """

    def __init__(self) -> None:
        self._pricing: dict[str, ModelPricing] = {}
        self._usage: dict[str, UsageRecord] = {}  # keyed by model
        self._lock = threading.Lock()

    def set_pricing(self, model: str, pricing: ModelPricing) -> None:
        """Set or update pricing for a model.

        Also supports prefix matching: set_pricing("gpt-4.1", ...)
        will match "gpt-4.1" exactly.
        """
        self._pricing[model] = pricing

    def set_pricing_bulk(self, pricing_map: dict[str, ModelPricing]) -> None:
        """Set pricing for multiple models at once."""
        self._pricing.update(pricing_map)

    def record(
        self,
        *,
        model: str,
        provider: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> float:
        """Record token usage and calculate cost for a single request.

        Args:
            model: Model name used in the request.
            provider: Provider that served the request.
            input_tokens: Number of input/prompt tokens.
            output_tokens: Number of output/completion tokens.

        Returns:
            Estimated cost in USD for this request.
        """
        pricing = self._get_pricing(model)
        cost = 0.0
        if pricing:
            cost = (
                (input_tokens / 1_000_000) * pricing.input_cost_per_million
                + (output_tokens / 1_000_000) * pricing.output_cost_per_million
            )

        with self._lock:
            if model not in self._usage:
                self._usage[model] = UsageRecord(model=model, provider=provider)

            record = self._usage[model]
            record.input_tokens += input_tokens
            record.output_tokens += output_tokens
            record.requests += 1
            record.total_cost_usd += cost

        return cost

    @property
    def total_cost(self) -> float:
        """Total accumulated cost in USD across all models."""
        with self._lock:
            return sum(r.total_cost_usd for r in self._usage.values())

    @property
    def total_tokens(self) -> dict[str, int]:
        """Total tokens used: {"input": N, "output": N, "total": N}."""
        with self._lock:
            inp = sum(r.input_tokens for r in self._usage.values())
            out = sum(r.output_tokens for r in self._usage.values())
            return {"input": inp, "output": out, "total": inp + out}

    @property
    def total_requests(self) -> int:
        """Total number of requests tracked."""
        with self._lock:
            return sum(r.requests for r in self._usage.values())

    def by_model(self) -> dict[str, UsageRecord]:
        """Get usage records grouped by model."""
        with self._lock:
            return dict(self._usage)

    def by_provider(self) -> dict[str, float]:
        """Get total cost grouped by provider."""
        with self._lock:
            costs: dict[str, float] = {}
            for record in self._usage.values():
                costs[record.provider] = costs.get(record.provider, 0.0) + record.total_cost_usd
            return costs

    def get_model_usage(self, model: str) -> UsageRecord | None:
        """Get usage record for a specific model."""
        with self._lock:
            return self._usage.get(model)

    def reset(self) -> None:
        """Reset all accumulated usage and costs."""
        with self._lock:
            self._usage.clear()

    def summary(self) -> dict[str, Any]:
        """Get a summary of all usage and costs."""
        with self._lock:
            inp = sum(r.input_tokens for r in self._usage.values())
            out = sum(r.output_tokens for r in self._usage.values())
            costs_by_provider: dict[str, float] = {}
            for record in self._usage.values():
                costs_by_provider[record.provider] = (
                    costs_by_provider.get(record.provider, 0.0) + record.total_cost_usd
                )
            return {
                "total_cost_usd": sum(r.total_cost_usd for r in self._usage.values()),
                "total_requests": sum(r.requests for r in self._usage.values()),
                "total_tokens": {"input": inp, "output": out, "total": inp + out},
                "by_provider": costs_by_provider,
                "by_model": {
                    model: {
                        "provider": r.provider,
                        "input_tokens": r.input_tokens,
                        "output_tokens": r.output_tokens,
                        "requests": r.requests,
                        "cost_usd": r.total_cost_usd,
                    }
                    for model, r in self._usage.items()
                },
            }

    def _get_pricing(self, model: str) -> ModelPricing | None:
        """Look up pricing for a model (exact match)."""
        return self._pricing.get(model)
