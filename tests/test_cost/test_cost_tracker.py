from __future__ import annotations

import pytest

from caip_responses.cost.tracker import CostTracker, ModelPricing, UsageRecord


class TestModelPricing:
    def test_defaults(self):
        p = ModelPricing()
        assert p.input_cost_per_million == 0.0
        assert p.output_cost_per_million == 0.0
        assert p.cached_input_cost_per_million == 0.0

    def test_custom_values(self):
        p = ModelPricing(
            input_cost_per_million=2.0,
            output_cost_per_million=8.0,
            cached_input_cost_per_million=0.5,
        )
        assert p.input_cost_per_million == 2.0
        assert p.output_cost_per_million == 8.0
        assert p.cached_input_cost_per_million == 0.5

    def test_immutable(self):
        p = ModelPricing(input_cost_per_million=3.0)
        with pytest.raises(AttributeError):
            p.input_cost_per_million = 5.0  # type: ignore[misc]


class TestUsageRecord:
    def test_defaults(self):
        r = UsageRecord(model="gpt-4.1", provider="openai")
        assert r.input_tokens == 0
        assert r.output_tokens == 0
        assert r.requests == 0
        assert r.total_cost_usd == 0.0

    def test_accumulation(self):
        r = UsageRecord(model="gpt-4.1", provider="openai")
        r.input_tokens += 500
        r.output_tokens += 200
        r.requests += 1
        assert r.input_tokens == 500
        assert r.output_tokens == 200


class TestCostTracker:
    @pytest.fixture
    def tracker(self):
        t = CostTracker()
        t.set_pricing("gpt-4.1", ModelPricing(
            input_cost_per_million=2.0,
            output_cost_per_million=8.0,
        ))
        t.set_pricing("claude-sonnet-4-20250514", ModelPricing(
            input_cost_per_million=3.0,
            output_cost_per_million=15.0,
        ))
        return t

    def test_initial_state(self, tracker):
        assert tracker.total_cost == 0.0
        assert tracker.total_requests == 0
        assert tracker.total_tokens == {"input": 0, "output": 0, "total": 0}

    def test_record_single_request(self, tracker):
        cost = tracker.record(
            model="gpt-4.1",
            provider="openai",
            input_tokens=1_000_000,
            output_tokens=500_000,
        )
        # 1M input * 2.0/M + 500K output * 8.0/M = 2.0 + 4.0 = 6.0
        assert cost == pytest.approx(6.0)
        assert tracker.total_cost == pytest.approx(6.0)
        assert tracker.total_requests == 1

    def test_record_multiple_models(self, tracker):
        tracker.record(model="gpt-4.1", provider="openai", input_tokens=100, output_tokens=50)
        tracker.record(
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            input_tokens=200,
            output_tokens=100,
        )
        assert tracker.total_requests == 2
        tokens = tracker.total_tokens
        assert tokens["input"] == 300
        assert tokens["output"] == 150
        assert tokens["total"] == 450

    def test_record_unknown_model_zero_cost(self, tracker):
        cost = tracker.record(
            model="unknown-model",
            provider="custom",
            input_tokens=1000,
            output_tokens=500,
        )
        assert cost == 0.0
        # But tokens should still be tracked
        assert tracker.total_requests == 1
        usage = tracker.get_model_usage("unknown-model")
        assert usage is not None
        assert usage.input_tokens == 1000

    def test_by_model(self, tracker):
        tracker.record(model="gpt-4.1", provider="openai", input_tokens=100, output_tokens=50)
        tracker.record(model="gpt-4.1", provider="openai", input_tokens=200, output_tokens=100)
        models = tracker.by_model()
        assert "gpt-4.1" in models
        assert models["gpt-4.1"].requests == 2
        assert models["gpt-4.1"].input_tokens == 300

    def test_by_provider(self, tracker):
        tracker.record(model="gpt-4.1", provider="openai", input_tokens=1_000_000, output_tokens=0)
        tracker.record(
            model="claude-sonnet-4-20250514", provider="anthropic", input_tokens=1_000_000, output_tokens=0
        )
        providers = tracker.by_provider()
        assert providers["openai"] == pytest.approx(2.0)
        assert providers["anthropic"] == pytest.approx(3.0)

    def test_get_model_usage_none(self, tracker):
        assert tracker.get_model_usage("nonexistent") is None

    def test_set_pricing_bulk(self, tracker):
        tracker.set_pricing_bulk({
            "gemini-2.0-flash": ModelPricing(input_cost_per_million=0.1),
        })
        cost = tracker.record(
            model="gemini-2.0-flash",
            provider="gemini",
            input_tokens=1_000_000,
            output_tokens=0,
        )
        assert cost == pytest.approx(0.1)

    def test_reset(self, tracker):
        tracker.record(model="gpt-4.1", provider="openai", input_tokens=100, output_tokens=50)
        tracker.reset()
        assert tracker.total_cost == 0.0
        assert tracker.total_requests == 0

    def test_summary(self, tracker):
        tracker.record(model="gpt-4.1", provider="openai", input_tokens=100, output_tokens=50)
        summary = tracker.summary()
        assert "total_cost_usd" in summary
        assert "total_requests" in summary
        assert "by_provider" in summary
        assert "by_model" in summary
        assert summary["total_requests"] == 1
        assert "gpt-4.1" in summary["by_model"]
