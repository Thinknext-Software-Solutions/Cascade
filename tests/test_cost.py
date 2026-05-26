"""Tests for cascade.cost (pricing, estimator, tracker, formatting)."""

from __future__ import annotations

import pytest

from cascade.cost import (
    PRICING,
    CallRecord,
    CostTracker,
    ModelPricing,
    compute_cost,
    estimate_cost,
    estimate_tokens,
    format_cost,
    get_pricing,
)


# --------- Pricing table ----------


class TestPricing:
    def test_known_anthropic_models_present(self):
        assert ("anthropic", "claude-opus-4-7") in PRICING
        assert ("anthropic", "claude-sonnet-4-6") in PRICING
        assert ("anthropic", "claude-haiku-4-5") in PRICING

    def test_known_openai_models_present(self):
        assert ("openai", "gpt-5") in PRICING
        assert ("openai", "gpt-4o-mini") in PRICING

    def test_known_google_models_present(self):
        assert ("google", "gemini-2.0-flash") in PRICING

    def test_self_hosted_zero_cost(self):
        assert PRICING[("claude_code", "claude-opus-4-7")].input_per_million == 0
        assert PRICING[("ollama", "llama3.1")].input_per_million == 0
        assert PRICING[("ollama", "llama3.1:70b")].output_per_million == 0


class TestGetPricing:
    def test_exact_match(self):
        p = get_pricing("anthropic", "claude-opus-4-7")
        assert p.input_per_million == 15.00
        assert p.output_per_million == 75.00

    def test_case_insensitive_provider(self):
        p = get_pricing("Anthropic", "claude-opus-4-7")
        assert p.input_per_million == 15.00

    def test_unknown_model_known_provider_falls_back(self):
        p = get_pricing("anthropic", "claude-99-omega")
        # Falls back to provider default (Opus pricing) with a note
        assert p.input_per_million == 15.00
        assert "unknown model" in p.notes.lower()

    def test_unknown_provider_returns_zero_pricing(self):
        p = get_pricing("nonexistent", "any-model")
        assert p.input_per_million == 0
        assert p.output_per_million == 0
        assert "unknown provider" in p.notes.lower()


# --------- compute_cost ----------


class TestComputeCost:
    def test_anthropic_opus_pricing(self):
        # 1M input + 1M output at Opus prices = 15 + 75 = $90
        cost = compute_cost(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            provider="anthropic",
            model="claude-opus-4-7",
        )
        assert cost == pytest.approx(90.00, rel=1e-3)

    def test_small_request_cheap(self):
        # 100 input + 200 output at Haiku
        cost = compute_cost(
            input_tokens=100,
            output_tokens=200,
            provider="anthropic",
            model="claude-haiku-4-5",
        )
        # (100/1M * 0.80) + (200/1M * 4.00) = 0.00008 + 0.0008 = ~$0.00088
        assert cost > 0
        assert cost < 0.01

    def test_zero_tokens_costs_zero(self):
        cost = compute_cost(
            input_tokens=0,
            output_tokens=0,
            provider="anthropic",
            model="claude-opus-4-7",
        )
        assert cost == 0

    def test_self_hosted_always_free(self):
        cost = compute_cost(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            provider="ollama",
            model="llama3.1:70b",
        )
        assert cost == 0


# --------- estimate_tokens ----------


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_short_string_at_least_one(self):
        assert estimate_tokens("hi") >= 1

    def test_longer_string_scales(self):
        short = estimate_tokens("hello world")
        long_text = estimate_tokens("hello world " * 100)
        assert long_text > short
        # Should scale roughly linearly
        assert long_text > 50

    def test_falls_back_without_tiktoken(self, monkeypatch):
        # Force the tiktoken path off
        import cascade.cost as cost_module

        monkeypatch.setattr(cost_module, "_TIKTOKEN_ENCODING", False)
        # ~80 chars / 3.5 chars per token = ~23
        count = estimate_tokens("x" * 80)
        assert 15 <= count <= 30


# --------- estimate_cost ----------


class TestEstimateCost:
    def test_returns_triple(self):
        cost, in_tokens, out_tokens = estimate_cost(
            input_text="some prompt text",
            expected_output_tokens=500,
            provider="anthropic",
            model="claude-haiku-4-5",
        )
        assert isinstance(cost, float)
        assert in_tokens > 0
        assert out_tokens == 500

    def test_higher_expected_output_costs_more(self):
        cost_low, _, _ = estimate_cost(
            input_text="prompt",
            expected_output_tokens=100,
            provider="anthropic",
            model="claude-opus-4-7",
        )
        cost_high, _, _ = estimate_cost(
            input_text="prompt",
            expected_output_tokens=10_000,
            provider="anthropic",
            model="claude-opus-4-7",
        )
        assert cost_high > cost_low


# --------- CostTracker ----------


class TestCostTracker:
    def test_empty_tracker(self):
        t = CostTracker()
        assert t.call_count == 0
        assert t.total_cost_usd == 0
        assert t.total_input_tokens == 0
        assert "no LLM calls" in t.summary_line()

    def test_accumulates_calls(self):
        t = CostTracker()
        t.add_call(
            stage="extract",
            provider="anthropic",
            model="claude-haiku-4-5",
            input_tokens=1000,
            output_tokens=500,
        )
        t.add_call(
            stage="plan",
            provider="anthropic",
            model="claude-opus-4-7",
            input_tokens=2000,
            output_tokens=1000,
        )
        assert t.call_count == 2
        assert t.total_input_tokens == 3000
        assert t.total_output_tokens == 1500
        assert t.total_cost_usd > 0

    def test_summary_line_includes_totals(self):
        t = CostTracker()
        t.add_call(
            stage="x",
            provider="anthropic",
            model="claude-opus-4-7",
            input_tokens=10_000,
            output_tokens=5_000,
        )
        s = t.summary_line()
        assert "1 LLM call" in s
        assert "10,000" in s
        assert "5,000" in s
        assert "$" in s


# --------- format_cost ----------


class TestFormatCost:
    def test_zero(self):
        assert format_cost(0) == "free"

    def test_tiny_amount(self):
        assert format_cost(0.001) == "<$0.01"

    def test_normal_amount(self):
        assert format_cost(0.42) == "$0.42"
        assert format_cost(1.99) == "$1.99"
        assert format_cost(9.99) == "$9.99"

    def test_large_amount(self):
        assert format_cost(10.5) == "$10.5"
        assert format_cost(150.0) == "$150.0"
