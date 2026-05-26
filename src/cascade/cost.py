"""Cost estimation and tracking for LLM operations.

Three pieces:
- A pricing table that maps (provider, model) to cost-per-million-tokens
- A token estimator that uses tiktoken when available, falls back to a
  character-based approximation otherwise
- A CostTracker that accumulates calls across a CLI invocation and
  produces session totals

Pricing is in USD per 1M tokens (input and output separately). Self-
hosted providers (ollama, claude_code) report $0.00 because the cost is
already covered by the user's local resources or subscription.

Pricing data is a snapshot; LLM provider pricing changes. The PRICING
dict is the single place to update when prices move.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional


# ----------------------------------------------------------------------------
# Pricing table
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelPricing:
    """Cost per 1M tokens for one model."""

    input_per_million: float  # USD per 1,000,000 input tokens
    output_per_million: float  # USD per 1,000,000 output tokens
    notes: str = ""  # human-readable notes (e.g., "covered by Claude Code subscription")


# Pricing snapshot as of 2026-Q2. Keep this list updated as providers
# change prices. When a model isn't listed, we fall back to a "best
# guess" pricing for the provider (see _provider_default below).
PRICING: dict[tuple[str, str], ModelPricing] = {
    # ---- Anthropic ----
    ("anthropic", "claude-opus-4-7"): ModelPricing(input_per_million=15.00, output_per_million=75.00),
    ("anthropic", "claude-sonnet-4-6"): ModelPricing(input_per_million=3.00, output_per_million=15.00),
    ("anthropic", "claude-haiku-4-5"): ModelPricing(input_per_million=0.80, output_per_million=4.00),
    # Legacy Claude 3.x for users on older models
    ("anthropic", "claude-3-5-sonnet"): ModelPricing(input_per_million=3.00, output_per_million=15.00),
    ("anthropic", "claude-3-5-haiku"): ModelPricing(input_per_million=0.80, output_per_million=4.00),

    # ---- OpenAI ----
    ("openai", "gpt-5"): ModelPricing(input_per_million=10.00, output_per_million=30.00),
    ("openai", "gpt-5-mini"): ModelPricing(input_per_million=0.50, output_per_million=1.50),
    ("openai", "gpt-4o"): ModelPricing(input_per_million=2.50, output_per_million=10.00),
    ("openai", "gpt-4o-mini"): ModelPricing(input_per_million=0.15, output_per_million=0.60),

    # ---- Google ----
    ("google", "gemini-2.0-flash"): ModelPricing(input_per_million=0.10, output_per_million=0.40),
    ("google", "gemini-2.0-pro"): ModelPricing(input_per_million=1.25, output_per_million=5.00),
    ("google", "gemini-1.5-flash"): ModelPricing(input_per_million=0.075, output_per_million=0.30),

    # ---- Free / self-hosted ----
    # Claude Code uses the user's existing subscription; no per-call cost
    ("claude_code", "claude-opus-4-7"): ModelPricing(
        input_per_million=0.0,
        output_per_million=0.0,
        notes="covered by Claude Code subscription",
    ),
    ("claude_code", "claude-sonnet-4-6"): ModelPricing(
        input_per_million=0.0,
        output_per_million=0.0,
        notes="covered by Claude Code subscription",
    ),
    # Ollama and other self-hosted runners
    ("ollama", "llama3.1"): ModelPricing(
        input_per_million=0.0, output_per_million=0.0, notes="self-hosted (local)"
    ),
    ("ollama", "llama3.1:70b"): ModelPricing(
        input_per_million=0.0, output_per_million=0.0, notes="self-hosted (local)"
    ),
    ("ollama", "qwen2.5"): ModelPricing(
        input_per_million=0.0, output_per_million=0.0, notes="self-hosted (local)"
    ),
}


# Fallback per-provider pricing for unknown models. Picked conservatively
# (higher than real prices) so users aren't surprised by undercount.
_PROVIDER_FALLBACK: dict[str, ModelPricing] = {
    "anthropic": ModelPricing(
        input_per_million=15.00,
        output_per_million=75.00,
        notes="unknown model; using Opus pricing as upper bound",
    ),
    "openai": ModelPricing(
        input_per_million=10.00,
        output_per_million=30.00,
        notes="unknown model; using GPT-5 pricing as upper bound",
    ),
    "google": ModelPricing(
        input_per_million=1.25,
        output_per_million=5.00,
        notes="unknown model; using Gemini Pro pricing as upper bound",
    ),
    "claude_code": ModelPricing(
        input_per_million=0.0, output_per_million=0.0,
        notes="covered by Claude Code subscription",
    ),
    "ollama": ModelPricing(
        input_per_million=0.0, output_per_million=0.0,
        notes="self-hosted (local)",
    ),
}


def get_pricing(provider: str, model: str) -> ModelPricing:
    """Look up pricing for a specific provider+model pair.

    Falls back to a conservative per-provider default if the exact model
    isn't in the table. Unknown providers return zero-cost pricing with
    a note (so cost reporting still works without raising).
    """
    key = (provider.lower(), model)
    if key in PRICING:
        return PRICING[key]
    # Provider known, model unknown
    if provider.lower() in _PROVIDER_FALLBACK:
        return _PROVIDER_FALLBACK[provider.lower()]
    # Both unknown
    return ModelPricing(
        input_per_million=0.0,
        output_per_million=0.0,
        notes=f"unknown provider '{provider}'; cost reported as $0",
    )


def compute_cost(
    *, input_tokens: int, output_tokens: int, provider: str, model: str
) -> float:
    """Compute the USD cost of one LLM call.

    Returns a float dollar amount. May be 0.0 for self-hosted providers.
    """
    pricing = get_pricing(provider, model)
    in_cost = (input_tokens / 1_000_000) * pricing.input_per_million
    out_cost = (output_tokens / 1_000_000) * pricing.output_per_million
    return in_cost + out_cost


# ----------------------------------------------------------------------------
# Token estimation (used for pre-call estimates before the LLM sees the prompt)
# ----------------------------------------------------------------------------


_TIKTOKEN_ENCODING = None


def _get_tiktoken_encoding():
    """Lazily load tiktoken if available. Returns None if not installed."""
    global _TIKTOKEN_ENCODING
    if _TIKTOKEN_ENCODING is False:  # tried and failed previously
        return None
    if _TIKTOKEN_ENCODING is None:
        try:
            import tiktoken  # type: ignore
            _TIKTOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
        except (ImportError, Exception):
            _TIKTOKEN_ENCODING = False
            return None
    return _TIKTOKEN_ENCODING


def estimate_tokens(text: str) -> int:
    """Estimate the token count of a string.

    Uses tiktoken's cl100k_base encoding when available (used by GPT-4 +
    GPT-5; close to Anthropic's tokenization for English text). Falls
    back to a chars-per-token heuristic otherwise.

    For English text, ~4 characters per token. Code and structured text
    tokenize denser; we use 3.5 to slightly over-estimate (safer).
    """
    if not text:
        return 0
    encoding = _get_tiktoken_encoding()
    if encoding is not None:
        return len(encoding.encode(text))
    # Heuristic fallback: ~3.5 chars per token
    return max(1, round(len(text) / 3.5))


def estimate_cost(
    *,
    input_text: str,
    expected_output_tokens: int,
    provider: str,
    model: str,
) -> tuple[float, int, int]:
    """Estimate the cost of an LLM call before making it.

    Args:
        input_text: The full prompt text (system + user).
        expected_output_tokens: How many output tokens you expect.
            Cascade typically passes max_output_tokens here as the
            upper bound.
        provider: LLM provider name.
        model: Model identifier.

    Returns:
        (estimated_cost_usd, estimated_input_tokens, expected_output_tokens)
    """
    input_tokens = estimate_tokens(input_text)
    cost = compute_cost(
        input_tokens=input_tokens,
        output_tokens=expected_output_tokens,
        provider=provider,
        model=model,
    )
    return cost, input_tokens, expected_output_tokens


# ----------------------------------------------------------------------------
# CostTracker -- session-level accumulator
# ----------------------------------------------------------------------------


@dataclass
class CallRecord:
    """One LLM call's cost contribution."""

    stage: str  # e.g., "extract", "plan", "code"
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class CostTracker:
    """Accumulates LLM-call costs across a CLI invocation.

    Thread-safe (the pipeline calls multiple LLM stages sequentially today,
    but locking lets future async/parallel work share a single tracker).
    """

    records: list[CallRecord] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_call(
        self,
        *,
        stage: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> CallRecord:
        """Record one LLM call. Returns the CallRecord for caller printing."""
        cost = compute_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            provider=provider,
            model=model,
        )
        record = CallRecord(
            stage=stage,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
        with self._lock:
            self.records.append(record)
        return record

    @property
    def total_cost_usd(self) -> float:
        with self._lock:
            return sum(r.cost_usd for r in self.records)

    @property
    def total_input_tokens(self) -> int:
        with self._lock:
            return sum(r.input_tokens for r in self.records)

    @property
    def total_output_tokens(self) -> int:
        with self._lock:
            return sum(r.output_tokens for r in self.records)

    @property
    def call_count(self) -> int:
        with self._lock:
            return len(self.records)

    def summary_line(self) -> str:
        """One-line summary suitable for end-of-command output."""
        if self.call_count == 0:
            return "(no LLM calls in this run)"
        return (
            f"{self.call_count} LLM call(s), "
            f"{self.total_input_tokens:,} in / {self.total_output_tokens:,} out tokens, "
            f"{format_cost(self.total_cost_usd)}"
        )


# ----------------------------------------------------------------------------
# Display helpers
# ----------------------------------------------------------------------------


def format_cost(amount_usd: float) -> str:
    """Render a USD amount for display.

    < $0.01 shows as "<$0.01" (so users don't see "$0.00" for tiny calls).
    >= $0.01 shows as "$X.XX" (cents precision).
    >= $10 shows as "$XX.X" (one decimal; that level of precision is enough).
    """
    if amount_usd == 0:
        return "free"
    if amount_usd < 0.01:
        return "<$0.01"
    if amount_usd < 10:
        return f"${amount_usd:.2f}"
    return f"${amount_usd:.1f}"
