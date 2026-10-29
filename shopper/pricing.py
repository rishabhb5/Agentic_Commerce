"""Token prices, so scripts and `--verbose` can show what each reply cost.

Prices are US dollars per 1M tokens (Anthropic API list prices). Cache writes
cost 1.25x the input price; cache reads are listed per model.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MODEL = "claude-sonnet-5-5"

# model: (input, output, cache read) $ per 1M tokens
PRICES: dict[str, tuple[float, float, float]] = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    "fake": (0.0, 0.0, 0.0),  # --fake-llm
}
CACHE_WRITE_MULTIPLIER = 1.25


@dataclass
class Usage:
    input_tokens: int = 0  # uncached input
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    calls: int = 0

    def add(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read + other.cache_read,
            self.cache_write + other.cache_write,
            self.calls + other.calls,
        )

    def cost(self, model: str) -> float | None:
        prices = PRICES.get(model)
        if prices is None:
            return None
        input_price, output_price, read_price = prices
        return (
            self.input_tokens * input_price
            + self.cache_write * input_price * CACHE_WRITE_MULTIPLIER
            + self.cache_read * read_price
            + self.output_tokens * output_price
        ) / 1_000_000

    def describe(self, model: str) -> str:
        cost = self.cost(model)
        price = f"${cost:.4f} (~{cost * 100:.2f}¢)" if cost is not None else "unknown price"
        cache = (
            f", cache read {self.cache_read:,} / write {self.cache_write:,}"
            if self.cache_read or self.cache_write
            else ""
        )
        calls = f"{self.calls} LLM call{'s' if self.calls != 1 else ''}, " if self.calls else ""
        return (
            f"{calls}{self.input_tokens:,} input + {self.output_tokens:,} output tokens"
            f"{cache} → {price}"
        )


def usage_from_anthropic(usage) -> Usage:
    """From an `anthropic` SDK response's `.usage`."""
    return Usage(
        input_tokens=usage.input_tokens or 0,
        output_tokens=usage.output_tokens or 0,
        cache_read=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_write=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        calls=1,
    )


def usage_from_langchain(message) -> Usage:
    """From a LangChain AIMessage's `.usage_metadata` (zeros if missing)."""
    meta = getattr(message, "usage_metadata", None) or {}
    details = meta.get("input_token_details") or {}
    cache_read = details.get("cache_read") or 0
    cache_write = details.get("cache_creation") or 0
    # LangChain's input_tokens already includes cached tokens; split them out.
    uncached = max((meta.get("input_tokens") or 0) - cache_read - cache_write, 0)
    return Usage(
        input_tokens=uncached,
        output_tokens=meta.get("output_tokens") or 0,
        cache_read=cache_read,
        cache_write=cache_write,
        calls=1,
    )
