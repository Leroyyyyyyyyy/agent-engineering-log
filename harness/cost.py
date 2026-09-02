"""
Per-run cost accounting.

Kept separate from tracing.py because a price table goes stale on a schedule
nobody controls, and mixing it into instrumentation means every price change
touches the file that emits spans.

Prices are USD per million tokens. Verify against the vendor page before
quoting a number - the shape of the calculation is the durable part, the
numbers are not.
"""

from __future__ import annotations

from dataclasses import dataclass

# Four rates per model, not two. Cache writes cost MORE than plain input
# (the provider has to store it); cache reads cost about a tenth. Collapsing
# these into one input rate is wrong in both directions, and the error moves
# with the cache hit rate - so it cannot be corrected with a fudge factor.
PRICES = {
    "claude-haiku-4-5-20251001": {
        "input": 1.00,
        "output": 5.00,
        "cache_write": 1.25,
        "cache_read": 0.10,
    },
    "claude-sonnet-5": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    "claude-opus-5": {
        "input": 15.00,
        "output": 75.00,
        "cache_write": 18.75,
        "cache_read": 1.50,
    },
}


@dataclass
class Cost:
    """What one run consumed, per step and in total."""

    model: str
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0

    def add(self, usage_event: dict) -> None:
        """Fold one usage event in."""
        self.steps += 1
        self.input_tokens += usage_event.get("input_tokens") or 0
        self.output_tokens += usage_event.get("output_tokens") or 0
        self.cache_write_tokens += usage_event.get("cache_creation_input_tokens") or 0
        self.cache_read_tokens += usage_event.get("cache_read_input_tokens") or 0

    @property
    def known_model(self) -> bool:
        return self.model in PRICES

    def usd(self) -> float | None:
        """Total cost, or None when the model has no price on file."""
        rates = PRICES.get(self.model)
        if rates is None:
            return None
        return (
            self.input_tokens * rates["input"]
            + self.output_tokens * rates["output"]
            + self.cache_write_tokens * rates["cache_write"]
            + self.cache_read_tokens * rates["cache_read"]
        ) / 1_000_000

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "steps": self.steps,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "usd": self.usd(),
        }


def track(events, model: str):
    """
    Pass every event through, accumulating cost. Yields (event, cost).

    The cost object is the SAME object every time, mutated in place, so the
    caller can read the running total mid-stream and the final total after the
    loop - without the generator having to signal which event was the last one.
    """
    cost = Cost(model=model)
    for event in events:
        if event["type"] == "usage":
            cost.add(event)
        yield event, cost
