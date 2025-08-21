from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import tiktoken

@dataclass(frozen=True)
class TokenCoster:
    # Constants for GPT-5 pricing
    PRICE_PER_1K_INPUT = 0.00125
    PRICE_PER_1K_OUTPUT = 0.01000

@dataclass(frozen=True)
class PricingHelper:
    input_per_1k: Optional[float] = None
    output_per_1k: Optional[float] = None

    def estimate(self, input_tokens: int, output_tokens: int) -> Optional[float]:
        if self.input_per_1k is None or self.output_per_1k is None:
            return None

        return (
            (input_tokens / 1000.0) * self.input_per_1k
            + (output_tokens / 1000.0) * self.output_per_1k
        )

    def num_tokens_from_string(self, string: str, model_name: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.encoding_for_model(model_name)
        num_tokens = len(encoding.encode(string))

        return num_tokens