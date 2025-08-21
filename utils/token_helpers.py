from __future__ import annotations
from dataclasses import dataclass
from typing import List
import tiktoken

@dataclass(frozen=True)
class TokenCoster:
    # Constants for GPT-5 pricing
    PRICE_PER_1K_INPUT = 0.00125
    PRICE_PER_1K_OUTPUT = 0.01000

@dataclass(frozen=True)
class PricingHelper:
    input_per_1k: float
    output_per_1k: float

    def price_input_tokens_usd(self, input_tokens: int) -> float:
        return (input_tokens / 1000.0) * self.input_per_1k

    def price_output_tokens_usd(self, output_tokens: int) -> float:
        return (output_tokens / 1000.0) * self.output_per_1k

    def num_tokens_from_string(self, string: str, model_name: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.encoding_for_model(model_name)
        num_tokens = len(encoding.encode(string))

        return num_tokens

def pctl(xs: List[float], p: float) -> float:
    if not xs: return 0.0
    xs2 = sorted(xs)
    k = max(0, min(len(xs2) - 1, int(round((p/100.0) * (len(xs2)-1)))))
    return xs2[k]