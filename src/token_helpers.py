from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class Pricing:
    input_per_1k: Optional[float] = None
    output_per_1k: Optional[float] = None

    def estimate(self, input_tokens: int, output_tokens: int) -> Optional[float]:
        if self.input_per_1k is None or self.output_per_1k is None:
            return None
        return (
            (input_tokens / 1000.0) * self.input_per_1k
            + (output_tokens / 1000.0) * self.output_per_1k
        )