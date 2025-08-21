from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class TrialConfig:
    model: str
    temperature: float
    top_p: float
    max_output_tokens: int
    reasoning_effort: str
    shots: int
    service_tier: Optional[str] = None # e.g., "auto"
    tool_choice: Optional[str] = None # e.g., "none"


@dataclass
class TrialResult:
    run_id: str
    suite: str
    task_id: str
    model: str
    temperature: float
    top_p: float
    shots: int
    cap: int
    latency_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_tokens: int
    reasoning_tokens: int
    price_input_usd: float
    price_output_usd: float
    price_reasoning_usd: float
    total_price_usd: float
    response_id: str
    response_text: str
    correct: int  # 1/0
    created_at: float
    code_rev: str
    prompt_fingerprint: str