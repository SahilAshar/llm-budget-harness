from __future__ import annotations
import os
import json

from typing import List, Dict, Any, Tuple, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "minimal")

class OpenAIAdapter:
    """
    Thin wrapper around the OpenAI client so we can:
      - pass messages (system/user/assistant)
      - set max_output_tokens reliably
      - read normalized usage & text
    """
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        top_p: float,
        max_output_tokens: int,
        reasoning_effort: str,
        service_tier: Optional[str] = None,
        tool_choice: Optional[str] = None
    ) -> Tuple[Dict[str, Any], str, Dict[str, int]]:
        """
        Returns: (raw_response_dict, response_text, usage_dict)
        usage_dict: {"input_tokens": int, "output_tokens": int, "total_tokens": int, "cached_tokens": int, reasoning_tokens: int}
        """
        resp = self.client.responses.create(
            model=model,
            input=[
                {
                    "role": m["role"],
                    "content": m["content"]
                } for m in messages
            ], # type: ignore
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            reasoning={"effort": reasoning_effort},
            service_tier=service_tier or "auto",
            tool_choice=tool_choice or "none",
        )
        raw: Dict[str, Any] = json.loads(resp.model_dump_json())  # dataclass -> dict
        # Text extraction
        text = ""
        if resp.output:
            for item in resp.output:
                if getattr(item, "type", None) == "message" and getattr(item, "content", None):
                    # concatenate text parts
                    for c in item.content:
                        if getattr(c, "type", None) == "output_text":
                            text += c.text
        # Usage extraction
        u: Any = resp.usage
        usage: Dict[str, int] = {
            "input_tokens": int(u.input_tokens or 0),
            "output_tokens": int(u.output_tokens or 0),
            "total_tokens": int(u.total_tokens or 0),
            "cached_tokens": int(getattr(u.input_tokens_details or {}, "cached_tokens", 0) or 0),
            "reasoning_tokens": int(getattr(u.output_tokens_details or {}, "reasoning_tokens", 0) or 0)
        }
        return raw, text, usage
