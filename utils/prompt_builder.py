from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Sequence, Optional
import hashlib
import random

"""
PromptBuilder: deterministic few-shot selection + message formatting

Usage:
    pb = PromptBuilder(
        system_prompt="You are a helpful assistant.",
        examples=[{"user": "...", "assistant": "..."}, ...],
        task_template="Question: {question}\nAnswer concisely."
    )
    messages = pb.build_messages(
        shots=6,
        task={"question": "2 + 3 = ?"},
        example_order_seed=7
    )
"""

RoleMsg = Dict[str, Any]
Example = Dict[str, str]

@dataclass(frozen=True)
class PromptBuilder:
    system_prompt: str
    examples: Sequence[Example]
    task_template: str
    user_prefix: str = ""
    assistant_prefix: str = ""

    def _render_task(self, task: Dict[str, Any]) -> str:
        return self.task_template.format(**task)

    def _select_examples(
        self,
        shots: int,
        seed: Optional[int]
    ) -> Sequence[Example]:

        if shots <= 0 or not self.examples:
            return ()

        # deterministic but shuffle-capable selection
        idxs = list(range(len(self.examples)))
        if seed is not None:
            rnd = random.Random(seed)
            rnd.shuffle(idxs)

        # take first N after possible shuffle
        idxs = idxs[:shots]

        return tuple(self.examples[i] for i in idxs)

    def build_messages(
        self,
        shots: int,
        task: Dict[str, Any],
        example_order_seed: Optional[int] = None
    ) -> List[RoleMsg]:

        msgs: List[RoleMsg] = [
            {"role": "system", "content": self.system_prompt}
        ]

        for ex in self._select_examples(shots, example_order_seed):
            if "user" in ex:
                msgs.append({
                    "role": "user",
                    "content": [{
                        "type": "input_text",
                        "text": f"{self.user_prefix}{ex['user']}"
                    }]
                })
            if "assistant" in ex:
                msgs.append({
                    "role": "assistant",
                    "content": [{
                        "type": "output_text",
                        "text": f"{self.assistant_prefix}{ex['assistant']}"
                    }]
                })

        msgs.append({"role": "user", "content": self._render_task(task)})

        return msgs

    def fingerprint(self) -> str:

        # Records prompt identity with the results for reproducibility
        h = hashlib.sha256()
        h.update(self.system_prompt.encode())

        for ex in self.examples:
            h.update((ex.get("user","") + "→" + ex.get("assistant","")).encode())

        h.update(self.task_template.encode())

        return h.hexdigest()[:12]