from __future__ import annotations
import json
from typing import Tuple, Callable, Dict, Any
from pathlib import Path
from utils.prompt_builder import PromptBuilder

# ---------------------------------------
# Suite definitions and scoring functions
# ---------------------------------------

def load_tasks_json(path: str) -> Dict[str, Dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    tasks = data["tasks"] if "tasks" in data else data
    return {k: {"question": v["question"], "expected": v["expected"]} for k, v in tasks.items()}

def score_text(reference: str, model_text: str) -> int:
    try:
        model_dict: Dict[str, str] = json.loads(model_text.strip().lower())
        if "result" not in model_dict:
            return 0
        pred = model_dict["result"]
        return 1 if str(pred) == str(reference).strip() else 0
    except Exception:
        return 0

def example_math_suite(math_suite_path: str) -> Tuple[PromptBuilder, Dict[str, Dict[str, Any]], Callable[[str, str], int]]:
    system = "You are a careful math assistant. Return only the final numeric answer in the JSON format: {\"result\": <answer>}"
    examples = [
        {"user": "What is 2 + 3?", "assistant": "{\"result\": 5}"},
        {"user": "What is 10 + 5?", "assistant": "{\"result\": 15}"},
        {"user": "What is 7 + 4?", "assistant": "{\"result\": 11}"},
        {"user": "What is 2 + 3?", "assistant": "{\"result\": 5}"},
        {"user": "What is 7 + 8?", "assistant": "{\"result\": 15}"},
        {"user": "What is 10 - 4?", "assistant": "{\"result\": 6}"},
        {"user": "What is 3 * 3?", "assistant": "{\"result\": 9}"},
        {"user": "If you have 5 apples and eat 2, how many remain?", "assistant": "{\"result\": 3}"},
        {"user": "10 - 7 + 2 = ?", "assistant": "{\"result\": 5}"},
        {"user": "Double 6 then subtract 4 =", "assistant": "{\"result\": 8}"},
        {"user": "3 boxes of 4 cookies each. Total?", "assistant": "{\"result\": 12}"},
        {"user": "15 divided by 3 + 1 =", "assistant": "{\"result\": 6}"},
        {"user": "8 + 9 - 5 =", "assistant": "{\"result\": 12}"},
        {"user": "9 x 3 - 10 =", "assistant": "{\"result\": 17}"},
        {"user": "24 divided by 6, then add 5 =", "assistant": "{\"result\": 9}"},
        {"user": "If a train travels 10 km then 6 km, total km?", "assistant": "{\"result\": 16}"},
        {"user": "7 more than 2 dozen is?", "assistant": "{\"result\": 31}"},
    ]

    tasks = load_tasks_json(math_suite_path)
    builder = PromptBuilder(system_prompt=system, examples=examples, task_template="Question: {question}\nAnswer:")
    return builder, tasks, score_text


def example_sentiment_suite(sentiment_suite_path: str) -> Tuple[PromptBuilder, Dict[str, Dict[str, Any]], Callable[[str, str], int]]:
    system = "You are a sentiment classifier. Output exactly one of the following words (positive, neutral, or negative) in the JSON format: {\"result\": <word>}"
    examples = [
        {"user": "I love this product", "assistant": "positive"},
        {"user": "It's okay, I guess", "assistant": "neutral"},
        {"user": "This is terrible", "assistant": "negative"},
    ]
    tasks = {
        "s1": {"question": "I love this product", "expected": "positive"},
        "s2": {"question": "It's okay, I guess", "expected": "neutral"},
        "s3": {"question": "This is terrible", "expected": "negative"},
    }
    builder = PromptBuilder(system_prompt=system, examples=examples, task_template="{question}")
    return builder, tasks, score_text
