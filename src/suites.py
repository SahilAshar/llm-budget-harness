from __future__ import annotations
import json
from typing import Tuple, Callable, Dict, Any
from utils.prompt_builder import PromptBuilder

# ---------------------------------------
# Suite definitions and scoring functions
# ---------------------------------------

def score_math(reference: str, model_text: str) -> int:
    # Extract first integer from model_text and compare with expected integer
    # import re
    # m = re.search(r"-?\d+", model_text.strip())
    # if not m:
    #     return 0
    # try:
    #     pred = int(m.group(0))
    #     return 1 if str(pred) == str(reference).strip() else 0
    # except Exception:
    #     return 0

    model_dict: Dict[str, str] = json.loads(model_text.strip().lower())
    if "result" not in model_dict:
        return 0
    try:
        pred = model_dict["result"]
        return 1 if str(pred) == str(reference).strip() else 0
    except Exception:
        return 0


def score_sentiment(reference: str, model_text: str) -> int:
    # norm = model_text.strip().lower()
    # # common aliases
    # mapping = {
    #     "pos": "positive", "neg": "negative", "neu": "neutral",
    #     "positive": "positive", "negative": "negative", "neutral": "neutral"
    # }
    # for k, v in mapping.items():
    #     if norm.startswith(k):
    #         norm = v
    #         break
    # return 1 if norm == reference.strip().lower() else 0

    model_dict: Dict[str, str] = json.loads(model_text.strip().lower())
    if "result" not in model_dict:
        return 0
    try:
        pred = model_dict["result"]
        return 1 if str(pred) == str(reference).strip() else 0
    except Exception:
        return 0

def example_math_suite() -> Tuple[PromptBuilder, Dict[str, Dict[str, Any]], Callable[[str, str], int]]:
    system = "You are a careful math assistant. Answer with the final integer in the JSON format: {\"result\": <integer>}"
    examples = [
        {"user": "What is 2 + 3?", "assistant": "5"},
        {"user": "What is 10 + 5?", "assistant": "15"},
        {"user": "What is 7 + 4?", "assistant": "11"},
    ]
    tasks = {
        "m1": {"question": "2 + 3 = ?", "expected": "5"},
        "m2": {"question": "10 + 5 = ?", "expected": "15"},
        "m3": {"question": "7 + 4 = ?", "expected": "11"},
    }
    builder = PromptBuilder(system_prompt=system, examples=examples, task_template="Question: {question}\nAnswer:")
    return builder, tasks, score_math


def example_sentiment_suite() -> Tuple[PromptBuilder, Dict[str, Dict[str, Any]], Callable[[str, str], int]]:
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
    return builder, tasks, score_sentiment
