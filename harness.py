import os
import time
import json
import statistics
import tiktoken
import csv
import hashlib
from pathlib import Path

from typing import Callable, List, Dict, Any
from time import perf_counter
from pprint import pprint
from dotenv import load_dotenv
from openai import OpenAI

# Env Vars
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "minimal")
VERBOSE = os.getenv("VERBOSE", "0") == "1"
RUN_DIR = Path(os.getenv("RUN_DIR", "runs"))
RUN_DIR.mkdir(parents=True, exist_ok=True)

client = OpenAI(api_key=OPENAI_API_KEY)

# --------- CONSTANTS ---------
BUDGETS: List[Dict[str, Any]] = [
    {"name": "cheap", "shots": 0, "max_output_tokens": 32, "sweep": "legacy", "level": None},
    {"name": "balanced", "shots": 2, "max_output_tokens": 64, "sweep": "legacy", "level": None},
    {"name": "deluxe", "shots": 6, "max_output_tokens": 128, "sweep": "legacy", "level": None},
]

SHOT_LEVELS: List[int] = [0, 2, 6, 8, 10, 12, 14]
CAP_LEVELS:  List[int] = [32, 64, 128, 256, 512]

PRICE_PER_1K_INPUT = 0.00125
PRICE_PER_1K_OUTPUT = 0.01000

# --------- LLM CALLER ---------
def call_llm(prompt: str, max_output_tokens: int) -> Dict[str, Any]:

    t0 = perf_counter()

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
        max_output_tokens=max_output_tokens,
        reasoning={"effort": REASONING_EFFORT},
        tool_choice="none",
        parallel_tool_calls=False,
        store=False
    )

    t1 = perf_counter()

    if VERBOSE:
        pprint(response)

    text = (response.output_text or "").strip()
    usage: Any = getattr(response, "usage", None) or {}
    in_tok  = int(getattr(usage, "input_tokens", 0) or 0)
    out_tok = int(getattr(usage, "output_tokens", 0) or 0)

    details_in  = getattr(usage, "input_tokens_details", None)
    cached_tok  = int(getattr(details_in, "cached_tokens", 0) or 0)
    details_out = getattr(usage, "output_tokens_details", None)
    reason_tok  = int(getattr(details_out, "reasoning_tokens", 0) or 0)

    if VERBOSE:
        print(f"Input tokens: {in_tok} (cached {cached_tok}) | Output tokens: {out_tok} | Reasoning: {reason_tok}")

    if in_tok == 0:
        in_tok = num_tokens_from_string(prompt, OPENAI_MODEL + "-")
    if out_tok == 0 and text:
        out_tok = num_tokens_from_string(text, OPENAI_MODEL + "-")

    cost = (in_tok / 1000.0) * PRICE_PER_1K_INPUT + (out_tok / 1000.0) * PRICE_PER_1K_OUTPUT

    return {
        "response_id": response,
        "text": text,
        "input_tokens": in_tok,
        "cached_input_tokens": cached_tok,
        "output_tokens": out_tok,
        "reasoning_tokens": reason_tok,
        "latency_ms": (t1 - t0) * 1000.0,
        "cost": cost,
    }

# --------- TOKENIZER ---------
def num_tokens_from_string(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# --------- PROMPT BUILDERS ---------
FEW_SHOTS_MATH = [
    ("If you have 5 apples and eat 2, how many remain?", "3"),
    ("10 - 7 + 2 = ?", "5"),
    ("Double 6 then subtract 4 =", "8"),
    ("3 boxes of 4 cookies each. Total?", "12"),
    ("15 divided by 3 + 1 =", "6"),
    ("8 + 9 - 5 =", "12")
]

FEW_SHOTS_SENT = [
    ("I absolutely love this.", "positive"),
    ("It's fine.", "neutral"),
    ("This is awful.", "negative"),
    ("Exceeded my expectations.", "positive"),
    ("Nothing to write home about.", "neutral"),
    ("It keeps crashing.", "negative")
]

FEWSHOT_POOLS: Dict[str, List[Dict[str, str]]] = {
    "math_suite": [
        {"q": "What is 2 + 3?", "a": "5"},
        {"q": "What is 7 + 8?", "a": "15"},
        {"q": "What is 10 - 4?", "a": "6"},
        {"q": "What is 3 * 3?", "a": "9"},
        {"q": "If you have 5 apples and eat 2, how many remain?", "a": "3"},
        {"q": "10 - 7 + 2 = ?", "a": "5"},
        {"q": "Double 6 then subtract 4 =", "a": "8"},
        {"q": "3 boxes of 4 cookies each. Total?", "a": "12"},
        {"q": "15 divided by 3 + 1 =", "a": "6"},
        {"q": "8 + 9 - 5 =", "a": "12"},
        {"q": "9 x 3 - 10 =", "a": "17"},
        {"q": "24 divided by 6, then add 5 =", "a": "9"},
        {"q": "If a train travels 10 km then 6 km, total km?", "a": "16"},
        {"q": "7 more than 2 dozen is?", "a": "31"},
    ],
    "sentiment_suite": [
        {"q": "I loved this product!", "a": "positive"},
        {"q": "It was okay, nothing special.", "a": "neutral"},
        {"q": "This was terrible.", "a": "negative"},
        {"q": "Amazing experience!", "a": "positive"},
        {"q": "Meh.", "a": "neutral"},
        {"q": "I hated it.", "a": "negative"},
        {"q": "Exceeded my expectations.", "a": "positive"},
        {"q": "Nothing to write home about.", "a": "neutral"},
        {"q": "It keeps crashing.", "a": "negative"},
        {"q": "Best purchase ever!", "a": "positive"},
        {"q": "Not worth the money.", "a": "negative"},
        {"q": "I would buy it again.", "a": "positive"},
        {"q": "It's fine, I guess.", "a": "neutral"},
        {"q": "Terrible customer service.", "a": "negative"}
    ],
}

def build_prompt(
    suite_name: str,
    task: Dict[str, Any],
    shots: int,
    fewshot_pool: List[Dict[str, str]] | None = None,
) -> str:
    """
    Builds a compact, deterministic prompt with N few-shot examples.
    Expects task['question'] and (for eval) task.get('answer').
    """
    pool = fewshot_pool if fewshot_pool is not None else FEWSHOT_POOLS.get(suite_name, [])
    examples = pool[: max(0, shots)]
    header = (
        "You are a precise classifier.\n"
        "- For math, reply with only the integer result.\n"
        "- For sentiment, reply with only one of: positive | neutral | negative.\n"
        "No punctuation. No explanation.\n"
    )
    parts = [header]
    for ex in examples:
        parts.append(f"Q: {ex['q']}\nA: {ex['a']}\n")
    parts.append(f"Q: {task['question']}\nA:")
    return "\n".join(parts)

def build_math_prompt(q: str, shots: int) -> str:
    lines = ["You are a calculator. Return ONLY the final integer. No words."]
    for s in FEW_SHOTS_MATH[:shots]:
        lines.append(f"Q: {s[0]}\nA: {s[1]}")
    lines.append(f"Q: {q}\nA:")
    return "\n".join(lines)

def build_sentiment_prompt(text: str, shots: int) -> str:
    lines = ["You are a sentiment classifier. Return ONLY the sentiment label: positive, negative, or neutral with no punctuation or whitespace."]
    for s in FEW_SHOTS_SENT[:shots]:
        lines.append(f"Text: {s[0]}\nLabel: {s[1]}")
    lines.append(f"Text: {text}\nLabel:")
    return "\n".join(lines)

# ---------- Experiment generators ----------
def build_shot_sweep(fixed_cap: int) -> List[Dict[str, Any]]:
    return [
        {"name": f"sweep_shots_s{shots}_cap{fixed_cap}", "shots": shots, "max_output_tokens": fixed_cap, "sweep": "shots", "level": shots}
        for shots in SHOT_LEVELS
    ]

def build_cap_sweep(fixed_shots: int) -> List[Dict[str, Any]]:
    return [
        {"name": f"sweep_cap_s{fixed_shots}_cap{cap}", "shots": fixed_shots, "max_output_tokens": cap, "sweep": "cap", "level": cap}
        for cap in CAP_LEVELS
    ]

def config_id(model: str, shots: int, cap: int) -> str:
    key = f"{model}|s={shots}|cap={cap}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]

# ---------- Run logging ----------
class RunLogger:
    def __init__(self, run_name: str):
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = RUN_DIR / f"{run_name}-{ts}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.raw_path = self.run_dir / "raw.jsonl"
        self.agg_path = self.run_dir / "aggregate.csv"
        self._raw_f = open(self.raw_path, "a", encoding="utf-8")
        self._agg_rows: List[Dict[str, Any]] = []

    def log_raw(self, row: Dict[str, Any]) -> None:
        self._raw_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._raw_f.flush()

    def add_aggregate_row(self, row: Dict[str, Any]) -> None:
        self._agg_rows.append(row)

    def flush_aggregate(self) -> None:
        if not self._agg_rows:
            return
        fieldnames = list(self._agg_rows[0].keys())
        with open(self.agg_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in self._agg_rows:
                w.writerow(r)

    def close(self) -> None:
        try:
            self._raw_f.close()
        finally:
            self.flush_aggregate()

# --------- EVALS ---------
def eval_math(pred: str, gold: str) -> int:
    import re
    m = re.search(r"-?\d+", pred)
    if not m:
        return 0
    try:
        return int(int(m.group(0)) == int(gold))
    except ValueError:
        return 0

def eval_sentiment(pred: str, gold: str) -> int:
    norm = pred.strip().lower()
    if "positive" in norm: norm = "positive"
    elif "negative" in norm: norm = "negative"
    else: norm = "neutral"
    return int(norm == gold)

def est_cost(inp_tok: int, out_tok: int) -> float:
    return (inp_tok/1000.0)*PRICE_PER_1K_INPUT + (out_tok/1000.0)*PRICE_PER_1K_OUTPUT

def pct(p, xs):
    xs_sorted = sorted(xs)
    k = max(0, min(len(xs_sorted)-1, int(round((p/100.0)*(len(xs_sorted)-1)))))
    return xs_sorted[k]

def build_results_per_budget(
    budget: Dict[str, Any],
    key_label: str,
    value_label: str,
    prompt_func: Callable[[Any, Any], str],
    eval_func: Callable[[str, str], int],
    data_file: Any
) -> Dict[str, Any]:

    results: Dict[str, Any] = {}

    scores, latencies, in_toks, cached_toks, out_toks, reason_toks, costs = [], [], [], [], [], [], []
    for data in data_file:
        prompt = prompt_func(data[key_label], budget["shots"])
        # prompt = build_math_prompt(ex["q"], b["shots"])
        resp = call_llm(prompt, budget["max_output_tokens"])

        scores.append(eval_func(resp["text"], data[value_label]))
        latencies.append(resp["latency_ms"])

        inp = resp.get("input_tokens", num_tokens_from_string(prompt, OPENAI_MODEL + "-"))
        cached = resp.get("cached_input_tokens", 0)
        out = resp.get("output_tokens", num_tokens_from_string(resp["text"], OPENAI_MODEL + "-"))
        reason = resp.get("reasoning_tokens", 0)

        in_toks.append(inp)
        cached_toks.append(cached)
        out_toks.append(out)
        reason_toks.append(reason)
        costs.append(est_cost(inp, out))

    results = {
            "acc": sum(scores)/len(scores),
            "p50_ms": pct(50, latencies),
            "p95_ms": pct(95, latencies),
            "in_tok": statistics.mean(in_toks),
            "cached_toks": statistics.mean(cached_toks),
            "out_tok": statistics.mean(out_toks),
            "reason_toks": statistics.mean(reason_toks),
            "cost": statistics.mean(costs) if any(costs) else None,
            "cost_per_100": (statistics.mean(costs)*100) if any(costs) else None,
        }

    return results

def run_math_suite(budgets: List[Dict[str, Any]], data_file_path: str) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"Data file {data_file_path} does not exist.")

    with open(data_file_path) as f:
        math_data = json.load(f)

    for budget in budgets:
        llm_result_data: Dict[str, Any] = build_results_per_budget(
            budget=budget,
            key_label="q",
            value_label="a",
            prompt_func=build_math_prompt,
            eval_func=eval_math,
            data_file=math_data
        )

        results[budget["name"]] = llm_result_data

    return results

def run_sentiment_suite(budgets: List[Dict[str, Any]], data_file_path: Any) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"Data file {data_file_path} does not exist.")

    with open(data_file_path) as f:
        sentiment_data = json.load(f)

    for budget in budgets:
        llm_result_data: Dict[str, Any] = build_results_per_budget(
            budget=budget,
            key_label="text",
            value_label="label",
            prompt_func=build_sentiment_prompt,
            eval_func=eval_sentiment,
            data_file=sentiment_data
        )

        results[budget["name"]] = llm_result_data

    return results

# --------- RUNNER ---------
def run_suite(
    suite_name: str,
    tasks: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    logger: RunLogger,
    fewshot_pool: List[Dict[str, str]] | None = None,
) -> Dict[str, Any]:
    preds: List[bool] = []
    latencies: List[float] = []
    in_toks: List[int] = []
    out_toks: List[int] = []
    cached_toks: List[int] = []
    reason_toks: List[int] = []
    costs: List[float] = []

    for i, task in enumerate(tasks):
        prompt = build_prompt(suite_name, task, cfg["shots"], fewshot_pool)
        result = call_llm(prompt, cfg["max_output_tokens"])
        pred_text = result["text"]
        gold = task.get("answer")
        correct = (gold is None) or (pred_text.strip().lower() == str(gold).strip().lower())

        row = {
            "ts": time.time(),
            "suite": suite_name,
            "item_id": i,
            "model": OPENAI_MODEL,
            "config_id": config_id(OPENAI_MODEL, cfg["shots"], cfg["max_output_tokens"]),
            "config_name": cfg["name"],
            "sweep": cfg.get("sweep"),
            "level": cfg.get("level"),
            "shots": cfg["shots"],
            "cap": cfg["max_output_tokens"],
            "latency_ms": result["latency_ms"],
            "in_tok": result["input_tokens"],
            "out_tok": result["output_tokens"],
            "cached_tok": result["cached_input_tokens"],
            "reason_tok": result["reasoning_tokens"],
            "cost_usd": result["cost"],
            "prediction": pred_text,
            "expected": gold,
            "correct": int(bool(correct)),
        }
        logger.log_raw(row)

        preds.append(bool(correct))
        latencies.append(result["latency_ms"])
        in_toks.append(result["input_tokens"])
        out_toks.append(result["output_tokens"])
        cached_toks.append(result["cached_input_tokens"])
        reason_toks.append(result["reasoning_tokens"])
        costs.append(result["cost"])

    def pctl(xs: List[float], p: float) -> float:
        if not xs: return 0.0
        xs2 = sorted(xs)
        k = max(0, min(len(xs2) - 1, int(round((p/100.0) * (len(xs2)-1)))))
        return xs2[k]

    summary = {
        "acc": (sum(preds) / max(1, len(preds))),
        "cached_toks": statistics.mean(cached_toks) if cached_toks else 0,
        "cost": sum(costs),
        "cost_per_100": (sum(costs) / max(1, len(preds))) * 100.0,
        "in_tok": statistics.mean(in_toks) if in_toks else 0,
        "out_tok": statistics.mean(out_toks) if out_toks else 0,
        "p50_ms": pctl(latencies, 50),
        "p95_ms": pctl(latencies, 95),
        "reason_toks": statistics.mean(reason_toks) if reason_toks else 0,
    }
    # stash aggregate row for CSV
    logger.add_aggregate_row({
        "suite": suite_name,
        "config_name": cfg["name"],
        "config_id": config_id(OPENAI_MODEL, cfg["shots"], cfg["max_output_tokens"]),
        "sweep": cfg.get("sweep"),
        "level": cfg.get("level"),
        "shots": cfg["shots"],
        "cap": cfg["max_output_tokens"],
        **summary,
    })
    return summary

# ---------- Matrix runner ----------
def run_matrix(
    suites: Dict[str, Dict[str, Any]],
    experiments: List[Dict[str, Any]],
    run_name: str,
) -> Dict[str, Dict[str, Any]]:
    logger = RunLogger(run_name)
    try:
        out: Dict[str, Dict[str, Any]] = {}
        for suite_name, suite in suites.items():
            tasks = suite["tasks"]
            pool  = suite.get("fewshot_pool", FEWSHOT_POOLS.get(suite_name, []))
            out[suite_name] = {}
            for cfg in experiments:
                summary = run_suite(suite_name, tasks, cfg, logger, fewshot_pool=pool)
                out[suite_name][cfg["name"]] = summary
        return out
    finally:
        logger.close()

if __name__ == "__main__":
    math_tasks = [
        {"question": "What is 2 + 3?", "answer": "5"},
        {"question": "What is 7 + 8?", "answer": "15"},
        {"question": "What is 9 + 2?", "answer": "11"},
    ]
    sentiment_tasks = [
        {"question": "I loved this product!", "answer": "positive"},
        {"question": "It was okay, nothing special.", "answer": "neutral"},
        {"question": "This was terrible.", "answer": "negative"},
    ]
    SUITES = {
        "math_suite": {"tasks": math_tasks, "fewshot_pool": FEWSHOT_POOLS["math_suite"]},
        "sentiment_suite": {"tasks": sentiment_tasks, "fewshot_pool": FEWSHOT_POOLS["sentiment_suite"]},
    }

    cap_fixed_for_shots = 32
    shots_fixed_for_caps = 0
    shot_exps = build_shot_sweep(fixed_cap=cap_fixed_for_shots)
    cap_exps  = build_cap_sweep(fixed_shots=shots_fixed_for_caps)

    print("\n=== SHOT SWEEP ===")
    shot_results = run_matrix(SUITES, shot_exps, run_name="shot-sweep")
    print(json.dumps(shot_results, indent=2))

    print("\n=== CAP SWEEP ===")
    cap_results = run_matrix(SUITES, cap_exps, run_name="cap-sweep")
    print(json.dumps(cap_results, indent=2))