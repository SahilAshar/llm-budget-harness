# llm-budget-harness

# LLM Token Harness — Accuracy, Latency & Cost-Per-Correct

A lightweight, reproducible harness for evaluating LLM configurations across token caps, few-shot counts, and reasoning effort. It logs per-trial accuracy, latency, and token usage, and lets you roll that up into Cost-Per-Correct (CPC) to make more efficient decisions.

> CPC (Cost-Per-Correct) = total $ spend ÷ number of correct outputs.

## Quickstart
```bash
git clone <your-repo-url>
cd <repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env to add your OpenAI key; optional: OPENAI_MODEL
```

Run a few presets (see `main.py` examples):
```bash
# Sweep shots at fixed cap
python main.py --suite math --mode shot --cap 128 --shots_list 0 2 8 --plot

# Sweep caps at fixed shots
python main.py --suite math --mode cap --shots 2 --caps_list 64 128 256 --plot

# Grid (shots x caps):
python main.py --suite math --mode grid --shots_list 0 2 8 --caps_list 64 128 256 --plot
```

Artifacts land in `results/<suite>_<mode>/<run_id>`:
- `trials.csv`: per-trial rows
- `trials.jsonl`: raw response + normalized summary
- `plot_*.png`: quick plots from utils/plot_result.py

## What’s included

### Datasets
- `data/tasks/tiny_math_v1.json`: small arithmetic + applied word problems
- `data/tasks/math_hard_v1.json`: harder 5-digit carry/borrow, order of ops, units, decimals, formatting noise, % problems
- `data/sentiment_v0.json`: tiny binary/ternary sentiment samples

### Core
- `src/adapter.py`: OpenAI wrapper (model, temp, top-p, max_output_tokens, reasoning={"effort": ...}), normalized usage extraction (input/output/total/cached/reasoning_tokens)

- `src/suites.py`: suite builders (math & sentiment), deterministic few-shot via PromptBuilder, strict JSON output contract, scorer

- `src/result_logger.py`: appends TrialResult rows to CSV + full JSONL

- `utils/prompt_builder.py`: deterministic example selection + prompt hashing

- `utils/plot_result.py`: quick plots: Accuracy vs. shots/cap, and heatmap

- `utils/token_helpers.py`: price map + helpers
  - Defaults (GPT-5 Pricing): PRICE_PER_1K_INPUT=0.00125, PRICE_PER_1K_OUTPUT=0.01000