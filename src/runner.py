from __future__ import annotations
import argparse
import dataclasses
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple
from dotenv import load_dotenv

from utils.prompt_builder import PromptBuilder
from utils.token_helpers import PricingHelper, TokenCoster
from src.adapter import OpenAIAdapter
from src.result_logger import ResultLogger
from src.trial import TrialConfig, TrialResult

# Env Vars
load_dotenv()
VERBOSE = os.getenv("VERBOSE", "0") == "1"
RUN_DIR = Path(os.getenv("RUN_DIR", "runs"))
RUN_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------
# Experiment runner
# --------------------------

def _git_rev() -> str:
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run_trial(
    *,
    adapter: OpenAIAdapter,
    suite_name: str,
    task_id: str,
    task_payload: Dict[str, Any],
    builder: PromptBuilder,
    config: TrialConfig,
    scorer: Callable[[str, str], int],
    logger: ResultLogger,
    run_id: str
) -> TrialResult:
    messages = builder.build_messages(
        shots=config.shots,
        task=task_payload,
        example_order_seed=7  # deterministic across runs
    )
    t0 = time.perf_counter()
    raw, text, usage = adapter.complete(
        model=config.model,
        messages=messages,
        temperature=config.temperature,
        top_p=config.top_p,
        max_output_tokens=config.max_output_tokens,
        reasoning_effort=config.reasoning_effort,
        service_tier=config.service_tier,
        tool_choice=config.tool_choice
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    correct = scorer(task_payload["expected"], text)
    created_at = raw.get("created_at") or time.time()
    response_id = raw.get("id", "")
    pricing_helper = PricingHelper(
        input_per_1k=TokenCoster.PRICE_PER_1K_INPUT,
        output_per_1k=TokenCoster.PRICE_PER_1K_OUTPUT
    )
    result = TrialResult(
        run_id=run_id,
        suite=suite_name,
        task_id=task_id,
        model=config.model,
        temperature=config.temperature,
        top_p=config.top_p,
        shots=config.shots,
        cap=config.max_output_tokens,
        latency_ms=latency_ms,
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        total_tokens=usage["total_tokens"],
        cached_tokens=usage["cached_tokens"],
        reasoning_tokens=usage["reasoning_tokens"],
        cost=pricing_helper.estimate(
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"]
        ) or 0.0,
        response_id=response_id,
        response_text=text,
        correct=int(correct),
        created_at=float(created_at),
        code_rev=_git_rev(),
        prompt_fingerprint=builder.fingerprint()
    )
    logger.log_trial(result, raw)
    return result


def run_sweep(
    *,
    mode: str,  # "shot", "cap", or "grid"
    shots_list: Sequence[int],
    caps_list: Sequence[int],
    suite_name: str,
    tasks: Dict[str, Dict[str, Any]],
    builder: PromptBuilder,
    base_cfg: TrialConfig,
    scorer: Callable[[str, str], int],
    out_dir: Path
) -> None:
    adapter = OpenAIAdapter()
    logger = ResultLogger(out_dir)
    run_id = f"{suite_name}_{mode}_{int(time.time())}"

    def sweep_pairs() -> Iterable[Tuple[int, int]]:
        if mode == "shot":
            for s in shots_list:
                yield (s, base_cfg.max_output_tokens)
        elif mode == "cap":
            for c in caps_list:
                yield (base_cfg.shots, c)
        else:  # grid
            for s in shots_list:
                for c in caps_list:
                    yield (s, c)

    for shots, cap in sweep_pairs():
        cfg = dataclasses.replace(base_cfg, shots=shots, max_output_tokens=cap)
        for task_id, payload in tasks.items():
            run_trial(
                adapter=adapter,
                suite_name=suite_name,
                task_id=task_id,
                task_payload=payload,
                builder=builder,
                config=cfg,
                scorer=scorer,
                logger=logger,
                run_id=run_id
            )

# --------------------------
# CLI
# --------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Shots/Cap Sweeps & Grid Experiments")
    p.add_argument("--suite", choices=["math", "sentiment"], required=True)
    p.add_argument("--mode", choices=["shot", "cap", "grid"], default="shot")
    p.add_argument("--model", default="gpt-5-2025-08-07")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--shots", type=int, default=0, help="Base shots (used in cap sweep)")
    p.add_argument("--cap", type=int, default=32, help="Base cap (used in shot sweep)")
    p.add_argument("--shots_list", type=int, nargs="*", default=[0,2,6,8,10,12,14])
    p.add_argument("--caps_list", type=int, nargs="*", default=[32,64,128,256,512])
    p.add_argument("--reasoning_effort", type=str, default="minimal", choices=["minimal", "low", "medium", "high"])
    p.add_argument("--out_dir", type=Path, default=Path("results"))
    p.add_argument("--plot", action="store_true")
    args = p.parse_args(argv)

    from src.suites import example_math_suite, example_sentiment_suite
    if args.suite == "math":
        builder, tasks, scorer = example_math_suite()
    else:
        builder, tasks, scorer = example_sentiment_suite()

    base_cfg = TrialConfig(
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.cap,
        reasoning_effort=args.reasoning_effort,
        shots=args.shots,
        service_tier="auto",
        tool_choice="none",
    )

    out = args.out_dir / f"{args.suite}_{args.mode}"
    out.mkdir(parents=True, exist_ok=True)
    run_sweep(
        mode=args.mode,
        shots_list=args.shots_list,
        caps_list=args.caps_list,
        suite_name=args.suite,
        tasks=tasks,
        builder=builder,
        base_cfg=base_cfg,
        scorer=scorer,
        out_dir=out
    )

    if args.plot:
        from utils.plot_result import plot_from_csv
        plot_from_csv(out / "trials.csv", mode=args.mode)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
