import csv
import json
import dataclasses
from pathlib import Path
from typing import Dict, Any

from src.trial import TrialResult

class ResultLogger:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.out_dir / "trials.csv"
        self.jsonl_path = self.out_dir / "trials.jsonl"
        self._csv_initialized = False

    def log_trial(self, result: TrialResult, raw_response: Dict[str, Any]) -> None:
        row = dataclasses.asdict(result)
        write_header = (not self._csv_initialized) and (not self.csv_path.exists())
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)
        self._csv_initialized = True

        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"result": row, "raw": raw_response}, ensure_ascii=False) + "\n")
