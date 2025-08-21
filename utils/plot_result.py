import sys
import csv
from collections import defaultdict
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ----------------
# Plotting helper
# ----------------

def plot_from_csv(csv_path: Path, mode: str) -> None:
    """
    Quick matplotlib plotter:
      - For 'shot' mode: accuracy vs shots, latency vs shots
      - For 'cap' mode: accuracy vs cap, latency vs cap
      - For 'grid' mode: heatmap of accuracy over shots x cap
    """

    df = pd.read_csv(csv_path / f"trials.csv")
    if mode in ("shot", "cap"):
        key = "shots" if mode == "shot" else "cap"
        g = df.groupby([key]).agg(acc=("correct", "mean"), lat=("latency_ms", "median")).reset_index()

        plt.figure()
        plt.plot(g[key], g["acc"], marker="o")
        plt.title(f"Accuracy vs {key}")
        plt.xlabel(key)
        plt.ylabel("Accuracy")
        plt.tight_layout()

        plt.figure()
        plt.plot(g[key], g["lat"], marker="o")
        plt.title(f"Median latency vs {key}")
        plt.xlabel(key)
        plt.ylabel("Latency (ms)")
        plt.tight_layout()
        plt.savefig(csv_path / f"plot_{mode}.png")
    else:
        # grid heatmap
        import numpy as np
        pivot = df.pivot_table(index="shots", columns="cap", values="correct", aggfunc="mean")
        plt.figure()
        plt.imshow(pivot.values, aspect="auto", origin="lower")
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45)
        plt.colorbar(label="Accuracy")
        plt.title("Accuracy heatmap (shots x cap)")
        plt.xlabel("cap (max_output_tokens)")
        plt.ylabel("shots")
        plt.tight_layout()
        plt.savefig(csv_path / f"plot_grid.png")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python plot_result.py <csv_path> <mode>")
        sys.exit(1)
    csv_path = Path(sys.argv[1])
    mode = sys.argv[2]
    plot_from_csv(csv_path, mode)
    print(f"Plots saved to {csv_path}")
    sys.exit(0)