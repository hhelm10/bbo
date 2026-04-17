#!/usr/bin/env python
"""Run pilot-estimated query selection experiment on all three datasets."""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from bbo.experiments.pilot_selection import run_pilot_experiment


DATASETS = {
    "motivating": {
        "npz": "results/motivating/motivating_responses.npz",
    },
    "system_prompt": {
        "npz": "results/system_prompt/embeddings/mistral-small__nomic-embed-text-v1.5.npz",
    },
    "rag": {
        "npz": "results/rag/embeddings/ministral-8b__nomic-embed-text-v1.5.npz",
    },
}

SELECTORS = ("uniform", "uniform_signal", "uniform_orthogonal", "greedy")


def main():
    parser = argparse.ArgumentParser(
        description="Run pilot query selection experiment"
    )
    parser.add_argument("datasets", nargs="*", default=list(DATASETS.keys()))
    parser.add_argument("--n-reps", type=int, default=500)
    parser.add_argument("--output-dir", default="results/pilot_selection")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for name in args.datasets:
        if name not in DATASETS:
            print(f"Unknown dataset: {name}")
            continue

        cfg = DATASETS[name]
        print(f"\n{'='*60}")
        print(f"Dataset: {name}")
        print(f"{'='*60}")

        data = np.load(cfg["npz"], allow_pickle=True)
        responses = data["responses"]
        labels = data["labels"]

        df = run_pilot_experiment(
            responses, labels,
            selectors=SELECTORS,
            n_reps=args.n_reps,
        )

        out_path = f"{args.output_dir}/{name}.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
