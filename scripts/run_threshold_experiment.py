#!/usr/bin/env python
"""Run threshold sensitivity experiment on all three datasets."""

import argparse
import numpy as np
from pathlib import Path

from bbo.experiments.pilot_selection import run_threshold_experiment


DATASETS = {
    "motivating": {
        "npz": "results/motivating/motivating_responses.npz",
        "pool_keys": ("sensitive_indices", "orthogonal_indices"),
    },
    "system_prompt": {
        "npz": "results/system_prompt/embeddings/ministral-8b__nomic-embed-text-v1.5.npz",
        "pool_keys": ("signal_indices", "orthogonal_indices"),
    },
    "rag": {
        "npz": "results/rag/embeddings/ministral-8b__nomic-embed-text-v1.5.npz",
        "pool_keys": ("signal_indices", "control_indices"),
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Run threshold sensitivity experiment"
    )
    parser.add_argument("datasets", nargs="*", default=list(DATASETS.keys()))
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--n-reps", type=int, default=100)
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

        pool_keys = cfg["pool_keys"]
        sig_idx = data[pool_keys[0]]
        orth_idx = data[pool_keys[1]]
        query_pool = np.concatenate([sig_idx, orth_idx])
        n_true_signal = len(sig_idx)
        print(f"  Query pool: {len(query_pool)} queries "
              f"(signal={n_true_signal}, orthogonal={len(orth_idx)})")

        df = run_threshold_experiment(
            responses, labels,
            query_pool=query_pool,
            n_true_signal=n_true_signal,
            m=args.m,
            n_reps=args.n_reps,
            classifier="lda",
        )

        out_path = f"{args.output_dir}/{name}__thresholds.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
