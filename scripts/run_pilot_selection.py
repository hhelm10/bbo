#!/usr/bin/env python
"""Run pilot-estimated query selection experiment on all three datasets."""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from bbo.experiments.pilot_selection import run_pilot_experiment


# NPZ paths per (dataset, embedding_model)
NPZ_PATHS = {
    ("motivating", "nomic-embed-text-v1.5"): "results/motivating/motivating_responses.npz",
    ("motivating", "all-MiniLM-L6-v2"): "results/motivating/embeddings/all-MiniLM-L6-v2.npz",
    ("motivating", "text-embedding-3-small"): "results/motivating/embeddings/text-embedding-3-small.npz",
    ("system_prompt", "nomic-embed-text-v1.5"): "results/system_prompt/embeddings/ministral-8b__nomic-embed-text-v1.5.npz",
    ("system_prompt", "all-MiniLM-L6-v2"): "results/system_prompt/embeddings/ministral-8b__all-MiniLM-L6-v2.npz",
    ("system_prompt", "text-embedding-3-small"): "results/system_prompt/embeddings/ministral-8b__text-embedding-3-small.npz",
    ("rag", "nomic-embed-text-v1.5"): "results/rag/embeddings/ministral-8b__nomic-embed-text-v1.5.npz",
    ("rag", "all-MiniLM-L6-v2"): "results/rag/embeddings/ministral-8b__all-MiniLM-L6-v2.npz",
    ("rag", "text-embedding-3-small"): "results/rag/embeddings/ministral-8b__text-embedding-3-small.npz",
}

POOL_KEYS = {
    "motivating": ("sensitive_indices", "orthogonal_indices"),
    "system_prompt": ("signal_indices", "orthogonal_indices"),
    "rag": ("signal_indices", "control_indices"),
}

DATASETS = list(POOL_KEYS.keys())
EMBEDDING_MODELS = ["nomic-embed-text-v1.5", "all-MiniLM-L6-v2", "text-embedding-3-small"]
SELECTORS = ("uniform", "uniform_signal", "stepwise")


def main():
    parser = argparse.ArgumentParser(
        description="Run pilot query selection experiment"
    )
    parser.add_argument("datasets", nargs="*", default=DATASETS)
    parser.add_argument("--embedding-model", type=str, default=None,
                        help="Embedding model (default: nomic-embed-text-v1.5)")
    parser.add_argument("--all-embeds", action="store_true",
                        help="Run all embedding models")
    parser.add_argument("--selectors", nargs="+", default=list(SELECTORS))
    parser.add_argument("--n-reps", type=int, default=500)
    parser.add_argument("--output-dir", default="results/pilot_selection")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.all_embeds:
        embed_models = EMBEDDING_MODELS
    elif args.embedding_model:
        embed_models = [args.embedding_model]
    else:
        embed_models = ["nomic-embed-text-v1.5"]

    for name in args.datasets:
        if name not in POOL_KEYS:
            print(f"Unknown dataset: {name}")
            continue

        for em in embed_models:
            key = (name, em)
            if key not in NPZ_PATHS:
                print(f"No NPZ for {name} + {em}, skipping.")
                continue

            npz_path = NPZ_PATHS[key]
            if not Path(npz_path).exists():
                print(f"NPZ not found: {npz_path}, skipping.")
                continue

            print(f"\n{'='*60}")
            print(f"Dataset: {name}, Embedding: {em}")
            print(f"{'='*60}")

            data = np.load(npz_path, allow_pickle=True)
            responses = data["responses"]
            labels = data["labels"]

            pool_keys = POOL_KEYS[name]
            sig_idx = data[pool_keys[0]]
            orth_idx = data[pool_keys[1]]
            query_pool = np.concatenate([sig_idx, orth_idx])
            n_true_signal = len(sig_idx)
            print(f"  Query pool: {len(query_pool)} queries "
                  f"(signal={n_true_signal}, orthogonal={len(orth_idx)})")

            df = run_pilot_experiment(
                responses, labels,
                query_pool=query_pool,
                n_true_signal=n_true_signal,
                selectors=tuple(args.selectors),
                n_reps=args.n_reps,
                classifier="lda",
            )

            # Output filename includes embed model if not default
            if em == "nomic-embed-text-v1.5" and not args.all_embeds:
                out_path = f"{args.output_dir}/{name}.csv"
            else:
                out_path = f"{args.output_dir}/{name}__{em}.csv"
            df.to_csv(out_path, index=False)
            print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
