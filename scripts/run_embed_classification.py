"""Run MDS classification for embedding models, saving incrementally."""

import numpy as np
import pandas as pd
import gc
import sys
from pathlib import Path

from bbo.queries.query_set import sample_queries
from bbo.queries.distributions import SubsetDistribution
from bbo.classification.evaluate import make_classifier
from bbo.distances.energy import pairwise_energy_distances_t0
from bbo.embedding.mds import ClassicalMDS


def run_single_point(responses, labels, signal_idx, n, m, n_reps=100, seed=42):
    """Run classification for one (n, m) point."""
    n_models, M, p = responses.shape
    dist = SubsetDistribution(signal_idx, mass=1.0)
    accs = []
    for rep in range(n_reps):
        rng = np.random.default_rng(seed + rep * 100003 + m * 1009 + n * 31)
        query_idx = sample_queries(M, m, distribution=dist, rng=rng)
        D = pairwise_energy_distances_t0(responses, query_idx)
        mds = ClassicalMDS()
        X = mds.fit_transform(D)
        class0 = np.where(labels == 0)[0]
        class1 = np.where(labels == 1)[0]
        n_per = n // 2
        sel0 = rng.choice(class0, n_per, replace=False)
        sel1 = rng.choice(class1, n_per, replace=False)
        train_idx = np.concatenate([sel0, sel1])
        test_idx = np.setdiff1d(np.arange(n_models), train_idx)
        clf = make_classifier("rf")
        clf.fit(X[train_idx], labels[train_idx])
        preds = clf.predict(X[test_idx])
        accs.append((preds == labels[test_idx]).mean())
    return np.mean(accs), np.std(accs)


if __name__ == "__main__":
    embed_model = sys.argv[1]
    n_reps = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    npz_path = f"results/system_prompt/embeddings/ministral-8b__{embed_model}.npz"
    data = np.load(npz_path, allow_pickle=True)
    responses = data["responses"]
    labels = data["labels"]
    signal_idx = data["signal_indices"]
    print(f"{embed_model}: {responses.shape}, n_reps={n_reps}", flush=True)

    out_csv = f"results/system_prompt/embed_panel_{embed_model}.csv"

    # Load existing results to skip completed points
    done = set()
    rows = []
    if Path(out_csv).exists():
        df_existing = pd.read_csv(out_csv)
        rows = df_existing.to_dict("records")
        for r in rows:
            done.add((r["n"], r["m"]))
        print(f"Loaded {len(rows)} existing results", flush=True)

    for n in [10, 80]:
        for m in [1, 2, 5, 10, 20, 50]:
            if (n, m) in done:
                print(f"  n={n} m={m}: SKIP (already done)", flush=True)
                continue
            mean_acc, std_acc = run_single_point(
                responses, labels, signal_idx, n, m, n_reps=n_reps
            )
            rows.append({
                "embed_model": embed_model, "n": n, "m": m,
                "mean_acc": mean_acc, "std_acc": std_acc,
            })
            # Save after each point
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"  n={n} m={m}: {mean_acc:.1%} (saved)", flush=True)
            gc.collect()

    print("Done!", flush=True)
