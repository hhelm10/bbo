"""Compute MDS classification results for multiple base models.

Runs serially across base models, uses joblib for parallelism within each (n,m) point.
All use nomic-embed-text-v1.5 as the embedding model.
"""

import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from bbo.queries.query_set import sample_queries
from bbo.queries.distributions import SubsetDistribution
from bbo.classification.evaluate import make_classifier
from bbo.distances.energy import pairwise_energy_distances_t0
from bbo.embedding.mds import ClassicalMDS


def _one_rep(responses, labels, signal_idx, M, n, m, seed):
    rng = np.random.default_rng(seed)
    dist = SubsetDistribution(signal_idx, mass=1.0)
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
    test_idx = np.setdiff1d(np.arange(len(labels)), train_idx)

    clf = make_classifier("rf")
    clf.fit(X[train_idx], labels[train_idx])
    preds = clf.predict(X[test_idx])
    return (preds == labels[test_idx]).mean()


def compute_for_base_model(base_model, embed_model="nomic-embed-text-v1.5",
                           n_values=(10, 80), m_values=(1, 2, 5, 10, 20, 50),
                           n_reps=200, seed=42, n_jobs=-1):
    npz_path = f"results/system_prompt/embeddings/{base_model}__{embed_model}.npz"
    data = np.load(npz_path, allow_pickle=True)
    responses = data["responses"]
    labels = data["labels"]
    signal_idx = data["signal_indices"]

    n_models, M, p = responses.shape
    print(f"  Loaded: {n_models} models, {M} queries, {p}-d", flush=True)

    rows = []
    for n in n_values:
        for m in m_values:
            seeds = [seed + rep * 100003 + m * 1009 + n * 31
                     for rep in range(n_reps)]
            accs = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_one_rep)(responses, labels, signal_idx, M, n, m, s)
                for s in seeds
            )
            rows.append({
                "base_model": base_model, "n": n, "m": m,
                "mean_acc": np.mean(accs), "std_acc": np.std(accs),
            })
            print(f"  {base_model} n={n} m={m}: {np.mean(accs):.1%}", flush=True)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    base_models = [
        "ministral-8b",
        "ministral-3b",
        "mistral-small",
        "gpt-4o-mini",
    ]

    # Allow filtering from command line
    if len(sys.argv) > 1:
        base_models = [bm for bm in base_models if bm in sys.argv[1:]]

    all_dfs = []
    for bm in base_models:
        print(f"=== {bm} ===", flush=True)
        df = compute_for_base_model(bm)
        df.to_csv(f"results/system_prompt/base_panel_{bm}.csv", index=False)
        all_dfs.append(df)
        print(flush=True)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv("results/system_prompt/base_panel_all.csv", index=False)
    print("All done!", flush=True)
