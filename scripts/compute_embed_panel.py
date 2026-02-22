"""Compute MDS classification results for multiple embedding models.

Runs serially across embedding models, uses joblib for parallelism within each (n,m) point.
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


def compute_for_embed(embed_model, n_values=(10, 80), m_values=(1, 2, 5, 10, 20, 50),
                      n_reps=200, seed=42, n_jobs=-1):
    npz_path = f"results/system_prompt/embeddings/ministral-8b__{embed_model}.npz"
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
                "embed_model": embed_model, "n": n, "m": m,
                "mean_acc": np.mean(accs), "std_acc": np.std(accs),
            })
            print(f"  {embed_model} n={n} m={m}: {np.mean(accs):.1%}", flush=True)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    embed_models = [
        "all-MiniLM-L6-v2",
        "text-embedding-3-small",
        "text-embedding-3-large",
        "gemini-embedding",
    ]

    all_dfs = []
    for em in embed_models:
        print(f"=== {em} ===", flush=True)
        df = compute_for_embed(em)
        df.to_csv(f"results/system_prompt/embed_panel_{em}.csv", index=False)
        all_dfs.append(df)
        print(flush=True)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv("results/system_prompt/embed_panel_all.csv", index=False)
    print("All done!", flush=True)
