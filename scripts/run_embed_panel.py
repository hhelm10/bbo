"""Run classification sweeps for all embedding models (right panel of figure).

Saves results after each model completes, so partial progress is preserved.
Usage: python scripts/run_embed_panel.py
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from bbo.queries.query_set import sample_queries
from bbo.queries.distributions import SubsetDistribution
from bbo.classification.evaluate import make_classifier
from bbo.distances.energy import pairwise_energy_distances_t0
from bbo.embedding.mds import ClassicalMDS


def compute_for_embed(embed_model, n_values=(10, 80), m_values=(1, 2, 5, 10, 20, 50),
                      n_reps=200, seed=42):
    npz_path = f"results/system_prompt/embeddings/ministral-8b__{embed_model}.npz"
    data = np.load(npz_path, allow_pickle=True)
    responses = data["responses"]
    labels = data["labels"]
    signal_idx = data["signal_indices"]

    n_models, M, p = responses.shape
    dist = SubsetDistribution(signal_idx, mass=1.0)
    print(f"  Shape: ({n_models}, {M}, {p})", flush=True)

    rows = []
    for n in n_values:
        for m in m_values:
            accs = []
            for rep in range(n_reps):
                rng = np.random.default_rng(seed + rep * 100003 + m * 1009 + n * 31)
                query_idx = sample_queries(M, m, distribution=dist, rng=rng)
                D = pairwise_energy_distances_t0(responses, query_idx)
                mds = ClassicalMDS(n_components=min(10, n_models - 1))
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

            rows.append({
                "embed_model": embed_model, "n": n, "m": m,
                "mean_acc": np.mean(accs), "std_acc": np.std(accs),
            })
            print(f"  {embed_model} n={n} m={m}: {np.mean(accs):.1%}", flush=True)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    models = ["all-MiniLM-L6-v2", "text-embedding-3-small",
              "text-embedding-3-large", "gemini-embedding"]

    # Allow starting from a specific model
    start_from = sys.argv[1] if len(sys.argv) > 1 else None
    if start_from:
        idx = models.index(start_from)
        models = models[idx:]

    out_dir = Path("results/system_prompt")
    all_dfs = []

    for em in models:
        csv_path = out_dir / f"embed_panel_{em}.csv"
        if csv_path.exists():
            print(f"=== {em} === (already done, loading)", flush=True)
            all_dfs.append(pd.read_csv(csv_path))
            continue

        print(f"=== {em} ===", flush=True)
        df = compute_for_embed(em)
        df.to_csv(csv_path, index=False)
        all_dfs.append(df)
        print(f"  Saved to {csv_path}", flush=True)
        print(flush=True)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(out_dir / "embed_panel_all.csv", index=False)
    print("All done!", flush=True)
