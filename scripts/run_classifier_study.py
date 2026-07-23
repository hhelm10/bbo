#!/usr/bin/env python
"""Classifier robustness study: effect of downstream classifier choice.

For each dataset, samples m signal queries uniformly, embeds models via the
distance -> MDS pipeline, then evaluates four classifiers (RF, 1NN, Linear
SVM, RBF SVM) on the SAME embedding per trial (paired comparison).

Outputs:
    results/classifier_study/summary.csv  (per-trial accuracies)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed

from bbo.distances.energy import pairwise_energy_distances_t0
from bbo.embedding.mds import ClassicalMDS
from bbo.classification.evaluate import make_classifier

DATASETS = {
    "motivating": ("results/motivating/motivating_responses.npz",
                   "sensitive_indices"),
    "system_prompt": ("results/system_prompt/embeddings/ministral-8b__nomic-embed-text-v1.5.npz",
                      "signal_indices"),
    "rag": ("results/rag/embeddings/ministral-8b__nomic-embed-text-v1.5.npz",
            "signal_indices"),
}

CLASSIFIERS = {
    "rf": ("rf", {}),
    "1nn": ("knn", {"n_neighbors": 1}),
    "linear_svm": ("svm", {"kernel": "linear"}),
    "rbf_svm": ("svm", {"kernel": "rbf"}),
}

M_GRID = [1, 2, 5, 10, 25, 50, 100]
N_REPS = 50
N_COMPONENTS = 8


def _one_trial(responses, labels, signal_idx, m, seed):
    rng = np.random.default_rng(seed)
    query_idx = rng.choice(signal_idx, size=m, replace=False)
    D = pairwise_energy_distances_t0(responses, query_idx)
    mds = ClassicalMDS(n_components=min(N_COMPONENTS, len(labels) - 1))
    X = mds.fit_transform(D)

    class0 = np.where(labels == 0)[0]
    class1 = np.where(labels == 1)[0]
    sel0 = rng.choice(class0, size=int(0.7 * len(class0)), replace=False)
    sel1 = rng.choice(class1, size=int(0.7 * len(class1)), replace=False)
    train_idx = np.concatenate([sel0, sel1])
    test_idx = np.setdiff1d(np.arange(len(labels)), train_idx)

    out = {}
    for clf_label, (name, kwargs) in CLASSIFIERS.items():
        clf = make_classifier(name, **kwargs)
        clf.fit(X[train_idx], labels[train_idx])
        preds = clf.predict(X[test_idx])
        out[clf_label] = float((preds == labels[test_idx]).mean())
    return out


def main():
    out_dir = Path("results/classifier_study")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ds, (npz_path, sig_key) in DATASETS.items():
        data = np.load(npz_path, allow_pickle=True)
        responses = data["responses"]
        labels = data["labels"]
        signal_idx = data[sig_key]
        print(f"\nDataset: {ds}")

        for m in M_GRID:
            results = Parallel(n_jobs=8)(
                delayed(_one_trial)(responses, labels, signal_idx, m,
                                    seed=1000 * m + rep)
                for rep in range(N_REPS))
            for rep, accs in enumerate(results):
                for clf_label, acc in accs.items():
                    rows.append({"dataset": ds, "m": m, "rep": rep,
                                 "classifier": clf_label, "accuracy": acc})
            means = {c: np.mean([r[c] for r in results]) for c in CLASSIFIERS}
            print(f"  m={m:3d}: " + ", ".join(f"{c}={v:.3f}"
                                              for c, v in means.items()))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary.csv", index=False)
    print(f"\nSaved to {out_dir}/summary.csv")


if __name__ == "__main__":
    main()
