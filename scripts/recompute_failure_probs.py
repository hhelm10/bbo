#!/usr/bin/env python
"""Recompute failure_probs.csv for all three real-data experiments with more reps."""

import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed

from bbo.queries.query_set import sample_queries
from bbo.queries.distributions import SubsetDistribution
from bbo.classification.evaluate import make_classifier
from bbo.distances.energy import pairwise_energy_distances_t0
from bbo.embedding.mds import ClassicalMDS


def _single_trial(responses, labels, M, m, dist, n_train, seed):
    """One trial: sample queries, MDS, classify, return 1 if error >= 0.5."""
    rng = np.random.default_rng(seed)
    query_idx = sample_queries(M, m, distribution=dist, rng=rng)
    D = pairwise_energy_distances_t0(responses, query_idx)
    X = ClassicalMDS().fit_transform(D)

    class0 = np.where(labels == 0)[0]
    class1 = np.where(labels == 1)[0]
    n_per = n_train // 2
    sel0 = rng.choice(class0, n_per, replace=False)
    sel1 = rng.choice(class1, n_per, replace=False)
    train_idx = np.concatenate([sel0, sel1])
    test_idx = np.setdiff1d(np.arange(len(labels)), train_idx)

    clf = make_classifier("rf")
    clf.fit(X[train_idx], labels[train_idx])
    preds = clf.predict(X[test_idx])
    acc = (preds == labels[test_idx]).mean()
    return int(acc <= 0.5)


def compute_failure_probs(responses, labels, all_idx, m_values, n_values,
                          n_reps=1000, seed=42, n_jobs=-1):
    """Compute P[err >= 0.5] with parallel reps."""
    n_models, M, p = responses.shape
    dist = SubsetDistribution(all_idx, mass=1.0)

    rows = []
    for n in n_values:
        for m in m_values:
            seeds = [seed + rep * 100003 + m * 1009 + n * 31
                     for rep in range(n_reps)]
            fails = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_single_trial)(responses, labels, M, m, dist, n, s)
                for s in seeds
            )
            fail_prob = np.mean(fails)
            rows.append({
                "query_set": "uniform",
                "n": n,
                "m": m,
                "failure_prob": fail_prob,
            })
            print(f"  n={n}, m={m}: P[fail]={fail_prob:.4f} ({sum(fails)}/{n_reps})")

    return pd.DataFrame(rows)


def compute_failure_probs_rag(responses, labels, signal_idx, control_idx,
                              m_values, n_values, n_reps=1000, seed=42,
                              n_jobs=-1):
    """Compute P[err >= 0.5] for RAG signal and control query sets."""
    n_models, M, p = responses.shape

    query_sets = {
        "signal": SubsetDistribution(signal_idx, mass=1.0),
        "control": SubsetDistribution(control_idx, mass=1.0),
    }

    rows = []
    for qs_name, dist in query_sets.items():
        for n in n_values:
            for m in m_values:
                seeds = [seed + rep * 100003 + m * 1009 + n * 31
                         + hash(qs_name) % 10000
                         for rep in range(n_reps)]
                fails = Parallel(n_jobs=n_jobs, backend="loky")(
                    delayed(_single_trial)(responses, labels, M, m, dist, n, s)
                    for s in seeds
                )
                fail_prob = np.mean(fails)
                rows.append({
                    "query_set": qs_name,
                    "n": n,
                    "m": m,
                    "failure_prob": fail_prob,
                })
                print(f"  {qs_name} n={n}, m={m}: P[fail]={fail_prob:.4f}")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-reps", type=int, default=1000)
    parser.add_argument("--datasets", nargs="*",
                        default=["motivating", "system_prompt", "rag"])
    args = parser.parse_args()

    m_values = [1, 2, 5, 10, 20, 50, 100]

    if "motivating" in args.datasets:
        print("\n=== Motivating ===")
        data = np.load("results/motivating/motivating_responses.npz",
                        allow_pickle=True)
        sig = data["sensitive_indices"]
        orth = data["orthogonal_indices"]
        all_idx = np.concatenate([sig, orth])
        df = compute_failure_probs(
            data["responses"], data["labels"], all_idx,
            m_values=m_values, n_values=[20, 80],
            n_reps=args.n_reps,
        )
        df.to_csv("results/motivating/failure_probs.csv", index=False)
        print("Saved results/motivating/failure_probs.csv")

    if "system_prompt" in args.datasets:
        print("\n=== System Prompt ===")
        data = np.load(
            "results/system_prompt/embeddings/mistral-small__nomic-embed-text-v1.5.npz",
            allow_pickle=True,
        )
        sig = data["signal_indices"]
        orth = data["orthogonal_indices"]
        all_idx = np.concatenate([sig, orth])
        df = compute_failure_probs(
            data["responses"], data["labels"], all_idx,
            m_values=m_values, n_values=[20, 80],
            n_reps=args.n_reps,
        )
        df.to_csv("results/system_prompt/failure_probs.csv", index=False)
        print("Saved results/system_prompt/failure_probs.csv")

    if "rag" in args.datasets:
        print("\n=== RAG ===")
        data = np.load(
            "results/rag/embeddings/ministral-8b__nomic-embed-text-v1.5.npz",
            allow_pickle=True,
        )
        df = compute_failure_probs_rag(
            data["responses"], data["labels"],
            data["signal_indices"], data["control_indices"],
            m_values=m_values, n_values=[20, 80],
            n_reps=args.n_reps,
        )
        df.to_csv("results/rag/failure_probs.csv", index=False)
        print("Saved results/rag/failure_probs.csv")
