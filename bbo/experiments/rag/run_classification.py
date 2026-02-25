"""Classification experiment for the RAG compliance auditing experiment.

Reuses the MDS-on-all-models pattern from the system_prompt experiment,
with distributions tailored to the RAG query structure:
  - signal: all finance + HR queries
  - finance_signal: finance queries only
  - hr_signal: HR queries only
  - control: public-info queries
  - uniform: all queries equally weighted
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from bbo.queries.query_set import sample_queries
from bbo.queries.distributions import SubsetDistribution, UniformDistribution
from bbo.classification.evaluate import make_classifier
from bbo.distances.energy import pairwise_energy_distances_t0
from bbo.embedding.mds import ClassicalMDS
from bbo.experiments.rag.config import RAGConfig


def _run_one_rep(responses, labels, M, m, dist, seed, n_components, classifier,
                 n_train=None):
    """Single trial: sample queries -> MDS on ALL models -> train/test split -> classify."""
    rng = np.random.default_rng(seed)

    query_idx = sample_queries(M, m, distribution=dist, rng=rng)
    D = pairwise_energy_distances_t0(responses, query_idx)

    if n_components is not None:
        mds = ClassicalMDS(n_components=min(n_components, len(labels) - 1))
    else:
        mds = ClassicalMDS()
    X = mds.fit_transform(D)

    class0 = np.where(labels == 0)[0]
    class1 = np.where(labels == 1)[0]
    if n_train is not None and n_train < len(labels):
        n_per = n_train // 2
        sel0 = rng.choice(class0, size=n_per, replace=False)
        sel1 = rng.choice(class1, size=n_per, replace=False)
        train_idx = np.concatenate([sel0, sel1])
    else:
        n_per0 = max(1, int(0.7 * len(class0)))
        n_per1 = max(1, int(0.7 * len(class1)))
        sel0 = rng.choice(class0, size=n_per0, replace=False)
        sel1 = rng.choice(class1, size=n_per1, replace=False)
        train_idx = np.concatenate([sel0, sel1])

    test_idx = np.setdiff1d(np.arange(len(labels)), train_idx)

    clf = make_classifier(classifier)
    clf.fit(X[train_idx], labels[train_idx])
    preds = clf.predict(X[test_idx])
    return (preds == labels[test_idx]).mean()


def _run_one_rep_concat(responses, labels, M, m, dist, seed, classifier,
                        n_train=None):
    """Baseline: concatenate embeddings directly."""
    rng = np.random.default_rng(seed)

    query_idx = sample_queries(M, m, distribution=dist, rng=rng)
    X = responses[:, query_idx, :].reshape(responses.shape[0], -1)

    class0 = np.where(labels == 0)[0]
    class1 = np.where(labels == 1)[0]
    if n_train is not None and n_train < len(labels):
        n_per = n_train // 2
        sel0 = rng.choice(class0, size=n_per, replace=False)
        sel1 = rng.choice(class1, size=n_per, replace=False)
        train_idx = np.concatenate([sel0, sel1])
    else:
        n_per0 = max(1, int(0.7 * len(class0)))
        n_per1 = max(1, int(0.7 * len(class1)))
        sel0 = rng.choice(class0, size=n_per0, replace=False)
        sel1 = rng.choice(class1, size=n_per1, replace=False)
        train_idx = np.concatenate([sel0, sel1])

    test_idx = np.setdiff1d(np.arange(len(labels)), train_idx)

    clf = make_classifier(classifier)
    clf.fit(X[train_idx], labels[train_idx])
    preds = clf.predict(X[test_idx])
    return (preds == labels[test_idx]).mean()


def run_classification(config: RAGConfig, base_model: str = None,
                       embedding_model: str = None,
                       include_baselines: bool = True) -> pd.DataFrame:
    """Run classification sweep for (base_model, embedding_model) pair."""

    bm = base_model or config.base_model
    em = embedding_model or config.embedding_model

    npz_path = config.npz_path(bm, em)
    data = np.load(str(npz_path), allow_pickle=True)
    responses = data["responses"]
    labels = data["labels"]
    signal_indices = data["signal_indices"]
    finance_signal_indices = data["finance_signal_indices"]
    hr_signal_indices = data["hr_signal_indices"]
    control_indices = data["control_indices"]

    n_models, M, p = responses.shape
    print(f"Loaded: {n_models} systems, {M} queries, {p}-d embeddings")
    print(f"  Signal: {len(signal_indices)} (finance: {len(finance_signal_indices)}, "
          f"HR: {len(hr_signal_indices)})")
    print(f"  Control: {len(control_indices)}")
    print(f"  Labels: {np.unique(labels, return_counts=True)}")

    distributions = {
        "signal": SubsetDistribution(signal_indices, mass=1.0),
        "finance_signal": SubsetDistribution(finance_signal_indices, mass=1.0),
        "hr_signal": SubsetDistribution(hr_signal_indices, mass=1.0),
        "uniform": UniformDistribution(),
    }
    if len(control_indices) > 0:
        distributions["control"] = SubsetDistribution(control_indices, mass=1.0)

    # Cap n_values to available data
    min_class_size = min(np.sum(labels == 0), np.sum(labels == 1))
    n_values = [n for n in config.n_values if n // 2 <= min_class_size]
    print(f"  n_values: {n_values}")

    results = []

    # MDS pipeline
    for n in n_values:
        n_train = n if n < n_models else None
        for dist_name, dist in distributions.items():
            dist_offset = hash(dist_name) % 10000
            desc = f"MDS n={n}, {dist_name}"
            for m in tqdm(config.m_values, desc=desc):
                seeds = [config.seed + rep * 100003 + m * 1009
                         + dist_offset * 7 + n * 31
                         for rep in range(config.n_reps)]
                accuracies = Parallel(n_jobs=config.n_jobs, backend="loky")(
                    delayed(_run_one_rep)(
                        responses, labels, M, m, dist, s,
                        config.n_components, config.classifier, n_train
                    )
                    for s in seeds
                )
                results.append({
                    "method": "mds",
                    "n": n,
                    "distribution": dist_name,
                    "m": m,
                    "mean_accuracy": np.mean(accuracies),
                    "std_accuracy": np.std(accuracies),
                })

    if include_baselines:
        # Concat baseline with signal distribution
        for n in n_values:
            n_train = n if n < n_models else None
            dist = distributions["signal"]
            for m in tqdm(config.m_values, desc=f"Concat n={n}"):
                seeds = [config.seed + rep * 100003 + m * 1009 + n * 31
                         for rep in range(config.n_reps)]
                acc_concat = Parallel(n_jobs=config.n_jobs, backend="loky")(
                    delayed(_run_one_rep_concat)(
                        responses, labels, M, m, dist, s,
                        config.classifier, n_train
                    )
                    for s in seeds
                )
                results.append({
                    "method": "concat",
                    "n": n,
                    "distribution": "signal",
                    "m": m,
                    "mean_accuracy": np.mean(acc_concat),
                    "std_accuracy": np.std(acc_concat),
                })

    return pd.DataFrame(results)
