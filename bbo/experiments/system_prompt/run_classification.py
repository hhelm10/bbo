"""Classification experiment for the system prompt experiment.

Reuses the MDS-on-all-models pattern from the motivating experiment,
plus baselines (single best query, raw concatenation).
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
from bbo.experiments.system_prompt.config import SystemPromptConfig


def _run_one_rep(responses, labels, M, m, dist, seed, n_components, classifier,
                 n_train=None):
    """Single trial: sample queries -> MDS on ALL models -> label n_train -> classify rest."""
    rng = np.random.default_rng(seed)

    query_idx = sample_queries(M, m, distribution=dist, rng=rng)
    D = pairwise_energy_distances_t0(responses, query_idx)

    mds = ClassicalMDS(n_components=min(n_components, len(labels) - 1))
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
    # (n_models, m, p) -> (n_models, m*p)
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


def _run_single_query_baseline(responses, labels, seed, classifier, n_train=None):
    """Baseline: classify using each single query's embedding. Return best accuracy."""
    rng = np.random.default_rng(seed)
    n_models, M, p = responses.shape

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

    best_acc = 0.0
    for q in range(M):
        X = responses[:, q, :]
        clf = make_classifier(classifier)
        clf.fit(X[train_idx], labels[train_idx])
        preds = clf.predict(X[test_idx])
        acc = (preds == labels[test_idx]).mean()
        best_acc = max(best_acc, acc)

    return best_acc


def run_classification(config: SystemPromptConfig, base_model: str = None,
                       embedding_model: str = None,
                       include_baselines: bool = True) -> pd.DataFrame:
    """Run classification sweep for a (base_model, embedding_model) pair."""

    bm = base_model or config.base_models[0]
    em = embedding_model or config.embedding_models[0]

    npz_path = config.npz_path(bm, em)
    data = np.load(str(npz_path), allow_pickle=True)
    responses = data["responses"]
    labels = data["labels"]
    signal_indices = data["signal_indices"]
    orthogonal_indices = data["orthogonal_indices"]
    # Load three-tier indices if available
    weak_signal_indices = (data["weak_signal_indices"]
                           if "weak_signal_indices" in data else np.array([], dtype=np.int64))
    null_indices = data["null_indices"] if "null_indices" in data else orthogonal_indices

    n_models, M, p = responses.shape
    print(f"Loaded: {n_models} models, {M} queries, {p}-d embeddings")
    print(f"  Signal: {len(signal_indices)}, Weak-signal: {len(weak_signal_indices)}, "
          f"Null: {len(null_indices)}")
    print(f"  Labels: {np.unique(labels, return_counts=True)}")

    distributions = {
        "relevant": SubsetDistribution(signal_indices, mass=1.0),
        "uniform": UniformDistribution(),
    }
    if len(weak_signal_indices) > 0:
        distributions["weak_signal"] = SubsetDistribution(weak_signal_indices, mass=1.0)
    if len(null_indices) > 0:
        distributions["orthogonal"] = SubsetDistribution(null_indices, mass=1.0)

    # Cap n_values
    min_class_size = min(np.sum(labels == 0), np.sum(labels == 1))
    n_values = [n for n in config.n_values if n // 2 <= min_class_size]
    print(f"  n_values: {n_values}")

    results = []

    # Main pipeline: MDS
    for n in n_values:
        n_train = n if n < n_models else None
        for dist_name, dist in distributions.items():
            dist_offset = hash(dist_name) % 10000
            desc = f"MDS n={n}, {dist_name}"
            for m in tqdm(config.m_values, desc=desc):
                seeds = [config.seed + rep * 100003 + m * 1009 + dist_offset * 7 + n * 31
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
        # Baseline: raw concatenation and PCA
        for n in n_values:
            n_train = n if n < n_models else None
            dist = distributions["relevant"]
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
                    "distribution": "relevant",
                    "m": m,
                    "mean_accuracy": np.mean(acc_concat),
                    "std_accuracy": np.std(acc_concat),
                })

        # Baseline: single best query
        for n in n_values:
            n_train = n if n < n_models else None
            seeds = [config.seed + rep * 100003 for rep in range(config.n_reps)]
            accs = Parallel(n_jobs=config.n_jobs, backend="loky")(
                delayed(_run_single_query_baseline)(
                    responses, labels, s, config.classifier, n_train
                )
                for s in tqdm(seeds, desc=f"Single-query n={n}")
            )
            results.append({
                "method": "single_best_query",
                "n": n,
                "distribution": "relevant",
                "m": 1,
                "mean_accuracy": np.mean(accs),
                "std_accuracy": np.std(accs),
            })

    return pd.DataFrame(results)
