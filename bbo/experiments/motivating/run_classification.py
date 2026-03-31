"""Classification experiment for the motivating example.

Like exp8_relevant_vs_orthogonal but uses a-priori topic partition
instead of post-hoc partition_queries_by_relevance().
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed

from bbo.queries.query_set import sample_queries
from bbo.queries.distributions import SubsetDistribution, UniformDistribution
from bbo.classification.evaluate import make_classifier
from bbo.distances.energy import pairwise_energy_distances_t0
from bbo.embedding.mds import ClassicalMDS
from bbo.experiments.motivating.config import MotivatingConfig


def _run_one_rep(responses, labels, M, m, dist, seed, n_components, classifier,
                 n_train=None):
    """Single trial: sample queries -> MDS on ALL models -> label n_train -> classify rest."""
    rng = np.random.default_rng(seed)

    # 1. Sample queries
    query_idx = sample_queries(M, m, distribution=dist, rng=rng)

    # 2. Compute distances on ALL models
    D = pairwise_energy_distances_t0(responses, query_idx)

    # 3. MDS on ALL models
    if n_components is not None:
        mds = ClassicalMDS(n_components=min(n_components, len(labels) - 1))
    else:
        mds = ClassicalMDS()
    X = mds.fit_transform(D)

    # 4. Select n_train models as labeled (balanced), rest are test
    class0 = np.where(labels == 0)[0]
    class1 = np.where(labels == 1)[0]
    if n_train is not None and n_train < len(labels):
        n_per = n_train // 2
        sel0 = rng.choice(class0, size=n_per, replace=False)
        sel1 = rng.choice(class1, size=n_per, replace=False)
        train_idx = np.concatenate([sel0, sel1])
    else:
        # Use 70% as train
        n_per0 = max(1, int(0.7 * len(class0)))
        n_per1 = max(1, int(0.7 * len(class1)))
        sel0 = rng.choice(class0, size=n_per0, replace=False)
        sel1 = rng.choice(class1, size=n_per1, replace=False)
        train_idx = np.concatenate([sel0, sel1])

    test_idx = np.setdiff1d(np.arange(len(labels)), train_idx)

    # 5. Train on labeled, predict on unlabeled
    clf = make_classifier(classifier)
    clf.fit(X[train_idx], labels[train_idx])
    preds = clf.predict(X[test_idx])
    accuracy = (preds == labels[test_idx]).mean()
    return accuracy, query_idx


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


def run_classification(config: MotivatingConfig) -> pd.DataFrame:
    """Run the classification experiment with a-priori query partition.

    Sweeps over n_values (number of adapters) and m_values (number of queries).
    Saves per-trial stats (min/max/median) and concat baseline.
    Also saves query indices for median and worst trials at m=10
    for the MDS scatter panels.

    Parameters
    ----------
    config : MotivatingConfig

    Returns
    -------
    df : DataFrame with columns: method, n, distribution, m,
         mean_accuracy, std_accuracy, min_accuracy, max_accuracy, median_accuracy
    """
    # Load embedded responses
    data = np.load(config.npz_path, allow_pickle=True)
    responses = data["responses"]
    labels = data["labels"]
    sensitive_indices = data["sensitive_indices"]
    orthogonal_indices = data["orthogonal_indices"]

    n_models, M, p = responses.shape
    print(f"Loaded: {n_models} models, {M} queries, {p}-d embeddings")
    print(f"  Sensitive queries: {len(sensitive_indices)}")
    print(f"  Orthogonal queries: {len(orthogonal_indices)}")
    print(f"  Labels: {np.unique(labels, return_counts=True)}")

    # A-priori distributions
    distributions = {
        "relevant": SubsetDistribution(sensitive_indices, mass=1.0),
        "orthogonal": SubsetDistribution(orthogonal_indices, mass=1.0),
        "uniform": UniformDistribution(),
    }

    n_values = getattr(config, "n_values", [n_models])

    results = []
    per_trial_data = {}  # for saving MDS scatter query indices

    # Cap n_values based on available class sizes
    min_class_size = min(np.sum(labels == 0), np.sum(labels == 1))
    n_values = [n for n in n_values if n // 2 <= min_class_size]
    print(f"  n_values (after capping to min class size {min_class_size}): {n_values}")

    for n in n_values:
        n_train = n if n < n_models else None
        for dist_name, dist in distributions.items():
            dist_offset = hash(dist_name) % 10000
            desc = f"n={n}, {dist_name}"
            for m in tqdm(config.m_values, desc=desc):
                seeds = [config.seed + rep * 100003 + m * 1009 + dist_offset * 7
                         + n * 31
                         for rep in range(config.n_reps)]

                # MDS trials (return accuracy + query_idx)
                trial_results = Parallel(n_jobs=config.n_jobs, backend="loky")(
                    delayed(_run_one_rep)(
                        responses, labels, M, m, dist, s,
                        config.n_components, config.classifier, n_train
                    )
                    for s in seeds
                )
                accuracies = np.array([r[0] for r in trial_results])

                results.append({
                    "method": "mds",
                    "n": n,
                    "distribution": dist_name,
                    "m": m,
                    "mean_accuracy": accuracies.mean(),
                    "std_accuracy": accuracies.std(),
                    "min_accuracy": accuracies.min(),
                    "max_accuracy": accuracies.max(),
                    "median_accuracy": np.median(accuracies),
                })

                # Save query indices for median/worst at m=10, relevant, largest n
                if m == 10 and dist_name == "relevant" and n == n_values[-1]:
                    median_idx = np.argsort(np.abs(accuracies - np.median(accuracies)))[0]
                    worst_idx = np.argmin(accuracies)
                    best_idx = np.argmax(accuracies)
                    per_trial_data["query_indices_median"] = trial_results[median_idx][1]
                    per_trial_data["query_indices_worst"] = trial_results[worst_idx][1]
                    per_trial_data["query_indices_best"] = trial_results[best_idx][1]
                    per_trial_data["accuracy_median"] = accuracies[median_idx]
                    per_trial_data["accuracy_worst"] = accuracies[worst_idx]
                    per_trial_data["accuracy_best"] = accuracies[best_idx]

                # Concat baseline
                acc_concat = Parallel(n_jobs=config.n_jobs, backend="loky")(
                    delayed(_run_one_rep_concat)(
                        responses, labels, M, m, dist, s,
                        config.classifier, n_train
                    )
                    for s in seeds
                )
                acc_concat = np.array(acc_concat)

                results.append({
                    "method": "concat",
                    "n": n,
                    "distribution": dist_name,
                    "m": m,
                    "mean_accuracy": acc_concat.mean(),
                    "std_accuracy": acc_concat.std(),
                    "min_accuracy": acc_concat.min(),
                    "max_accuracy": acc_concat.max(),
                    "median_accuracy": np.median(acc_concat),
                })

    # Save per-trial query indices for MDS scatter
    if per_trial_data:
        out_dir = Path(config.npz_path).parent
        npz_path = out_dir / "per_trial_m10.npz"
        np.savez(str(npz_path), **per_trial_data)
        print(f"Saved per-trial m=10 data to {npz_path}")

    return pd.DataFrame(results)
