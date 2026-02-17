"""Classification experiment for the motivating example.

Like exp8_relevant_vs_orthogonal but uses a-priori topic partition
instead of post-hoc partition_queries_by_relevance().
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from bbo.queries.query_set import sample_queries
from bbo.queries.distributions import SubsetDistribution, UniformDistribution
from bbo.classification.evaluate import single_trial
from bbo.experiments.motivating.config import MotivatingConfig


def _run_one_rep(responses, labels, M, m, dist, seed, n_components, classifier,
                 n_subsample=None):
    """Single trial: optionally subsample adapters, sample queries -> MDS -> classify."""
    rng = np.random.default_rng(seed)

    # Subsample adapters if requested
    if n_subsample is not None and n_subsample < len(labels):
        class0 = np.where(labels == 0)[0]
        class1 = np.where(labels == 1)[0]
        n_per = n_subsample // 2
        sel0 = rng.choice(class0, size=n_per, replace=False)
        sel1 = rng.choice(class1, size=n_per, replace=False)
        sel = np.sort(np.concatenate([sel0, sel1]))
        responses = responses[sel]
        labels = labels[sel]

    query_idx = sample_queries(M, m, distribution=dist, rng=rng)
    error = single_trial(responses, labels, query_idx,
                         n_components=n_components, classifier_name=classifier)
    return 1.0 - error


def run_classification(config: MotivatingConfig) -> pd.DataFrame:
    """Run the classification experiment with a-priori query partition.

    Sweeps over n_values (number of adapters) and m_values (number of queries).

    Parameters
    ----------
    config : MotivatingConfig

    Returns
    -------
    df : DataFrame with columns: n, distribution, m, mean_accuracy, std_accuracy
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
    for n in n_values:
        n_sub = n if n < n_models else None
        for dist_name, dist in distributions.items():
            dist_offset = hash(dist_name) % 10000
            desc = f"n={n}, {dist_name}"
            for m in tqdm(config.m_values, desc=desc):
                seeds = [config.seed + rep * 100003 + m * 1009 + dist_offset * 7
                         + n * 31
                         for rep in range(config.n_reps)]
                accuracies = Parallel(n_jobs=config.n_jobs, backend="loky")(
                    delayed(_run_one_rep)(
                        responses, labels, M, m, dist, s,
                        config.n_components, config.classifier, n_sub
                    )
                    for s in seeds
                )
                accuracies = np.array(accuracies)

                results.append({
                    "n": n,
                    "distribution": dist_name,
                    "m": m,
                    "mean_accuracy": accuracies.mean(),
                    "std_accuracy": accuracies.std(),
                })

    return pd.DataFrame(results)
