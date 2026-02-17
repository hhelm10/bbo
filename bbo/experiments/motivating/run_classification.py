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


def _run_one_rep(responses, labels, M, m, dist, seed, n_components, classifier):
    """Single trial: sample queries -> MDS -> classify -> accuracy."""
    rng = np.random.default_rng(seed)
    query_idx = sample_queries(M, m, distribution=dist, rng=rng)
    error = single_trial(responses, labels, query_idx,
                         n_components=n_components, classifier_name=classifier)
    return 1.0 - error


def run_classification(config: MotivatingConfig) -> pd.DataFrame:
    """Run the classification experiment with a-priori query partition.

    Parameters
    ----------
    config : MotivatingConfig

    Returns
    -------
    df : DataFrame with columns: distribution, m, mean_accuracy, std_accuracy
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

    # A-priori distributions (mass=1.0 for pure subset sampling)
    # Keys match plot_exp8() expectations: "relevant", "orthogonal", "uniform"
    distributions = {
        "relevant": SubsetDistribution(sensitive_indices, mass=1.0),
        "orthogonal": SubsetDistribution(orthogonal_indices, mass=1.0),
        "uniform": UniformDistribution(),
    }

    results = []
    for dist_name, dist in distributions.items():
        dist_offset = hash(dist_name) % 10000
        for m in tqdm(config.m_values, desc=f"Classification ({dist_name})"):
            seeds = [config.seed + rep * 100003 + m * 1009 + dist_offset * 7
                     for rep in range(config.n_reps)]
            accuracies = Parallel(n_jobs=config.n_jobs, backend="loky")(
                delayed(_run_one_rep)(
                    responses, labels, M, m, dist, s,
                    config.n_components, config.classifier
                )
                for s in seeds
            )
            accuracies = np.array(accuracies)

            results.append({
                "distribution": dist_name,
                "m": m,
                "mean_accuracy": accuracies.mean(),
                "std_accuracy": accuracies.std(),
            })

    return pd.DataFrame(results)
