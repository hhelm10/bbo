"""Exp 8: Relevant vs orthogonal queries.

Compare classification accuracy when drawing queries from relevant, orthogonal,
or uniform subsets.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from bbo.queries.query_set import sample_queries
from bbo.queries.distributions import SubsetDistribution, UniformDistribution
from bbo.classification.evaluate import single_trial
from bbo.experiments.real.data_loader import partition_queries_by_relevance


def _run_one_rep(responses, labels, M, m, dist, seed, n_components, classifier):
    rng = np.random.default_rng(seed)
    query_idx = sample_queries(M, m, distribution=dist, rng=rng)
    error = single_trial(responses, labels, query_idx,
                         n_components=n_components, classifier_name=classifier)
    return 1.0 - error


def run_exp8(responses: np.ndarray, labels: np.ndarray,
             m_values: list = None, n_reps: int = 100,
             seed: int = 42, n_jobs: int = -1,
             n_components=None, classifier_name: str = "knn") -> pd.DataFrame:
    """Run Exp 8 with parallel reps.

    Parameters
    ----------
    responses : ndarray of shape (n_models, M, p)
    labels : ndarray of shape (n_models,)
    m_values : list of int
    n_reps : int
    seed : int
    n_jobs : int

    Returns
    -------
    df : DataFrame
    """
    if m_values is None:
        m_values = [1, 2, 5, 10, 20, 50, 100]

    M = responses.shape[1]
    relevant_idx, orthogonal_idx = partition_queries_by_relevance(responses, labels)

    distributions = {
        "uniform": UniformDistribution(),
        "relevant": SubsetDistribution(relevant_idx, 0.95),
        "orthogonal": SubsetDistribution(orthogonal_idx, 0.95),
    }

    results = []
    for dist_name, dist in distributions.items():
        dist_offset = hash(dist_name) % 10000
        for m in tqdm(m_values, desc=f"Exp 8 ({dist_name})"):
            seeds = [seed + rep * 100003 + m * 1009 + dist_offset * 7
                     for rep in range(n_reps)]
            accuracies = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_run_one_rep)(responses, labels, M, m, dist, s,
                                      n_components, classifier_name)
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
