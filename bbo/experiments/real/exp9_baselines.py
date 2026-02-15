"""Exp 9: Comparison with baselines.

Compare MDS embedding classification against:
- Raw feature baseline (concatenated embeddings without MDS)
- Single-query classifiers (best individual query)
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from bbo.queries.query_set import sample_queries
from bbo.classification.evaluate import single_trial, classify_and_evaluate


def _raw_feature_trial(responses, labels, query_indices, classifier_name="knn", **kw):
    """Classify using concatenated raw features (no MDS)."""
    R = responses[:, query_indices, :]  # (n, m, p)
    X = R.reshape(R.shape[0], -1)  # (n, m*p)
    return classify_and_evaluate(X, labels, classifier_name, **kw)


def _best_single_query_trial(responses, labels, classifier_name="knn", **kw):
    """Find the best single-query classifier."""
    M = responses.shape[1]
    best_error = 1.0
    for q in range(M):
        X = responses[:, q, :]  # (n, p)
        error = classify_and_evaluate(X, labels, classifier_name, **kw)
        best_error = min(best_error, error)
    return best_error


def _run_one_rep_both(responses, labels, M, m, seed, n_components, classifier_name):
    """Run both MDS and raw-feature trials for one rep."""
    rng = np.random.default_rng(seed)
    query_idx = sample_queries(M, m, rng=rng)
    mds_error = single_trial(responses, labels, query_idx,
                             n_components=n_components,
                             classifier_name=classifier_name)
    raw_error = _raw_feature_trial(responses, labels, query_idx,
                                   classifier_name=classifier_name)
    return 1.0 - mds_error, 1.0 - raw_error


def run_exp9(responses: np.ndarray, labels: np.ndarray,
             m_values: list = None, n_reps: int = 100,
             seed: int = 42, classifier_name: str = "knn",
             n_jobs: int = -1, n_components: int = 10) -> pd.DataFrame:
    """Run Exp 9 with parallel reps.

    Parameters
    ----------
    responses : ndarray of shape (n_models, M, p)
    labels : ndarray of shape (n_models,)

    Returns
    -------
    df : DataFrame with columns method, m, mean_accuracy
    """
    if m_values is None:
        m_values = [1, 2, 5, 10, 20, 50, 100]

    M = responses.shape[1]
    results = []

    # Best single query (fixed, doesn't depend on m)
    best_single_error = _best_single_query_trial(responses, labels, classifier_name)
    for m in m_values:
        results.append({
            "method": "Best single query",
            "m": m,
            "mean_accuracy": 1.0 - best_single_error,
        })

    # MDS embedding and raw features — parallel across reps
    for m in tqdm(m_values, desc="Exp 9: Baselines"):
        seeds = [seed + rep * 100003 + m * 1009
                 for rep in range(n_reps)]
        pairs = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_run_one_rep_both)(responses, labels, M, m, s,
                                       n_components, classifier_name)
            for s in seeds
        )
        mds_accs = np.array([p[0] for p in pairs])
        raw_accs = np.array([p[1] for p in pairs])

        results.append({
            "method": "MDS embedding",
            "m": m,
            "mean_accuracy": mds_accs.mean(),
        })
        results.append({
            "method": "Raw features",
            "m": m,
            "mean_accuracy": raw_accs.mean(),
        })

    return pd.DataFrame(results)
