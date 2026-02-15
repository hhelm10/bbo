"""Exp 7: Classification accuracy vs m (main real-data result).

Sweep m values, sample queries uniformly, compute MDS + classify via LOO-CV.
Expected: sigmoidal accuracy curve rising from ~0.5 to near 1-L*.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from bbo.queries.query_set import sample_queries
from bbo.classification.evaluate import single_trial
from bbo.experiments.config import Exp7Config


def _run_one_rep(responses, labels, M, m, seed, n_components, classifier):
    rng = np.random.default_rng(seed)
    query_idx = sample_queries(M, m, rng=rng)
    error = single_trial(responses, labels, query_idx,
                         n_components=n_components, classifier_name=classifier)
    return 1.0 - error


def run_exp7(responses: np.ndarray, labels: np.ndarray,
             config: Exp7Config = None) -> pd.DataFrame:
    """Run Exp 7 sweep over m values with parallel reps.

    Parameters
    ----------
    responses : ndarray of shape (n_models, M, p)
    labels : ndarray of shape (n_models,)
    config : Exp7Config

    Returns
    -------
    df : DataFrame with columns m, mean_accuracy, std_accuracy
    """
    if config is None:
        config = Exp7Config()

    M = responses.shape[1]
    results = []

    for m in tqdm(config.m_values, desc="Exp 7: Accuracy vs m"):
        seeds = [config.seed + rep * 100003 + m * 1009
                 for rep in range(config.n_reps)]
        accuracies = Parallel(n_jobs=config.n_jobs, backend="loky")(
            delayed(_run_one_rep)(responses, labels, M, m, s,
                                  config.n_components, config.classifier)
            for s in seeds
        )
        accuracies = np.array(accuracies)

        results.append({
            "m": m,
            "mean_accuracy": accuracies.mean(),
            "std_accuracy": accuracies.std(),
            "n_reps": config.n_reps,
        })

    return pd.DataFrame(results)
