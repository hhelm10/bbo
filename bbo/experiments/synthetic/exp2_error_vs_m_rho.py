"""Exp 2: Error vs m for varying rho.

Fix r=5, M=100, n=200, uniform Pi_Q.
Sweep p (activation probability) in {0.1, 0.3, 0.5, 0.8}.
Slope changes as log(1-p).
"""

import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from bbo.models.synthetic import make_problem, get_all_responses, get_labels
from bbo.queries.query_set import sample_queries
from bbo.classification.evaluate import single_trial
from bbo.experiments.config import Exp2Config


def _knn_k(n):
    """k = floor(log(n)) rounded down to nearest odd integer."""
    k = int(math.log(n))
    if k % 2 == 0:
        k -= 1
    return max(k, 1)


def _run_one_rep(responses, labels, M, m, seed, n_components, classifier,
                 n_neighbors):
    rng = np.random.default_rng(seed)
    query_idx = sample_queries(M, m, rng=rng)
    return single_trial(responses, labels, query_idx,
                        n_components=n_components, classifier_name=classifier,
                        n_neighbors=n_neighbors)


def run_exp2(config: Exp2Config = None) -> pd.DataFrame:
    """Run full Exp 2 sweep with parallel reps."""
    if config is None:
        config = Exp2Config()

    k = _knn_k(config.n_models)
    # Use n_components = r so MDS rank matches the true discriminative rank
    n_comp = min(config.r, config.n_models - 1)

    results = []

    for signal_prob in config.signal_prob_values:
        rho = 1.0 - signal_prob
        print(f"  p={signal_prob}, rho={rho:.2f}...")

        problem = make_problem(
            M=config.M, r=config.r, signal_prob=signal_prob,
            sigma=config.sigma, p=config.p, rng=np.random.default_rng(config.seed),
        )
        models = problem.generate_models(config.n_models,
                                          rng=np.random.default_rng(config.seed + 1))
        responses = get_all_responses(models)
        labels = get_labels(models)

        sp_offset = int(signal_prob * 1000)
        for m in tqdm(config.m_values, desc=f"  p={signal_prob}", leave=False):
            seeds = [config.seed + rep * 100003 + m * 1009 + sp_offset * 7
                     for rep in range(config.n_reps)]
            errors = Parallel(n_jobs=config.n_jobs, backend="loky")(
                delayed(_run_one_rep)(responses, labels, config.M, m, s,
                                      n_comp, config.classifier, k)
                for s in seeds
            )
            errors = np.array(errors)

            results.append({
                "signal_prob": signal_prob,
                "rho": rho,
                "m": m,
                "prob_high_error": (errors >= 0.5).mean(),
                "mean_error": errors.mean(),
                "n_reps": config.n_reps,
            })

    return pd.DataFrame(results)
