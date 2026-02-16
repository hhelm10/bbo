"""Exp 4: Error vs n (sample complexity).

Fix r=10, m=50 (large enough for dimension coverage), uniform Pi_Q.
Vary n to characterize gamma(n) — the finite-sample term.
Expected: P[error >= 0.5] decreases with n as kNN learns parity.
"""

import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from bbo.models.synthetic import make_problem, get_all_responses, get_labels
from bbo.queries.query_set import sample_queries
from bbo.classification.evaluate import single_trial
from bbo.experiments.config import Exp4Config


def _knn_k(n):
    """k = floor(log(n)) rounded down to nearest odd integer."""
    k = int(math.log(n))
    if k % 2 == 0:
        k -= 1
    return max(k, 1)


def _run_one_rep(problem, n_models, m, M, seed, n_components, classifier,
                 n_neighbors, eta=0.0):
    rng = np.random.default_rng(seed)
    models = problem.generate_models(n_models, eta=eta, rng=rng)
    responses = get_all_responses(models)
    labels = get_labels(models)
    query_idx = sample_queries(M, m, rng=rng)
    return single_trial(responses, labels, query_idx,
                        n_components=min(n_components, n_models - 1),
                        classifier_name=classifier,
                        n_neighbors=n_neighbors)


def run_exp4(config: Exp4Config = None) -> pd.DataFrame:
    """Run full Exp 4 sweep with parallel reps."""
    if config is None:
        config = Exp4Config()

    problem = make_problem(
        M=config.M, r=config.r, signal_prob=config.signal_prob,
        p_embed=config.p_embed, rng=np.random.default_rng(config.seed),
    )

    results = []

    for n_models in tqdm(config.n_values, desc="Exp 4: Error vs n"):
        k = _knn_k(n_models)
        n_comp = min(config.r, n_models - 1)
        seeds = [config.seed + rep * 100003 + n_models * 1009
                 for rep in range(config.n_reps)]
        errors = Parallel(n_jobs=config.n_jobs, backend="loky")(
            delayed(_run_one_rep)(problem, n_models, config.m, config.M, s,
                                  n_comp, config.classifier, k,
                                  config.eta)
            for s in seeds
        )
        errors = np.array(errors)

        results.append({
            "n_models": n_models,
            "r": config.r,
            "m": config.m,
            "mean_error": errors.mean(),
            "std_error": errors.std(),
            "prob_high_error": (errors >= 0.5).mean(),
            "n_reps": config.n_reps,
        })

    return pd.DataFrame(results)
