"""Exp 5: Bayes convergence (Theorem 2 validation).

Construct a problem with known L* (noisy classes), fix large m, vary n.
Expected: inf_h L -> L* as n -> infinity.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from bbo.models.synthetic import make_problem, get_all_responses, get_labels
from bbo.queries.query_set import sample_queries
from bbo.classification.evaluate import single_trial
from bbo.experiments.config import Exp5Config


def estimate_bayes_error(noise_level: float, r: int) -> float:
    """Estimate the Bayes error for a noisy synthetic problem.

    With noise_level applied per-query-per-dimension, each signal query has
    probability noise_level of giving the wrong class's token. Over many
    signal queries, the empirical distribution determines the class.
    For small noise_level and large signal sets, L* ~ noise_level.
    """
    return noise_level


def _run_one_rep(problem, n_models, m, M, seed, n_components, classifier):
    rng = np.random.default_rng(seed)
    models = problem.generate_models(n_models, rng=rng)
    responses = get_all_responses(models)
    labels = get_labels(models)
    query_idx = sample_queries(M, m, rng=rng)
    return single_trial(responses, labels, query_idx,
                        n_components=min(n_components, n_models - 1),
                        classifier_name=classifier)


def run_exp5(config: Exp5Config = None) -> pd.DataFrame:
    """Run full Exp 5 sweep with parallel reps."""
    if config is None:
        config = Exp5Config()

    bayes_error = estimate_bayes_error(config.noise_level, config.r)

    problem = make_problem(
        M=config.M, r=config.r, signal_prob=config.signal_prob,
        noise_level=config.noise_level, p=config.p,
        rng=np.random.default_rng(config.seed),
    )

    results = []

    for n_models in tqdm(config.n_values, desc="Exp 5: Bayes convergence"):
        seeds = [config.seed + rep * 100003 + n_models * 1009
                 for rep in range(config.n_reps)]
        errors = Parallel(n_jobs=config.n_jobs, backend="loky")(
            delayed(_run_one_rep)(problem, n_models, config.m, config.M, s,
                                  config.n_components, config.classifier)
            for s in seeds
        )
        errors = np.array(errors)

        results.append({
            "n_models": n_models,
            "m": config.m,
            "mean_error": errors.mean(),
            "std_error": errors.std(),
            "bayes_error": bayes_error,
            "n_reps": config.n_reps,
        })

    return pd.DataFrame(results)
