"""Exp 4: Error vs n (sample complexity).

Fix r=5, m=50 (large enough), uniform Pi_Q.
Vary n from 10 to 500.
Expected: error decreases with n, characterizing gamma(n).
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from bbo.models.synthetic import make_problem, get_all_responses, get_labels
from bbo.queries.query_set import sample_queries
from bbo.classification.evaluate import single_trial
from bbo.experiments.config import Exp4Config


def _run_one_rep(problem, n_models, m, M, seed, n_components, classifier):
    rng = np.random.default_rng(seed)
    models = problem.generate_models(n_models, rng=rng)
    responses = get_all_responses(models)
    labels = get_labels(models)
    query_idx = sample_queries(M, m, rng=rng)
    return single_trial(responses, labels, query_idx,
                        n_components=min(n_components, n_models - 1),
                        classifier_name=classifier)


def run_exp4(config: Exp4Config = None) -> pd.DataFrame:
    """Run full Exp 4 sweep with parallel reps."""
    if config is None:
        config = Exp4Config()

    problem = make_problem(
        M=config.M, r=config.r, signal_prob=config.signal_prob,
        noise_level=config.noise_level, p=config.p,
        rng=np.random.default_rng(config.seed),
    )

    results = []

    for n_models in tqdm(config.n_values, desc="Exp 4: Error vs n"):
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
            "n_reps": config.n_reps,
        })

    return pd.DataFrame(results)
