"""Exp F: Mean error vs n for varying label noise eta.

Fix r=5, m=50, uniform Pi_Q, p=0.3 (rho=0.7).
Sweep eta in {0.05, 0.1, 0.2} and n in {10, 20, 50, 100, 200, 500}.
Expected: mean classification error converges to L* = eta as n grows.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from bbo.models.synthetic import make_problem, get_all_responses, get_labels
from bbo.queries.query_set import sample_queries
from bbo.classification.evaluate import single_trial
from bbo.experiments.config import ExpFConfig


def _run_one_rep(problem, n_models, m, M, seed, n_components, classifier,
                 eta=0.0):
    rng = np.random.default_rng(seed)
    models = problem.generate_models(n_models, eta=eta, rng=rng)
    responses = get_all_responses(models)
    labels = get_labels(models)
    query_idx = sample_queries(M, m, rng=rng)
    return single_trial(responses, labels, query_idx,
                        n_components=min(n_components, n_models - 1),
                        classifier_name=classifier, seed=seed)


def run_exp_f(config: ExpFConfig = None) -> pd.DataFrame:
    """Run full Exp F sweep with parallel reps."""
    if config is None:
        config = ExpFConfig()

    r = config.r
    p_embed = max(config.p_embed, r)
    problem = make_problem(
        M=config.M, r=r, signal_prob=config.signal_prob,
        p_embed=p_embed, rng=np.random.default_rng(config.seed),
    )

    results = []

    for eta in config.eta_values:
        print(f"  eta={eta}...")
        for n_models in tqdm(config.n_values, desc=f"  eta={eta}", leave=False):
            n_comp = min(r, n_models - 1)
            seeds = [config.seed + rep * 100003 + n_models * 1009 + int(eta * 10000)
                     for rep in range(config.n_reps)]
            errors = Parallel(n_jobs=config.n_jobs, backend="loky")(
                delayed(_run_one_rep)(problem, n_models, config.m, config.M, s,
                                      n_comp, config.classifier, eta)
                for s in seeds
            )
            errors = np.array(errors)

            results.append({
                "eta": eta,
                "n_models": n_models,
                "m": config.m,
                "mean_error": errors.mean(),
                "std_error": errors.std(),
                "bayes_risk": eta,
                "n_reps": config.n_reps,
            })

    return pd.DataFrame(results)
