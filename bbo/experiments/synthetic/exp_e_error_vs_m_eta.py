"""Exp E: Mean error vs m for varying label noise eta.

Fix r=5, n=100 models, uniform Pi_Q, p=0.3 (rho=0.7).
Sweep eta in {0.05, 0.1, 0.2}.
Expected: mean classification error converges to L* = eta as m grows.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from bbo.models.synthetic import make_problem, get_all_responses, get_labels
from bbo.queries.query_set import sample_queries
from bbo.classification.evaluate import single_trial
from bbo.experiments.config import ExpEConfig


def _run_one_rep(responses, labels, M, m, seed, n_components, classifier):
    rng = np.random.default_rng(seed)
    query_idx = sample_queries(M, m, rng=rng)
    return single_trial(responses, labels, query_idx,
                        n_components=n_components, classifier_name=classifier,
                        seed=seed)


def run_exp_e(config: ExpEConfig = None) -> pd.DataFrame:
    """Run full Exp E sweep with parallel reps."""
    if config is None:
        config = ExpEConfig()

    r = config.r
    p_embed = max(config.p_embed, r)
    problem = make_problem(
        M=config.M, r=r, signal_prob=config.signal_prob,
        p_embed=p_embed, rng=np.random.default_rng(config.seed),
    )

    results = []

    for eta in config.eta_values:
        print(f"  eta={eta}...")
        models = problem.generate_models(config.n_models, eta=eta,
                                          rng=np.random.default_rng(config.seed + 1))
        responses = get_all_responses(models)
        labels = get_labels(models)

        n_comp = min(r, config.n_models - 1)

        for m in tqdm(config.m_values, desc=f"  eta={eta}", leave=False):
            seeds = [config.seed + rep * 100003 + m * 1009 + int(eta * 10000)
                     for rep in range(config.n_reps)]
            errors = Parallel(n_jobs=config.n_jobs, backend="loky")(
                delayed(_run_one_rep)(responses, labels, config.M, m, s,
                                      n_comp, config.classifier)
                for s in seeds
            )
            errors = np.array(errors)

            results.append({
                "eta": eta,
                "m": m,
                "mean_error": errors.mean(),
                "std_error": errors.std(),
                "bayes_risk": eta,
                "n_reps": config.n_reps,
            })

    return pd.DataFrame(results)
