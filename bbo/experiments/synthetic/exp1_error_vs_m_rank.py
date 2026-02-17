"""Exp 1: Error vs m for varying discriminative rank r.

Fix M=100, n=1000 models, uniform Pi_Q, p=0.3 (rho=0.7).
Sweep r in {2, 4, 6, 8}.
Expected: log P[error >= 0.5] vs m has slope log(0.7), intercept log(r).
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from bbo.models.synthetic import make_problem, get_all_responses, get_labels
from bbo.queries.query_set import sample_queries
from bbo.classification.evaluate import single_trial
from bbo.experiments.config import Exp1Config


def _run_one_rep(responses, labels, M, m, seed, n_components, classifier):
    rng = np.random.default_rng(seed)
    query_idx = sample_queries(M, m, rng=rng)
    return single_trial(responses, labels, query_idx,
                        n_components=n_components, classifier_name=classifier,
                        seed=seed)


def run_exp1(config: Exp1Config = None) -> pd.DataFrame:
    """Run full Exp 1 sweep with parallel reps."""
    if config is None:
        config = Exp1Config()

    results = []

    for r in config.r_values:
        print(f"  r={r}...")
        # Embedding dimension must be >= r for orthogonal directions
        p_embed = max(config.p_embed, r)
        problem = make_problem(
            M=config.M, r=r, signal_prob=config.signal_prob,
            p_embed=p_embed, rng=np.random.default_rng(config.seed),
        )
        models = problem.generate_models(config.n_models, eta=config.eta,
                                          rng=np.random.default_rng(config.seed + 1))
        responses = get_all_responses(models)
        labels = get_labels(models)

        # Use n_components = r so MDS rank matches the true discriminative rank
        n_comp = min(r, config.n_models - 1)

        for m in tqdm(config.m_values, desc=f"  r={r}", leave=False):
            seeds = [config.seed + rep * 100003 + m * 1009 + r * 7
                     for rep in range(config.n_reps)]
            errors = Parallel(n_jobs=config.n_jobs, backend="loky")(
                delayed(_run_one_rep)(responses, labels, config.M, m, s,
                                      n_comp, config.classifier)
                for s in seeds
            )
            errors = np.array(errors)

            results.append({
                "r": r,
                "m": m,
                "rho": problem.rho,
                "prob_high_error": (errors >= 0.5).mean(),
                "mean_error": errors.mean(),
                "n_reps": config.n_reps,
            })

    return pd.DataFrame(results)
