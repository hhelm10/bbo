"""Exp 4: Error vs n (sample complexity).

Fix r=5, vary m and n to show query-sample complexity interplay.
For small m, rho^m is large so error plateaus even as n -> inf.
For large m, rho^m ~ 0 and error converges to 0 as gamma(n) -> 0.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from bbo.models.synthetic import make_problem, get_all_responses, get_labels
from bbo.queries.query_set import sample_queries
from bbo.classification.evaluate import single_trial
from bbo.experiments.config import Exp4Config


def _run_one_rep(problem, n_models, m, M, seed, n_components, classifier,
                 eta=0.0):
    rng = np.random.default_rng(seed)
    models = problem.generate_models(n_models, eta=eta, rng=rng)
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

    r = config.r
    p_embed = max(config.p_embed, r)
    problem = make_problem(
        M=config.M, r=r, signal_prob=config.signal_prob,
        p_embed=p_embed, rng=np.random.default_rng(config.seed),
    )

    results = []

    for m in config.m_values:
        print(f"  m={m}...")
        for n_models in tqdm(config.n_values, desc=f"  m={m}", leave=False):
            n_comp = min(r, n_models - 1)
            seeds = [config.seed + rep * 100003 + n_models * 1009 + m * 7
                     for rep in range(config.n_reps)]
            errors = Parallel(n_jobs=config.n_jobs, backend="loky")(
                delayed(_run_one_rep)(problem, n_models, m, config.M, s,
                                      n_comp, config.classifier,
                                      config.eta)
                for s in seeds
            )
            errors = np.array(errors)

            results.append({
                "r": r,
                "n_models": n_models,
                "m": m,
                "mean_error": errors.mean(),
                "std_error": errors.std(),
                "prob_high_error": (errors >= 0.5).mean(),
                "n_reps": config.n_reps,
            })

    return pd.DataFrame(results)
