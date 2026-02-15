"""Exp 3: Effect of query distribution Pi_Q.

Fix r=5, M=100, signal_prob=0.2, n=200.
Three distributions: uniform, concentrated on signal, concentrated on Q_perp.
Expected: signal-concentrated converges fastest; orthogonal never converges.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from bbo.models.synthetic import make_problem, get_all_responses, get_labels
from bbo.queries.query_set import sample_queries
from bbo.queries.distributions import UniformDistribution, SubsetDistribution
from bbo.classification.evaluate import single_trial
from bbo.experiments.config import Exp3Config


# Map distribution names to integers for seed hashing
_DIST_SEED_OFFSET = {"uniform": 0, "signal": 1, "orthogonal": 2}


def _run_one_rep(responses, labels, M, m, dist, seed, n_components, classifier):
    rng = np.random.default_rng(seed)
    query_idx = sample_queries(M, m, distribution=dist, rng=rng)
    return single_trial(responses, labels, query_idx,
                        n_components=n_components, classifier_name=classifier)


def run_exp3(config: Exp3Config = None) -> pd.DataFrame:
    """Run full Exp 3 sweep with parallel reps."""
    if config is None:
        config = Exp3Config()

    problem = make_problem(
        M=config.M, r=config.r, signal_prob=config.signal_prob,
        noise_level=config.noise_level, p=config.p,
        rng=np.random.default_rng(config.seed),
    )
    models = problem.generate_models(config.n_models,
                                      rng=np.random.default_rng(config.seed + 1))
    responses = get_all_responses(models)
    labels = get_labels(models)

    signal_idx = problem.signal_queries
    orth_idx = problem.orthogonal_queries

    distributions = {
        "uniform": UniformDistribution(),
        "signal": SubsetDistribution(signal_idx, config.concentration),
        "orthogonal": SubsetDistribution(orth_idx, config.concentration),
    }

    results = []

    for dist_name, dist in distributions.items():
        dist_offset = _DIST_SEED_OFFSET[dist_name]
        print(f"  distribution={dist_name}...")

        for m in tqdm(config.m_values, desc=f"  {dist_name}", leave=False):
            seeds = [config.seed + rep * 100003 + m * 1009 + dist_offset * 7
                     for rep in range(config.n_reps)]
            errors = Parallel(n_jobs=config.n_jobs, backend="loky")(
                delayed(_run_one_rep)(responses, labels, config.M, m, dist, s,
                                      config.n_components, config.classifier)
                for s in seeds
            )
            errors = np.array(errors)

            results.append({
                "distribution": dist_name,
                "m": m,
                "mean_error": errors.mean(),
                "accuracy": 1.0 - errors.mean(),
                "prob_high_error": (errors >= 0.5).mean(),
                "n_reps": config.n_reps,
            })

    return pd.DataFrame(results)
