"""Synthetic estimation validation experiments (bottom row of Figure 2).

(e) Rank recovery: P[r̂ = r] vs m
(f) Zero-set probability recovery: ρ̂ vs m
(g) Predicted vs empirical failure probability

Also: (c) P[fail] vs m for varying n (top row panel c).
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from bbo.models.synthetic import make_problem, get_all_responses, get_labels
from bbo.queries.query_set import sample_queries
from bbo.queries.distributions import SubsetDistribution, UniformDistribution
from bbo.classification.evaluate import single_trial
from bbo.distances.energy import per_query_energy_tensor
from bbo.estimation.rank_rho import (
    compute_E_disc, estimate_discriminative_rank, estimate_rho,
)


def _one_rep_classify(responses, labels, M, m, seed, n_components, classifier,
                      distribution=None):
    """Single classification trial. Returns error >= 0.5 indicator."""
    rng = np.random.default_rng(seed)
    query_idx = sample_queries(M, m, distribution=distribution, rng=rng)
    err = single_trial(responses, labels, query_idx,
                       n_components=n_components, classifier_name=classifier,
                       seed=seed)
    return int(err >= 0.5)


def _one_rep_estimate(responses, labels, M, m, seed, r_true=None):
    """Single estimation trial. Returns (r_hat, rho_hats).

    If r_true is provided, uses it for GMM (oracle rank) instead of estimated r̂.
    """
    rng = np.random.default_rng(seed)
    query_idx = sample_queries(M, m, rng=rng)

    resp_sub = responses[:, query_idx, :]
    E, pairs = per_query_energy_tensor(resp_sub)

    try:
        r_hat, U, s = estimate_discriminative_rank(E)
        r_for_rho = r_true if r_true is not None else r_hat
        rho_hats, _ = estimate_rho(U, r_for_rho)
    except (ValueError, np.linalg.LinAlgError):
        r_hat = 1
        rho_hats = np.array([1.0])

    return r_hat, rho_hats


def _one_rep_estimate_and_classify(responses, labels, M, m, seed,
                                   n_components, classifier, r_true=None):
    """Single trial: estimate parameters AND classify. Returns (fail, r_hat, rho_hat)."""
    rng = np.random.default_rng(seed)
    query_idx = sample_queries(M, m, rng=rng)

    # Estimation — use oracle r for ρ̂ if provided
    resp_sub = responses[:, query_idx, :]
    E, pairs = per_query_energy_tensor(resp_sub)

    try:
        r_hat, U, s = estimate_discriminative_rank(E)
        r_for_rho = r_true if r_true is not None else r_hat
        rho_hats, _ = estimate_rho(U, r_for_rho)
    except (ValueError, np.linalg.LinAlgError):
        r_hat = 1
        rho_hats = np.array([1.0])

    # Classification
    err = single_trial(responses, labels, query_idx,
                       n_components=n_components, classifier_name=classifier,
                       seed=seed)

    return int(err >= 0.5), r_hat, rho_hats


def run_panel_c(m_values=(1, 2, 5, 10, 20, 50, 100),
                n_values=(20, 50, 100),
                r=5, signal_prob=0.3, M=100, p_embed=20,
                n_reps=1000, seed=42, n_jobs=-1):
    """Panel (c): P[fail] vs m for varying n."""
    problem = make_problem(M=M, r=r, signal_prob=signal_prob, p_embed=p_embed,
                           rng=np.random.default_rng(seed))
    n_comp = min(r, max(n_values) - 1)

    results = []
    for n in n_values:
        models = problem.generate_models(n, rng=np.random.default_rng(seed + 1))
        responses = get_all_responses(models)
        labels = get_labels(models)

        for m in tqdm(m_values, desc=f"panel_c n={n}"):
            seeds = [seed + rep * 100003 + m * 1009 + n * 31
                     for rep in range(n_reps)]
            fails = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_one_rep_classify)(
                    responses, labels, M, m, s, n_comp, "rf"
                ) for s in seeds
            )
            results.append({
                "n_models": n, "m": m,
                "prob_high_error": np.mean(fails),
                "rho": 1 - signal_prob, "r": r, "n_reps": n_reps,
            })
            print(f"  n={n}, m={m}: P[fail]={np.mean(fails):.3f}")

    return pd.DataFrame(results)


def run_panel_e(m_values=(5, 10, 20, 50, 100),
                r_values=(1, 2, 5),
                signal_prob=0.3, n_models=100, M=100, p_embed=20,
                n_reps=1000, seed=42, n_jobs=-1):
    """Panel (e): P[r̂ = r] vs m for varying r."""
    results = []
    for r in r_values:
        problem = make_problem(M=M, r=r, signal_prob=signal_prob,
                               p_embed=max(p_embed, r),
                               rng=np.random.default_rng(seed))
        models = problem.generate_models(n_models,
                                          rng=np.random.default_rng(seed + 1))
        responses = get_all_responses(models)
        labels = get_labels(models)

        for m in tqdm(m_values, desc=f"panel_e r={r}"):
            seeds = [seed + rep * 100003 + m * 1009 + r * 31
                     for rep in range(n_reps)]
            est = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_one_rep_estimate)(
                    responses, labels, M, m, s
                ) for s in seeds
            )
            r_hats = np.array([e[0] for e in est])
            p_correct = np.mean(r_hats == r)
            results.append({
                "r": r, "m": m,
                "p_correct_rank": p_correct,
                "n_reps": n_reps,
            })
            print(f"  r={r}, m={m}: P[r̂=r]={p_correct:.3f}")

    return pd.DataFrame(results)


def run_panel_f(m_values=(5, 10, 20, 50, 100),
                rho_values=(0.3, 0.5, 0.7),
                r=1, n_models=100, M=100, p_embed=20,
                n_reps=1000, seed=42, n_jobs=-1):
    """Panel (f): ρ̂ vs m for varying ρ, conditioned on true r."""
    results = []
    for rho in rho_values:
        signal_prob = 1 - rho
        problem = make_problem(M=M, r=r, signal_prob=signal_prob,
                               p_embed=p_embed,
                               rng=np.random.default_rng(seed))
        models = problem.generate_models(n_models,
                                          rng=np.random.default_rng(seed + 1))
        responses = get_all_responses(models)
        labels = get_labels(models)

        for m in tqdm(m_values, desc=f"panel_f rho={rho}"):
            seeds = [seed + rep * 100003 + m * 1009 + int(rho * 1000)
                     for rep in range(n_reps)]
            est = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_one_rep_estimate)(
                    responses, labels, M, m, s, r_true=r
                ) for s in seeds
            )
            # Use first direction's rho_hat (strongest signal direction)
            rho_hats_all = np.array([e[1][0] if len(e[1]) > 0 else np.nan for e in est])
            rho_errors = np.abs(rho_hats_all - rho)
            results.append({
                "rho_true": rho, "m": m,
                "rho_hat_mean": np.nanmean(rho_hats_all),
                "mean_rho_error": np.nanmean(rho_errors),
                "std_rho_error": np.nanstd(rho_errors),
                "n_reps": n_reps,
            })
            print(f"  rho={rho}, m={m}: |ρ̂-ρ|={np.nanmean(rho_errors):.3f}")

    return pd.DataFrame(results)


def run_panel_g(m_values=(5, 10, 20, 50, 100),
                rho_values=(0.3, 0.7, 0.9),
                r=5, n_models=100, M=100, p_embed=20,
                n_reps=1000, seed=42, n_jobs=-1):
    """Panel (g): empirical P[fail] vs predicted bound using estimated ρ̂ (oracle r)."""
    results = []
    for rho in rho_values:
        signal_prob = 1 - rho
        problem = make_problem(M=M, r=r, signal_prob=signal_prob,
                               p_embed=p_embed,
                               rng=np.random.default_rng(seed))
        models = problem.generate_models(n_models,
                                          rng=np.random.default_rng(seed + 1))
        responses = get_all_responses(models)
        labels = get_labels(models)
        n_comp = min(r, n_models - 1)

        for m in tqdm(m_values, desc=f"panel_g rho={rho}"):
            seeds = [seed + rep * 100003 + m * 1009 + int(rho * 1000)
                     for rep in range(n_reps)]
            trials = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_one_rep_estimate_and_classify)(
                    responses, labels, M, m, s, n_comp, "rf",
                    r_true=r,
                ) for s in seeds
            )
            fails = [t[0] for t in trials]
            # Use first-direction ρ̂ with oracle r
            rho_hats_1 = np.array([t[2][0] if len(t[2]) > 0 else 1.0
                                   for t in trials])

            # Predicted bound using mean ρ̂: min(1, r · E[ρ̂]^m)
            rho_hat_mean = np.mean(rho_hats_1)
            predicted_bound = min(1.0, r * rho_hat_mean ** m)

            results.append({
                "rho_true": rho, "m": m,
                "prob_high_error": np.mean(fails),
                "predicted_bound_mean": predicted_bound,
                "rho_hat_mean": rho_hat_mean,
                "true_bound": r * rho ** m,
                "r": r, "n_reps": n_reps,
            })
            print(f"  rho={rho}, m={m}: P[fail]={np.mean(fails):.3f}, "
                  f"pred={predicted_bound:.3f}, "
                  f"true={r * rho ** m:.3f}")

    return pd.DataFrame(results)
