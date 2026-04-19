"""Pilot-estimated query selection experiment.

Validates that targeted query selection using estimated loadings |α̂_{q,ℓ}|
achieves lower classification error than uniform random sampling at matched
query budget m.

Selectors:
- uniform: sample m queries uniformly at random
- uniform_signal: sample m queries uniformly from estimated signal set
- uniform_orthogonal: sample m queries uniformly from estimated zero set
- greedy: Gram-Schmidt sequential selection on loading vectors

The train set serves double duty: estimation (pilot) and classifier fitting.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from bbo.distances.energy import pairwise_energy_distances_t0, per_query_energy_tensor
from bbo.embedding.mds import ClassicalMDS
from bbo.estimation.rank_rho import (
    compute_E_disc, estimate_discriminative_rank, estimate_rho,
)
from bbo.classification.evaluate import make_classifier


def select_queries_greedy(U, r_hat, m):
    """Gram-Schmidt greedy selection: pick queries with maximum residual signal.

    At each step, selects the query with the largest residual loading norm,
    then projects out its contribution so the next pick is decorrelated.
    """
    R = np.abs(U[:, :r_hat]).copy()  # (M, r_hat) residual loadings
    M = R.shape[0]
    selected = []

    for _ in range(min(m, M)):
        scores = np.linalg.norm(R, axis=1)
        scores[selected] = -1  # exclude already selected
        q_star = int(np.argmax(scores))
        selected.append(q_star)

        # Project out q_star's direction from all residuals
        v = R[q_star]
        norm_sq = np.dot(v, v)
        if norm_sq > 1e-12:
            R = R - np.outer(R @ v, v) / norm_sq

    return np.array(selected)


def _get_signal_orthogonal_sets(U, r_hat, gmm_info):
    """Partition queries into estimated signal and zero sets using GMM labels."""
    # A query is "signal" if it's active (label=1) in ANY direction
    M = U.shape[0]
    is_signal = np.zeros(M, dtype=bool)
    for ell in range(r_hat):
        dir_labels = gmm_info["per_direction"][ell]["labels"]
        is_signal |= (dir_labels == 1)

    signal_idx = np.where(is_signal)[0]
    ortho_idx = np.where(~is_signal)[0]
    return signal_idx, ortho_idx


def _single_trial(responses, labels, query_indices, train_idx, test_idx,
                  n_components, classifier_name):
    """Classify with fixed queries on a given train/test split."""
    D = pairwise_energy_distances_t0(responses, query_indices)
    n = len(labels)
    mds = ClassicalMDS(n_components=min(n_components, n - 1))
    X = mds.fit_transform(D)

    clf = make_classifier(classifier_name)
    clf.fit(X[train_idx], labels[train_idx])
    preds = clf.predict(X[test_idx])
    return (preds != labels[test_idx]).mean()


def _run_one_rep(responses, labels, query_pool, n_true_signal, m, train_idx,
                 test_idx, n_components, classifier_name, seed, selectors):
    """One repetition: estimate loadings from train, run all selectors.

    query_pool : ndarray of query indices to use for estimation and selection.
        First n_true_signal entries are true signal queries.
    n_true_signal : int
        Number of true signal queries at the start of query_pool.
    """
    rng = np.random.default_rng(seed)
    M_pool = len(query_pool)

    # Estimate loadings from train set on query_pool only
    train_resp = responses[train_idx][:, query_pool, :]
    E, pairs = per_query_energy_tensor(train_resp)
    # Use raw E (not E_disc) for query selection
    r_hat, U, s = estimate_discriminative_rank(E)
    rho_hats, gmm_info = estimate_rho(U, r_hat)

    # Partition queries into estimated signal / orthogonal (indices into query_pool)
    signal_idx, ortho_idx = _get_signal_orthogonal_sets(U, r_hat, gmm_info)

    results = {}
    for sel_name in selectors:
        if sel_name == "uniform":
            pool_idx = rng.choice(M_pool, size=m, replace=False)
        elif sel_name == "uniform_signal":
            if len(signal_idx) >= m:
                pool_idx = rng.choice(signal_idx, size=m, replace=False)
            else:
                pool_idx = signal_idx.copy()
        elif sel_name == "uniform_orthogonal":
            if len(ortho_idx) >= m:
                pool_idx = rng.choice(ortho_idx, size=m, replace=False)
            else:
                pool_idx = ortho_idx.copy()
        elif sel_name == "greedy":
            pool_idx = select_queries_greedy(U, r_hat, m)
        elif sel_name == "top_k":
            magnitudes = np.linalg.norm(np.abs(U[:, :r_hat]), axis=1)
            pool_idx = np.argsort(magnitudes)[::-1][:m]
        elif sel_name == "oracle_signal":
            oracle_sig = np.arange(n_true_signal)
            if len(oracle_sig) >= m:
                pool_idx = rng.choice(oracle_sig, size=m, replace=False)
            else:
                pool_idx = oracle_sig.copy()
        elif sel_name == "oracle_orthogonal":
            oracle_orth = np.arange(n_true_signal, M_pool)
            if len(oracle_orth) >= m:
                pool_idx = rng.choice(oracle_orth, size=m, replace=False)
            else:
                pool_idx = oracle_orth.copy()
        else:
            raise ValueError(f"Unknown selector: {sel_name}")

        # Map back to original query indices
        qi = query_pool[pool_idx]

        err = _single_trial(responses, labels, qi, train_idx, test_idx,
                            n_components, classifier_name)
        results[sel_name] = err

    return results


def run_pilot_experiment(
    responses, labels,
    query_pool=None,
    n_true_signal=None,
    n_train_values=(20, 80),
    m_values=(2, 5, 10, 20, 50, 100),
    selectors=("uniform", "uniform_signal", "uniform_orthogonal", "greedy"),
    n_reps=500,
    n_components=8,
    classifier="rf",
    seed=42,
    n_jobs=-1,
):
    """Run the pilot query selection experiment.

    Parameters
    ----------
    responses : ndarray of shape (n_models, M, p)
    labels : ndarray of shape (n_models,)
    query_pool : ndarray of int, optional
        Indices of queries to use for estimation and selection.
        If None, uses all M queries.
    n_train_values : sequence of int
    m_values : sequence of int
    selectors : sequence of str
    n_reps, n_components, classifier, seed, n_jobs : ...

    Returns
    -------
    df : DataFrame with columns: n_train, m, selector, mean_error, std_error
    """
    n_models, M, p = responses.shape
    if query_pool is None:
        query_pool = np.arange(M)
    if n_true_signal is None:
        n_true_signal = len(query_pool) // 2
    class0 = np.where(labels == 0)[0]
    class1 = np.where(labels == 1)[0]

    all_results = []

    for n_train in n_train_values:
        n_per_class_train = n_train // 2

        for m in tqdm(m_values, desc=f"n_train={n_train}"):
            # Generate train/test splits
            splits = []
            for rep in range(n_reps):
                rng = np.random.default_rng(seed + rep * 100003 + n_train * 31)
                sel0 = rng.choice(class0, size=n_per_class_train, replace=False)
                sel1 = rng.choice(class1, size=n_per_class_train, replace=False)
                train_idx = np.concatenate([sel0, sel1])
                test_idx = np.setdiff1d(np.arange(n_models), train_idx)
                splits.append((train_idx, test_idx))

            # Run all reps in parallel
            rep_results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_run_one_rep)(
                    responses, labels, query_pool, n_true_signal, m,
                    train_idx, test_idx,
                    n_components, classifier,
                    seed + rep * 100003 + n_train * 31 + m * 1009,
                    selectors,
                )
                for rep, (train_idx, test_idx) in enumerate(splits)
            )

            # Aggregate per selector
            for sel_name in selectors:
                errors = np.array([r[sel_name] for r in rep_results])
                all_results.append({
                    "n_train": n_train,
                    "m": m,
                    "selector": sel_name,
                    "mean_error": errors.mean(),
                    "std_error": errors.std(),
                })
                print(f"  n={n_train}, m={m}, {sel_name}: "
                      f"err={errors.mean():.3f} ± {errors.std():.3f}")

    return pd.DataFrame(all_results)
