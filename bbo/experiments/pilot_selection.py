"""Pilot-estimated query selection experiment.

Validates that targeted query selection using estimated loadings |α̂_{q,ℓ}|
achieves lower classification error than uniform random sampling at matched
query budget m.

Selectors:
- uniform: sample m queries uniformly at random
- uniform_signal: sample m queries uniformly from estimated signal set
- uniform_orthogonal: sample m queries uniformly from estimated zero set
- greedy: Gram-Schmidt sequential selection on loading vectors
- cv_greedy: 10-fold CV greedy buildup (full MDS+RF at each step)
- stepwise: forward stepwise regression on energy tensor

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


def select_queries_cv_greedy(responses, labels, query_pool, train_idx,
                             m_max, n_components, classifier_name, n_folds=10):
    """CV greedy: build query set one at a time, maximizing CV accuracy.

    Returns the full ordered sequence of selected query indices (into query_pool).
    """
    from sklearn.model_selection import StratifiedKFold

    train_labels = labels[train_idx]
    M_pool = len(query_pool)
    selected = []
    candidates = list(range(M_pool))

    # Pre-compute per-query squared distance contributions for train models
    # D²(i,j) = Σ_q ||g(f_i(q)) - g(f_j(q))||²
    # We can accumulate D² incrementally as we add queries
    n_train = len(train_idx)
    D_sq = np.zeros((n_train, n_train))

    # Pre-compute per-query distance contributions
    train_resp = responses[train_idx][:, query_pool, :]  # (n_train, M_pool, p)
    # diffs[q][i,j] = ||resp[i,q,:] - resp[j,q,:]||²
    per_q_dist = np.zeros((M_pool, n_train, n_train))
    for q in range(M_pool):
        R_q = train_resp[:, q, :]  # (n_train, p)
        diff = R_q[:, None, :] - R_q[None, :, :]  # (n_train, n_train, p)
        per_q_dist[q] = np.sum(diff ** 2, axis=-1)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    folds = list(skf.split(np.zeros(n_train), train_labels))

    # Pre-compute loading magnitudes for candidate filtering
    resp_flat = train_resp.reshape(n_train, M_pool, -1)
    var_per_q = np.var(resp_flat, axis=0).sum(axis=1)  # (M_pool,)

    for step in range(min(m_max, M_pool)):
        # Subsample candidates to top 50 by variance (speeds up search)
        if len(candidates) > 50:
            cand_vars = [(q, var_per_q[q]) for q in candidates]
            cand_vars.sort(key=lambda x: -x[1])
            eval_candidates = [q for q, _ in cand_vars[:50]]
        else:
            eval_candidates = candidates

        best_score = -1
        best_q = eval_candidates[0]

        for q in eval_candidates:
            # Trial D² = current D² + this query's contribution
            D_sq_trial = D_sq + per_q_dist[q]
            D_trial = np.sqrt(np.maximum(D_sq_trial, 0))

            # MDS on train set
            try:
                mds = ClassicalMDS(n_components=min(n_components, n_train - 1))
                X = mds.fit_transform(D_trial)
            except Exception:
                continue

            # K-fold CV accuracy
            correct = 0
            total = 0
            for fold_train, fold_test in folds:
                clf = make_classifier(classifier_name)
                clf.fit(X[fold_train], train_labels[fold_train])
                preds = clf.predict(X[fold_test])
                correct += (preds == train_labels[fold_test]).sum()
                total += len(fold_test)

            score = correct / total if total > 0 else 0
            if score > best_score:
                best_score = score
                best_q = q

        selected.append(best_q)
        candidates.remove(best_q)
        D_sq += per_q_dist[best_q]

    return np.array(selected)


def select_queries_stepwise(E, pairs, labels_train, m):
    """Forward stepwise regression: select queries predicting class-match.

    Response: z_k = 1[y_i != y_j] for each pair k
    Predictors: E[q, k] for each query q
    Forward selection by largest partial correlation with residual.

    Returns ordered sequence of selected query indices (into E's row space).
    """
    z = (labels_train[pairs[:, 0]] != labels_train[pairs[:, 1]]).astype(float)
    z = z - z.mean()  # center

    M = E.shape[0]
    selected = []
    remaining = set(range(M))
    residual = z.copy()

    for _ in range(min(m, M)):
        best_score = -1
        best_q = -1
        for q in remaining:
            e_q = E[q]
            denom = np.sqrt(np.dot(e_q, e_q) * np.dot(residual, residual))
            if denom < 1e-12:
                continue
            score = abs(np.dot(e_q, residual)) / denom
            if score > best_score:
                best_score = score
                best_q = q

        if best_q < 0:
            break
        selected.append(best_q)
        remaining.discard(best_q)

        # Project out selected query's contribution from residual
        e_best = E[best_q]
        norm_sq = np.dot(e_best, e_best)
        if norm_sq > 1e-12:
            residual = residual - e_best * (np.dot(residual, e_best) / norm_sq)

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
    try:
        D = pairwise_energy_distances_t0(responses, query_indices)
        n = len(labels)
        mds = ClassicalMDS(n_components=min(n_components, n - 1))
        X = mds.fit_transform(D)

        clf = make_classifier(classifier_name)
        clf.fit(X[train_idx], labels[train_idx])
        preds = clf.predict(X[test_idx])
        return (preds != labels[test_idx]).mean()
    except Exception:
        return 0.5  # chance level on failure


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
        elif sel_name == "stepwise":
            train_labels = labels[train_idx]
            seq = select_queries_stepwise(E, pairs, train_labels, m)
            pool_idx = seq[:m]
        elif sel_name == "cv_greedy":
            seq = select_queries_cv_greedy(
                responses, labels, query_pool, train_idx, m,
                n_components, classifier_name)
            pool_idx = seq[:m]
        else:
            raise ValueError(f"Unknown selector: {sel_name}")

        # Map back to original query indices
        qi = query_pool[pool_idx]

        err = _single_trial(responses, labels, qi, train_idx, test_idx,
                            n_components, classifier_name)
        results[sel_name] = err

    return results


def _run_one_split_all_m(responses, labels, query_pool, n_true_signal,
                         m_values, train_idx, test_idx, n_components,
                         classifier_name, seed, selectors):
    """One split: compute all selectors across all m values.

    Sequence selectors (cv_greedy, stepwise) build once, evaluate at each m.
    Random selectors (uniform, uniform_signal) draw fresh for each m.
    """
    rng = np.random.default_rng(seed)
    M_pool = len(query_pool)
    m_max = max(m_values)

    # Estimate loadings from train set
    train_resp = responses[train_idx][:, query_pool, :]
    E, pairs = per_query_energy_tensor(train_resp)
    r_hat, U, s = estimate_discriminative_rank(E)
    rho_hats, gmm_info = estimate_rho(U, r_hat)
    signal_idx, ortho_idx = _get_signal_orthogonal_sets(U, r_hat, gmm_info)

    # Pre-compute sequences for deterministic selectors
    # Cap cv_greedy at 20 steps (expensive; interesting range is small m)
    sequences = {}
    if "stepwise" in selectors:
        train_labels = labels[train_idx]
        sequences["stepwise"] = select_queries_stepwise(E, pairs, train_labels, m_max)
    if "cv_greedy" in selectors:
        cv_max = min(m_max, 20)
        sequences["cv_greedy"] = select_queries_cv_greedy(
            responses, labels, query_pool, train_idx, cv_max,
            n_components, classifier_name)

    results = {}  # {(sel_name, m): error}
    for m in m_values:
        for sel_name in selectors:
            if sel_name == "uniform":
                pool_idx = rng.choice(M_pool, size=m, replace=False)
            elif sel_name == "uniform_signal":
                if len(signal_idx) >= m:
                    pool_idx = rng.choice(signal_idx, size=m, replace=False)
                else:
                    pool_idx = signal_idx.copy()
            elif sel_name == "stepwise":
                pool_idx = sequences["stepwise"][:m]
            elif sel_name == "cv_greedy":
                seq = sequences["cv_greedy"]
                if m > len(seq):
                    results[(sel_name, m)] = np.nan
                    continue
                pool_idx = seq[:m]
            elif sel_name == "uniform_orthogonal":
                if len(ortho_idx) >= m:
                    pool_idx = rng.choice(ortho_idx, size=m, replace=False)
                else:
                    pool_idx = ortho_idx.copy()
            elif sel_name == "oracle_signal":
                oracle_sig = np.arange(n_true_signal)
                pool_idx = rng.choice(oracle_sig, size=min(m, len(oracle_sig)), replace=False)
            elif sel_name == "oracle_orthogonal":
                oracle_orth = np.arange(n_true_signal, M_pool)
                pool_idx = rng.choice(oracle_orth, size=min(m, len(oracle_orth)), replace=False)
            else:
                raise ValueError(f"Unknown selector: {sel_name}")

            qi = query_pool[pool_idx]
            err = _single_trial(responses, labels, qi, train_idx, test_idx,
                                n_components, classifier_name)
            results[(sel_name, m)] = err

    return results


def run_pilot_experiment(
    responses, labels,
    query_pool=None,
    n_true_signal=None,
    n_train_values=(20, 80),
    m_values=(2, 5, 10, 20, 50, 100),
    selectors=("uniform", "uniform_signal", "stepwise", "cv_greedy"),
    n_reps=100,
    n_components=8,
    classifier="rf",
    seed=42,
    n_jobs=-1,
):
    """Run the pilot query selection experiment.

    Each rep uses a different train/test split. Sequence selectors
    (cv_greedy, stepwise) are computed once per split. Random selectors
    get one draw per split.

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

        # Generate splits
        splits = []
        for rep in range(n_reps):
            rng = np.random.default_rng(seed + rep * 100003 + n_train * 31)
            sel0 = rng.choice(class0, size=n_per_class_train, replace=False)
            sel1 = rng.choice(class1, size=n_per_class_train, replace=False)
            train_idx = np.concatenate([sel0, sel1])
            test_idx = np.setdiff1d(np.arange(n_models), train_idx)
            splits.append((train_idx, test_idx))

        # Run all splits (each handles all m values)
        print(f"  Running {n_reps} splits for n_train={n_train}...")
        split_results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_run_one_split_all_m)(
                responses, labels, query_pool, n_true_signal,
                m_values, train_idx, test_idx,
                n_components, classifier,
                seed + rep * 100003 + n_train * 31,
                selectors,
            )
            for rep, (train_idx, test_idx) in enumerate(
                tqdm(splits, desc=f"n_train={n_train}"))
        )

        # Aggregate
        for sel_name in selectors:
            for m in m_values:
                errors = np.array([r[(sel_name, m)] for r in split_results])
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
