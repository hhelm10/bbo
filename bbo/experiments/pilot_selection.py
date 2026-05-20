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


def select_queries_backward(E, pairs, labels_train, m):
    """Backward elimination: start with all queries, remove least useful.

    Uses partial-correlation approach for efficiency: at each step, orthogonalize
    all selected queries against each candidate-for-removal, then drop the one
    with the smallest partial correlation with z.

    Returns (remaining_indices, elimination_order).
    """
    z = (labels_train[pairs[:, 0]] != labels_train[pairs[:, 1]]).astype(float)
    z = z - z.mean()

    M = E.shape[0]
    selected = list(range(M))
    elimination_order = []

    # Work with copies we can orthogonalize
    E_work = E.copy().astype(float)

    while len(selected) > m:
        # For each selected query, compute how much R² drops if we remove it.
        # The query whose removal costs least has the smallest |partial corr|
        # with z after accounting for all other selected queries.
        #
        # Efficient: for query j in selected, its partial correlation with z
        # given all others equals its correlation with the residual of z
        # regressed on all others. But computing this for each j is expensive.
        #
        # Instead: use the OLS t-statistic via (X^T X)^{-1}.
        # For large sets, use a chunked approach.
        n_sel = len(selected)

        if n_sel <= 50:
            # Direct OLS approach for small sets
            X = E_work[selected].T
            try:
                XtX = X.T @ X
                Xtz = X.T @ z
                beta = np.linalg.solve(XtX + 1e-10 * np.eye(n_sel), Xtz)
                XtX_inv_diag = np.diag(np.linalg.inv(XtX + 1e-10 * np.eye(n_sel)))
                scores = beta ** 2 / (XtX_inv_diag + 1e-12)
            except np.linalg.LinAlgError:
                scores = np.array([np.dot(E_work[q], E_work[q]) for q in selected])
        else:
            # For large sets, approximate: score each query by its squared
            # correlation with z (ignoring interactions). This is O(M) per step.
            scores = np.array([
                (np.dot(E_work[q], z) ** 2) / (np.dot(E_work[q], E_work[q]) + 1e-12)
                for q in selected
            ])

        worst_idx = int(np.argmin(scores))
        worst_q = selected[worst_idx]
        elimination_order.append(worst_q)
        selected.pop(worst_idx)

    return np.array(selected), elimination_order


def select_queries_bidirectional(E, pairs, labels_train, m):
    """Bidirectional stepwise: forward selection with backward pruning.

    Like forward selection, but after each addition, check if any previously
    selected query can be removed without increasing residual SS.
    Uses the same partial-correlation framework as forward for efficiency.
    """
    z = (labels_train[pairs[:, 0]] != labels_train[pairs[:, 1]]).astype(float)
    z = z - z.mean()

    M = E.shape[0]
    selected = []
    remaining = set(range(M))
    residual = z.copy()

    for step in range(min(m, M)):
        # --- Forward step: add query with largest |partial corr| with residual ---
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

        # Update residual: project out new query
        e_best = E[best_q]
        norm_sq = np.dot(e_best, e_best)
        if norm_sq > 1e-12:
            residual = residual - e_best * (np.dot(residual, e_best) / norm_sq)

        # --- Backward step: check if any selected query can be dropped ---
        if len(selected) > 1:
            # Recompute residual from scratch for each leave-one-out
            worst_score = np.inf
            worst_idx = -1
            for i, q in enumerate(selected):
                # Compute R² without query q
                others = selected[:i] + selected[i+1:]
                X = E[others].T
                try:
                    beta = np.linalg.lstsq(X, z, rcond=None)[0]
                    resid = z - X @ beta
                    ss = np.dot(resid, resid)
                except np.linalg.LinAlgError:
                    ss = np.dot(z, z)
                # The query whose removal causes the smallest SS increase is weakest
                if ss < worst_score:
                    worst_score = ss
                    worst_idx = i

            # Check if removing worst is better than keeping it
            # (only if the set is larger than needed for this step)
            X_full = E[selected].T
            try:
                beta_full = np.linalg.lstsq(X_full, z, rcond=None)[0]
                resid_full = z - X_full @ beta_full
                ss_full = np.dot(resid_full, resid_full)
            except np.linalg.LinAlgError:
                ss_full = np.dot(z, z)

            # Remove if the query contributes less than 1% of total SS reduction
            ss_total = np.dot(z, z)
            contribution = worst_score - ss_full
            if contribution < 0.01 * (ss_total - ss_full) and worst_idx >= 0:
                removed_q = selected.pop(worst_idx)
                remaining.add(removed_q)
                # Recompute residual from current selected set
                if selected:
                    X = E[selected].T
                    try:
                        beta = np.linalg.lstsq(X, z, rcond=None)[0]
                        residual = z - X @ beta
                    except np.linalg.LinAlgError:
                        residual = z.copy()
                else:
                    residual = z.copy()

    # Fill to m if backward removed too many
    while len(selected) < m and remaining:
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
        e_best = E[best_q]
        norm_sq = np.dot(e_best, e_best)
        if norm_sq > 1e-12:
            residual = residual - e_best * (np.dot(residual, e_best) / norm_sq)

    return np.array(selected[:m])


def compute_pca_embeddings(responses, query_pool=None):
    """Compute PCA on stacked embeddings (nested: one SVD serves all d).

    Parameters
    ----------
    responses : ndarray of shape (n_models, M, p)
    query_pool : ndarray of query indices (optional, defaults to all)

    Returns
    -------
    Vt : ndarray of shape (min(n*M_pool, p), p) — right singular vectors
    mean : ndarray of shape (p,) — column mean used for centering
    """
    if query_pool is not None:
        R = responses[:, query_pool, :]
    else:
        R = responses
    n, M_pool, p = R.shape
    stacked = R.reshape(-1, p)  # (n*M_pool, p)
    mean = stacked.mean(axis=0)
    centered = stacked - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return Vt, mean


def _pca_reduce(responses, query_pool, Vt, mean, d):
    """Project responses to d-dimensional PCA space.

    Returns ndarray of shape (n_models, M_pool, d).
    """
    if query_pool is not None:
        R = responses[:, query_pool, :]
    else:
        R = responses
    n, M_pool, p = R.shape
    stacked = R.reshape(-1, p) - mean
    reduced = stacked @ Vt[:d].T  # (n*M_pool, d)
    return reduced.reshape(n, M_pool, d)


def _grouped_projection_score(X_q, R):
    """Score = trace(R^T P_{X_q} R) where P = X(X^TX)^+ X^T.

    X_q : (n, d), R : (n, k). Uses lstsq for robustness when d > n.
    """
    proj_coef, _, _, _ = np.linalg.lstsq(X_q, R, rcond=None)  # (d, k)
    proj = X_q @ proj_coef  # (n, k)
    return np.sum(proj ** 2)


def select_queries_pca_bidirectional(responses_pca, y, m):
    """Bidirectional stepwise on PCA-reduced embeddings.

    Each query contributes a block of d features. Forward selection adds the
    query whose d-dimensional block most reduces residual SS; backward step
    removes any query whose contribution is < 1% of total SS reduction.

    Parameters
    ----------
    responses_pca : ndarray of shape (n, M, d)
    y : ndarray of shape (n,) or (n, k)
    m : int — target number of queries

    Returns
    -------
    ndarray of selected query indices (into responses_pca's query axis)
    """
    n, M, d = responses_pca.shape
    if y.ndim == 1:
        y = y[:, None]
    R = y - y.mean(axis=0)
    ss_total = np.sum(R ** 2)

    selected = []
    remaining = set(range(M))

    for step in range(min(m, M)):
        # --- Forward: add query with largest grouped projection score ---
        best_score = -1
        best_q = -1
        for q in remaining:
            X_q = responses_pca[:, q, :]
            score = _grouped_projection_score(X_q, R)
            if score > best_score:
                best_score = score
                best_q = q

        if best_q < 0:
            break
        selected.append(best_q)
        remaining.discard(best_q)

        # Update residual
        X_q = responses_pca[:, best_q, :]
        coef, _, _, _ = np.linalg.lstsq(X_q, R, rcond=None)
        R = R - X_q @ coef

        # --- Backward: check if any selected query can be dropped ---
        if len(selected) > 1:
            X_full = np.hstack([responses_pca[:, q, :] for q in selected])
            coef_full, _, _, _ = np.linalg.lstsq(X_full, y - y.mean(0), rcond=None)
            resid_full = (y - y.mean(0)) - X_full @ coef_full
            ss_full = np.sum(resid_full ** 2)

            worst_ss = np.inf
            worst_idx = -1
            for i in range(len(selected)):
                others = selected[:i] + selected[i+1:]
                X_others = np.hstack([responses_pca[:, q, :] for q in others])
                coef_o, _, _, _ = np.linalg.lstsq(X_others, y - y.mean(0), rcond=None)
                resid_o = (y - y.mean(0)) - X_others @ coef_o
                ss_o = np.sum(resid_o ** 2)
                if ss_o < worst_ss:
                    worst_ss = ss_o
                    worst_idx = i

            contribution = worst_ss - ss_full
            if contribution < 0.01 * (ss_total - ss_full) and worst_idx >= 0:
                removed_q = selected.pop(worst_idx)
                remaining.add(removed_q)
                # Recompute residual
                if selected:
                    X_sel = np.hstack([responses_pca[:, q, :] for q in selected])
                    coef_s, _, _, _ = np.linalg.lstsq(X_sel, y - y.mean(0), rcond=None)
                    R = (y - y.mean(0)) - X_sel @ coef_s
                else:
                    R = y - y.mean(axis=0)

    # Fill to m if backward removed too many
    while len(selected) < m and remaining:
        best_score = -1
        best_q = -1
        for q in remaining:
            X_q = responses_pca[:, q, :]
            score = _grouped_projection_score(X_q, R)
            if score > best_score:
                best_score = score
                best_q = q
        if best_q < 0:
            break
        selected.append(best_q)
        remaining.discard(best_q)
        X_q = responses_pca[:, best_q, :]
        coef, _, _, _ = np.linalg.lstsq(X_q, R, rcond=None)
        R = R - X_q @ coef

    return np.array(selected[:m])


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
        nc = min(n_components, n - 1) if n_components is not None else None
        mds = ClassicalMDS(n_components=nc)
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
        elif sel_name == "backward":
            train_labels = labels[train_idx]
            remaining, _ = select_queries_backward(E, pairs, train_labels, m)
            pool_idx = remaining
        elif sel_name == "bidirectional":
            train_labels = labels[train_idx]
            pool_idx = select_queries_bidirectional(E, pairs, train_labels, m)
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
    train_labels = labels[train_idx]
    if "stepwise" in selectors:
        sequences["stepwise"] = select_queries_stepwise(E, pairs, train_labels, m_max)
    if "backward" in selectors:
        # Backward produces a set for a given m; run for each m in the loop
        # But we can compute the full elimination order once (M → min_m)
        min_m = min(m_values)
        remaining, elim_order = select_queries_backward(E, pairs, train_labels, min_m)
        # Reconstruct: for m queries, the selected set is all queries minus
        # the first (M - m) eliminated queries
        all_queries = set(range(M_pool))
        sequences["backward_elim"] = elim_order  # order of removal
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
            elif sel_name == "backward":
                elim = sequences["backward_elim"]
                # Remove first (M_pool - m) queries from the full set
                removed = set(elim[:M_pool - m])
                pool_idx = np.array([q for q in range(M_pool) if q not in removed])
            elif sel_name == "bidirectional":
                pool_idx = select_queries_bidirectional(E, pairs, train_labels, m)
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


def _run_one_split_thresholds(responses, labels, query_pool, n_true_signal,
                              thresholds, m, train_idx, test_idx,
                              n_components, classifier_name, seed):
    """One split: sweep loading percentile thresholds + uniform reference."""
    rng = np.random.default_rng(seed)
    M_pool = len(query_pool)

    # Estimate loadings from train set
    train_resp = responses[train_idx][:, query_pool, :]
    E, pairs = per_query_energy_tensor(train_resp)
    r_hat, U, s = estimate_discriminative_rank(E)

    # Loading magnitude per query: max over directions
    loading_mag = np.max(np.abs(U[:, :r_hat]), axis=1)  # (M_pool,)

    # GMM-based partition for reference
    rho_hats, gmm_info = estimate_rho(U, r_hat)
    signal_idx, ortho_idx = _get_signal_orthogonal_sets(U, r_hat, gmm_info)
    gmm_frac = len(signal_idx) / M_pool  # fraction labeled signal by GMM

    results = {}

    # Threshold sweep
    for tau in thresholds:
        cutoff = np.percentile(loading_mag, tau)
        sig_set = np.where(loading_mag > cutoff)[0]
        if len(sig_set) >= m:
            pool_idx = rng.choice(sig_set, size=m, replace=False)
        elif len(sig_set) > 0:
            pool_idx = sig_set.copy()
        else:
            pool_idx = rng.choice(M_pool, size=m, replace=False)

        qi = query_pool[pool_idx]
        err = _single_trial(responses, labels, qi, train_idx, test_idx,
                            n_components, classifier_name)
        results[("threshold", tau)] = err

    # Uniform reference
    pool_idx = rng.choice(M_pool, size=m, replace=False)
    qi = query_pool[pool_idx]
    err = _single_trial(responses, labels, qi, train_idx, test_idx,
                        n_components, classifier_name)
    results[("uniform", 0)] = err

    # GMM-based est. signal reference
    if len(signal_idx) >= m:
        pool_idx = rng.choice(signal_idx, size=m, replace=False)
    else:
        pool_idx = signal_idx.copy()
    qi = query_pool[pool_idx]
    err = _single_trial(responses, labels, qi, train_idx, test_idx,
                        n_components, classifier_name)
    results[("gmm_signal", 0)] = err

    results[("gmm_frac", 0)] = gmm_frac

    return results


def run_threshold_experiment(
    responses, labels,
    query_pool=None,
    n_true_signal=None,
    thresholds=None,
    m=5,
    n_reps=100,
    n_train=80,
    n_components=8,
    classifier="lda",
    seed=42,
    n_jobs=-1,
):
    """Sweep loading magnitude percentile thresholds for signal set definition.

    Returns
    -------
    df : DataFrame with columns: threshold, selector, mean_error, std_error, gmm_percentile
    """
    if thresholds is None:
        thresholds = list(range(10, 100, 10))

    n_models, M, p = responses.shape
    if query_pool is None:
        query_pool = np.arange(M)
    if n_true_signal is None:
        n_true_signal = len(query_pool) // 2
    class0 = np.where(labels == 0)[0]
    class1 = np.where(labels == 1)[0]

    n_per_class_train = n_train // 2

    splits = []
    for rep in range(n_reps):
        rng = np.random.default_rng(seed + rep * 100003 + n_train * 31)
        sel0 = rng.choice(class0, size=n_per_class_train, replace=False)
        sel1 = rng.choice(class1, size=n_per_class_train, replace=False)
        train_idx = np.concatenate([sel0, sel1])
        test_idx = np.setdiff1d(np.arange(n_models), train_idx)
        splits.append((train_idx, test_idx))

    print(f"  Running {n_reps} splits for threshold sweep (m={m})...")
    split_results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_run_one_split_thresholds)(
            responses, labels, query_pool, n_true_signal,
            thresholds, m, train_idx, test_idx,
            n_components, classifier,
            seed + rep * 100003 + n_train * 31,
        )
        for rep, (train_idx, test_idx) in enumerate(
            tqdm(splits, desc="threshold sweep"))
    )

    # Aggregate
    all_results = []

    for tau in thresholds:
        errors = np.array([r[("threshold", tau)] for r in split_results])
        all_results.append({
            "threshold": tau,
            "selector": "threshold",
            "mean_error": errors.mean(),
            "std_error": errors.std(),
        })
        print(f"  tau={tau}: err={errors.mean():.3f} ± {errors.std():.3f}")

    # Uniform reference
    errors = np.array([r[("uniform", 0)] for r in split_results])
    all_results.append({
        "threshold": 0,
        "selector": "uniform",
        "mean_error": errors.mean(),
        "std_error": errors.std(),
    })
    print(f"  uniform: err={errors.mean():.3f} ± {errors.std():.3f}")

    # GMM signal reference
    errors = np.array([r[("gmm_signal", 0)] for r in split_results])
    gmm_fracs = np.array([r[("gmm_frac", 0)] for r in split_results])
    all_results.append({
        "threshold": 0,
        "selector": "gmm_signal",
        "mean_error": errors.mean(),
        "std_error": errors.std(),
    })
    print(f"  gmm_signal: err={errors.mean():.3f} ± {errors.std():.3f}")
    print(f"  gmm fraction signal: {gmm_fracs.mean():.2f} "
          f"(percentile ≈ {(1 - gmm_fracs.mean()) * 100:.0f})")

    df = pd.DataFrame(all_results)
    # Store mean GMM percentile as metadata
    df.attrs["gmm_percentile"] = (1 - gmm_fracs.mean()) * 100
    return df


def _run_one_split_pca(responses, labels, query_pool, m_values, train_idx,
                       test_idx, n_components, classifier_name, d_values, seed):
    """One split: PCA bidirectional stepwise at each d, plus uniform baseline."""
    rng = np.random.default_rng(seed)
    M_pool = len(query_pool)
    m_max = max(m_values)

    # PCA on train embeddings only
    train_resp = responses[train_idx][:, query_pool, :]
    Vt, mean = compute_pca_embeddings(train_resp)

    train_labels = labels[train_idx]

    results = {}  # {(selector, m): error}

    for d in d_values:
        d_eff = min(d, Vt.shape[0])
        # Reduce train responses to d dims (project ALL models using train PCA)
        all_resp_pool = responses[:, query_pool, :]
        n_all, M_p, p = all_resp_pool.shape
        stacked = all_resp_pool.reshape(-1, p) - mean
        reduced = stacked @ Vt[:d_eff].T
        resp_pca = reduced.reshape(n_all, M_p, d_eff)

        # Bidirectional stepwise on train subset
        train_pca = resp_pca[train_idx]
        seq = select_queries_pca_bidirectional(train_pca, train_labels.astype(float), m_max)

        sel_name = f"pca_d{d}"
        for m in m_values:
            pool_idx = seq[:m] if len(seq) >= m else seq
            qi = query_pool[pool_idx]
            err = _single_trial(responses, labels, qi, train_idx, test_idx,
                                n_components, classifier_name)
            results[(sel_name, m)] = err

    # Uniform baseline
    for m in m_values:
        pool_idx = rng.choice(M_pool, size=m, replace=False)
        qi = query_pool[pool_idx]
        err = _single_trial(responses, labels, qi, train_idx, test_idx,
                            n_components, classifier_name)
        results[("uniform", m)] = err

    return results


def run_pca_stepwise_experiment(
    responses, labels,
    query_pool=None,
    n_train=80,
    m_values=(2, 5, 10, 20, 50, 100),
    d_values=(2, 8, 32, 128, 768),
    n_reps=100,
    n_components=8,
    classifier="rf",
    seed=42,
    n_jobs=-1,
):
    """Run PCA bidirectional stepwise experiment across d and m.

    Returns DataFrame with columns: d_pca, m, selector, mean_error, std_error
    """
    n_models, M, p = responses.shape
    if query_pool is None:
        query_pool = np.arange(M)
    class0 = np.where(labels == 0)[0]
    class1 = np.where(labels == 1)[0]
    n_per_class_train = n_train // 2

    splits = []
    for rep in range(n_reps):
        rng = np.random.default_rng(seed + rep * 100003 + n_train * 31)
        sel0 = rng.choice(class0, size=n_per_class_train, replace=False)
        sel1 = rng.choice(class1, size=n_per_class_train, replace=False)
        train_idx = np.concatenate([sel0, sel1])
        test_idx = np.setdiff1d(np.arange(n_models), train_idx)
        splits.append((train_idx, test_idx))

    print(f"Running PCA stepwise: d={d_values}, m={m_values}, {n_reps} reps")
    split_results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_run_one_split_pca)(
            responses, labels, query_pool, m_values, train_idx, test_idx,
            n_components, classifier, d_values,
            seed + rep * 100003 + n_train * 31,
        )
        for rep, (train_idx, test_idx) in enumerate(
            tqdm(splits, desc="pca_stepwise"))
    )

    all_results = []
    selectors = [f"pca_d{d}" for d in d_values] + ["uniform"]
    for sel_name in selectors:
        for m in m_values:
            errors = np.array([r.get((sel_name, m), np.nan) for r in split_results])
            errors = errors[~np.isnan(errors)]
            if len(errors) == 0:
                continue
            d_pca = int(sel_name.split("d")[1]) if sel_name.startswith("pca_d") else 0
            all_results.append({
                "d_pca": d_pca,
                "m": m,
                "selector": sel_name,
                "mean_error": errors.mean(),
                "std_error": errors.std(),
            })
            print(f"  {sel_name}, m={m}: err={errors.mean():.3f} ± {errors.std():.3f}")

    return pd.DataFrame(all_results)


def _compute_bic(y, X):
    """BIC for OLS regression of y on X. Lower is better."""
    n = X.shape[0]
    p = X.shape[1] if X.ndim == 2 and X.shape[1] > 0 else 0
    if p == 0:
        rss = np.sum((y - y.mean(axis=0)) ** 2)
    else:
        coef, _, _, _ = np.linalg.lstsq(X, y - y.mean(axis=0), rcond=None)
        rss = np.sum((y - y.mean(axis=0) - X @ coef) ** 2)
    rss = max(rss, 1e-30)
    return n * np.log(rss / n) + np.log(n) * (p + 1)


def select_queries_pca_bic(responses_pca, y):
    """Bidirectional stepwise on PCA embeddings with BIC stopping.

    Adds queries as long as BIC decreases. Returns selected indices.
    """
    n, M, d = responses_pca.shape
    if y.ndim == 1:
        y = y[:, None]

    selected = []
    remaining = set(range(M))
    bic_current = _compute_bic(y, np.empty((n, 0)))

    for step in range(M):
        # --- Forward: find query that most decreases BIC ---
        best_bic = bic_current
        best_q = -1
        for q in remaining:
            X_trial = np.hstack(
                [responses_pca[:, s, :] for s in selected] + [responses_pca[:, q, :]]
            )
            bic_trial = _compute_bic(y, X_trial)
            if bic_trial < best_bic:
                best_bic = bic_trial
                best_q = q

        if best_q < 0:
            break
        selected.append(best_q)
        remaining.discard(best_q)
        bic_current = best_bic

        # --- Backward: remove if any query's removal decreases BIC ---
        if len(selected) > 1:
            improved = True
            while improved and len(selected) > 1:
                improved = False
                best_remove_bic = bic_current
                best_remove_idx = -1
                for i in range(len(selected)):
                    others = selected[:i] + selected[i+1:]
                    X_others = np.hstack([responses_pca[:, s, :] for s in others])
                    bic_r = _compute_bic(y, X_others)
                    if bic_r < best_remove_bic:
                        best_remove_bic = bic_r
                        best_remove_idx = i
                if best_remove_idx >= 0:
                    removed = selected.pop(best_remove_idx)
                    remaining.add(removed)
                    bic_current = best_remove_bic
                    improved = True

    return np.array(selected)


def _run_one_pool_trial(responses, labels, full_pool, pool_size, n_train,
                        d_pca, n_components, classifier_name, seed):
    """One trial: random query pool + random train/test split + BIC stepwise."""
    rng = np.random.default_rng(seed)
    n_models = len(labels)
    class0 = np.where(labels == 0)[0]
    class1 = np.where(labels == 1)[0]

    # Sample query pool
    ps = min(pool_size, len(full_pool))
    query_pool = rng.choice(full_pool, size=ps, replace=False)

    # Sample train/test split
    n_per = n_train // 2
    if n_per > len(class0) or n_per > len(class1):
        return {"error_stepwise": 0.5, "error_uniform": 0.5, "error_random_m": 0.5,
                "m_selected": 0, "pool_size": ps, "n_train": n_train}
    sel0 = rng.choice(class0, size=n_per, replace=False)
    sel1 = rng.choice(class1, size=n_per, replace=False)
    train_idx = np.concatenate([sel0, sel1])
    test_idx = np.setdiff1d(np.arange(n_models), train_idx)

    # PCA on train embeddings
    train_resp = responses[train_idx][:, query_pool, :]
    Vt, mean = compute_pca_embeddings(train_resp)
    d_eff = min(d_pca, Vt.shape[0])

    # Project all models
    all_resp = responses[:, query_pool, :]
    n_all, M_p, p = all_resp.shape
    stacked = all_resp.reshape(-1, p) - mean
    reduced = stacked @ Vt[:d_eff].T
    resp_pca = reduced.reshape(n_all, M_p, d_eff)

    # BIC stepwise on train
    train_pca = resp_pca[train_idx]
    train_labels = labels[train_idx].astype(float)
    sel = select_queries_pca_bic(train_pca, train_labels)

    if len(sel) == 0:
        return {"error_stepwise": 0.5, "error_uniform": 0.5, "error_random_m": 0.5,
                "m_selected": 0, "pool_size": ps, "n_train": n_train}

    # Evaluate stepwise selection
    qi = query_pool[sel]
    err_sw = _single_trial(responses, labels, qi, train_idx, test_idx,
                           n_components, classifier_name)

    # Evaluate uniform (all queries in pool)
    err_unif = _single_trial(responses, labels, query_pool, train_idx, test_idx,
                             n_components, classifier_name)

    # Evaluate uniform matched-m (random m queries from pool)
    m_sel = len(sel)
    rand_idx = rng.choice(ps, size=m_sel, replace=False)
    qi_rand = query_pool[rand_idx]
    err_rand = _single_trial(responses, labels, qi_rand, train_idx, test_idx,
                             n_components, classifier_name)

    return {"error_stepwise": err_sw, "error_uniform": err_unif,
            "error_random_m": err_rand,
            "m_selected": m_sel, "pool_size": ps, "n_train": n_train}


def run_pool_experiment(
    responses, labels,
    query_pool=None,
    pool_sizes=(10, 20, 50, 100, 200),
    n_train_values=(10, 20, 50, 80),
    d_pca=2,
    n_reps=25,
    n_components=8,
    classifier="rf",
    seed=42,
    n_jobs=-1,
):
    """Sweep pool size and model count with BIC-stopping PCA stepwise.

    Returns DataFrame with columns:
        pool_size, n_train, mean_error, std_error, mean_m, std_m
    """
    n_models = len(labels)
    if query_pool is None:
        query_pool = np.arange(responses.shape[1])

    tasks = []
    for ps in pool_sizes:
        for nt in n_train_values:
            if nt > n_models:
                continue
            for rep in range(n_reps):
                tasks.append((ps, nt, seed + rep * 100003 + ps * 37 + nt * 13))

    print(f"Running pool experiment: {len(tasks)} trials "
          f"(pools={pool_sizes}, n_train={n_train_values}, {n_reps} reps)")

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_run_one_pool_trial)(
            responses, labels, query_pool, ps, nt,
            d_pca, n_components, classifier, s,
        )
        for ps, nt, s in tqdm(tasks, desc="pool_experiment")
    )

    # Aggregate — one row per (selector, pool_size, n_train)
    rows = []
    for ps in pool_sizes:
        for nt in n_train_values:
            trial_results = [r for r in results
                             if r["pool_size"] == ps and r["n_train"] == nt]
            if not trial_results:
                continue
            ms = np.array([r["m_selected"] for r in trial_results])
            for sel, key in [("stepwise", "error_stepwise"),
                             ("uniform_all", "error_uniform"),
                             ("random_m", "error_random_m")]:
                errors = np.array([r[key] for r in trial_results])
                rows.append({
                    "selector": sel,
                    "pool_size": ps,
                    "n_train": nt,
                    "mean_error": errors.mean(),
                    "std_error": errors.std(),
                    "mean_m": ms.mean(),
                    "std_m": ms.std(),
                })
            sw = np.array([r["error_stepwise"] for r in trial_results])
            uf = np.array([r["error_uniform"] for r in trial_results])
            print(f"  pool={ps}, n={nt}: stepwise={sw.mean():.3f}, "
                  f"uniform_all={uf.mean():.3f}, m={ms.mean():.1f}±{ms.std():.1f}")

    return pd.DataFrame(rows)
