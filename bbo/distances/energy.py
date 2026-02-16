"""Pairwise distance computation for black-box model comparison.

At temperature 0, each model produces a deterministic response per query.
The cumulative squared distance over m queries is:

    D^2_m(f, f') = sum_{k=1}^m ||g(f(q_k)) - g(f'(q_k))||^2

Using squared norms ensures the discriminative factorization holds:
    ||diff(q)||^2 = sum_l alpha_l(q) * phi_l(f, f')
with alpha_l(q) >= 0 and phi_l(f,f') >= 0.
"""

import numpy as np


def pairwise_energy_distances_t0(responses: np.ndarray, query_indices: np.ndarray = None) -> np.ndarray:
    """Compute pairwise cumulative distances at temperature 0.

    D^2_m(f, f') = sum_{k=1}^m ||g(f(q_k)) - g(f'(q_k))||^2
    Returns D where D[i,j] = sqrt(D^2_m) for metric use.

    Parameters
    ----------
    responses : ndarray of shape (n_models, M, p)
        Precomputed embedded responses g(f(q)) for all models and queries.
    query_indices : ndarray of shape (m,), optional
        Indices of queries to use. If None, use all queries.

    Returns
    -------
    D : ndarray of shape (n_models, n_models)
        Symmetric distance matrix.
    """
    if query_indices is not None:
        R = responses[:, query_indices, :]  # (n, m, p)
    else:
        R = responses  # (n, M, p)

    n = R.shape[0]

    D = np.zeros((n, n))
    for i in range(n):
        diffs = R[i] - R[i + 1:]  # (n-i-1, m, p)
        sq_norms = np.sum(diffs**2, axis=-1)  # (n-i-1, m)
        D_sq = sq_norms.sum(axis=-1)  # (n-i-1,)
        D[i, i + 1:] = np.sqrt(D_sq)

    D = D + D.T
    return D


def per_query_energy_tensor(responses: np.ndarray) -> np.ndarray:
    """Compute the full M x n_pairs squared-distance tensor.

    For SVD analysis (Exp 6): each entry is ||g(f_i(q)) - g(f_j(q))||^2 for
    a single query and model pair.

    Parameters
    ----------
    responses : ndarray of shape (n_models, M, p)
        Precomputed embedded responses.

    Returns
    -------
    T : ndarray of shape (M, n_pairs)
        Where n_pairs = n_models * (n_models - 1) / 2.
        T[q, k] = ||g(f_i(q)) - g(f_j(q))||^2 for the k-th pair (i, j).
    pairs : ndarray of shape (n_pairs, 2)
        The (i, j) indices for each pair.
    """
    n, M, p = responses.shape
    n_pairs = n * (n - 1) // 2

    T = np.zeros((M, n_pairs))
    pairs = np.zeros((n_pairs, 2), dtype=int)

    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            diff = responses[i] - responses[j]
            T[:, idx] = np.sum(diff**2, axis=-1)
            pairs[idx] = [i, j]
            idx += 1

    return T, pairs
