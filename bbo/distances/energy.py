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


def per_query_dissimilarity_tensor(responses: np.ndarray, metric: str = "sq_euclidean"):
    """Compute the M x n_pairs per-query dissimilarity tensor for a given metric.

    Generalization of per_query_energy_tensor to alternative response-space
    dissimilarities delta. At temperature 0 each model emits one deterministic
    response per query, so delta compares two embedded points.

    Parameters
    ----------
    responses : ndarray of shape (n_models, M, p)
        Precomputed embedded responses.
    metric : str
        One of "sq_euclidean", "euclidean", "cosine", "l1", "rbf".
        "rbf" is the MMD with an RBF kernel between point masses:
        2 * (1 - exp(-||x-y||^2 / (2 * sigma^2))), sigma via median heuristic.

    Returns
    -------
    T : ndarray of shape (M, n_pairs)
    pairs : ndarray of shape (n_pairs, 2)
    """
    n, M, p = responses.shape
    n_pairs = n * (n - 1) // 2

    T = np.zeros((M, n_pairs))
    pairs = np.zeros((n_pairs, 2), dtype=int)

    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            diff = responses[i] - responses[j]
            sq = np.sum(diff**2, axis=-1)
            if metric == "sq_euclidean":
                T[:, idx] = sq
            elif metric == "euclidean":
                T[:, idx] = np.sqrt(sq)
            elif metric == "l1":
                T[:, idx] = np.sum(np.abs(diff), axis=-1)
            elif metric == "cosine":
                dots = np.sum(responses[i] * responses[j], axis=-1)
                norms = (np.linalg.norm(responses[i], axis=-1)
                         * np.linalg.norm(responses[j], axis=-1))
                T[:, idx] = 1.0 - dots / np.maximum(norms, 1e-12)
            elif metric == "rbf":
                T[:, idx] = sq  # placeholder; converted below
            else:
                raise ValueError(f"Unknown metric: {metric}")
            pairs[idx] = [i, j]
            idx += 1

    if metric == "rbf":
        # Median heuristic on pairwise Euclidean distances
        sigma_sq = np.median(T[T > 0]) / 2.0
        T = 2.0 * (1.0 - np.exp(-T / (2.0 * sigma_sq)))

    return T, pairs


def _pairwise_delta(A: np.ndarray, B: np.ndarray, metric: str,
                    sigma_sq: float = None) -> np.ndarray:
    """Base dissimilarity delta between all draw pairs.

    A, B : ndarray of shape (M, K, p). Returns (M, K, K).
    """
    if metric == "cosine":
        dots = np.einsum("mkp,mlp->mkl", A, B)
        norms = (np.linalg.norm(A, axis=-1)[:, :, None]
                 * np.linalg.norm(B, axis=-1)[:, None, :])
        return 1.0 - dots / np.maximum(norms, 1e-12)
    diffs = A[:, :, None, :] - B[:, None, :, :]
    if metric == "euclidean":
        return np.linalg.norm(diffs, axis=-1)
    elif metric == "sq_euclidean":
        return np.sum(diffs**2, axis=-1)
    elif metric == "l1":
        return np.sum(np.abs(diffs), axis=-1)
    elif metric == "rbf":
        sq = np.sum(diffs**2, axis=-1)
        return 2.0 * (1.0 - np.exp(-sq / (2.0 * sigma_sq)))
    raise ValueError(f"Unknown metric: {metric}")


def per_query_energy_ustat_tensor(responses: np.ndarray, metric: str = "euclidean",
                                  rng: np.random.Generator = None):
    """Per-query generalized squared energy distance tensor from repeated draws.

    For temperature > 0: each model emits K draws per query, and the squared
    energy distance between response distributions is estimated by the
    unbiased U-statistic

        E^2(P_i, P_j; q) = 2 E delta(X, Y) - E delta(X, X') - E delta(Y, Y')

    with the within-sample terms averaged over ordered pairs a != b.
    Valid for any negative-type base dissimilarity delta:
    - "euclidean": classical energy distance
    - "sq_euclidean": reduces to 2 ||mu_X - mu_Y||^2 (mean embedding)
    - "l1": coordinatewise energy distance
    - "cosine": energy distance on the unit sphere
    - "rbf": equals 2 * MMD^2 with an RBF kernel (sigma via median heuristic
      on cross-pair squared Euclidean distances, subsampled)

    Parameters
    ----------
    responses : ndarray of shape (n_models, M, K, p)
        K embedded draws per (model, query).

    Returns
    -------
    T : ndarray of shape (M, n_pairs)
        Unbiased estimates (may be slightly negative under noise).
    pairs : ndarray of shape (n_pairs, 2)
    """
    n, M, K, p = responses.shape
    if K < 2:
        raise ValueError("U-statistic requires K >= 2 draws per query.")
    n_pairs = n * (n - 1) // 2

    sigma_sq = None
    if metric == "rbf":
        # Median heuristic on cross-pair squared Euclidean distances,
        # estimated from a subsample of model pairs.
        rng = rng or np.random.default_rng(0)
        sq_samples = []
        for _ in range(min(200, n_pairs)):
            i, j = rng.choice(n, size=2, replace=False)
            diffs = responses[i][:, :, None, :] - responses[j][:, None, :, :]
            sq_samples.append(np.sum(diffs**2, axis=-1).ravel())
        sq_samples = np.concatenate(sq_samples)
        sigma_sq = np.median(sq_samples[sq_samples > 0]) / 2.0

    # Within-sample means: w[i, q] = mean_{a != b} delta(x_a, x_b)
    within = np.zeros((n, M))
    iu = np.triu_indices(K, k=1)
    for i in range(n):
        d = _pairwise_delta(responses[i], responses[i], metric, sigma_sq)
        within[i] = d[:, iu[0], iu[1]].mean(axis=-1)

    T = np.zeros((M, n_pairs))
    pairs = np.zeros((n_pairs, 2), dtype=int)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = _pairwise_delta(responses[i], responses[j], metric, sigma_sq)
            cross = d.reshape(M, -1).mean(axis=-1)
            T[:, idx] = 2.0 * cross - within[i] - within[j]
            pairs[idx] = [i, j]
            idx += 1

    return T, pairs


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
