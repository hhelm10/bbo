"""Estimate discriminative rank r̂ and zero-set probability ρ̂ from the E matrix.

The E matrix is M × C(n,2) where E_{q,(i,j)} = ||g(f_i(q)) - g(f_j(q))||².
Its SVD reveals the discriminative rank (number of independent directions
along which queries separate model pairs) and the zero-set probability
(fraction of queries that are non-discriminative along each direction).
"""

import numpy as np

from bbo.embedding.mds import select_dimension


def estimate_discriminative_rank(E: np.ndarray, n_elbows: int = 1):
    """Estimate r̂ from the singular value spectrum of E.

    Uses the Zhu & Ghodsi (2006) profile likelihood method with 1 elbow
    to find the discriminative rank.

    Parameters
    ----------
    E : ndarray of shape (M, n_pairs)
        Per-query energy tensor (squared distances for each query × model pair).
    n_elbows : int, default=1
        Number of elbows to find in the singular value spectrum.

    Returns
    -------
    r_hat : int
        Estimated discriminative rank.
    U : ndarray of shape (M, k)
        Left singular vectors (query loadings).
    s : ndarray of shape (k,)
        Singular values in descending order.
    """
    U, s, Vt = np.linalg.svd(E, full_matrices=False)
    r_hat = select_dimension(s, n_elbows=n_elbows)
    return r_hat, U, s


def estimate_rho(U: np.ndarray, r_hat: int, tau: float = 0.05):
    """Estimate ρ̂ from the U loadings of the rank-r̂ truncated SVD.

    For each direction ℓ, ρ̂_ℓ = fraction of queries with
    |U_{q,ℓ}| < τ · max_q |U_{·,ℓ}|.  Returns the worst-case (maximum) ρ̂.

    Parameters
    ----------
    U : ndarray of shape (M, k)
        Left singular vectors from SVD of E.
    r_hat : int
        Estimated discriminative rank.
    tau : float, default=0.05
        Threshold fraction for considering a query's loading as "zero".

    Returns
    -------
    rho_hat : float
        Worst-case (maximum) zero-set probability across directions.
    rhos : list of float
        Per-direction zero-set probabilities ρ̂_ℓ for ℓ = 1, …, r̂.
    """
    U_r = U[:, :r_hat]
    rhos = []
    for ell in range(r_hat):
        col = np.abs(U_r[:, ell])
        threshold = tau * col.max()
        rho_ell = (col < threshold).mean()
        rhos.append(float(rho_ell))
    return max(rhos), rhos


def predict_mstar(r_hat: int, rho_hat: float, epsilon: float = 0.05):
    """Predict the query budget m* from the theoretical bound.

    m* = ⌈log(2r / ε) / (−log ρ)⌉

    Parameters
    ----------
    r_hat : int
        Estimated discriminative rank.
    rho_hat : float
        Estimated worst-case zero-set probability.
    epsilon : float, default=0.05
        Desired failure probability.

    Returns
    -------
    mstar : int or float
        Predicted query budget. Returns 1 if ρ̂ = 0 (every query discriminates),
        np.inf if ρ̂ ≥ 1 (no query discriminates).
    """
    if rho_hat >= 1:
        return np.inf
    if rho_hat <= 0:
        return 1
    return int(np.ceil(np.log(2 * r_hat / epsilon) / (-np.log(rho_hat))))
