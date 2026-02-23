"""Estimate discriminative rank r̂ and zero-set probability ρ̂ from the E matrix.

The E matrix is M × C(n,2) where E_{q,(i,j)} = ||g(f_i(q)) - g(f_j(q))||².
Its SVD reveals the discriminative rank (number of independent directions
along which queries separate model pairs) and the zero-set probability
(fraction of queries that are non-discriminative along each direction).
"""

import numpy as np
from sklearn.mixture import GaussianMixture

from bbo.embedding.mds import select_dimension


def estimate_discriminative_rank(E: np.ndarray, n_elbows: int = 1):
    """Estimate r̂ from the singular value spectrum of E.

    Uses the Zhu & Ghodsi (2006) profile likelihood method to find the
    discriminative rank via spectral gap detection.

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


def estimate_rho(U: np.ndarray, r_hat: int):
    """Estimate ρ̂ via Gaussian mixture model on loading norms.

    Fits a two-component GMM to the row norms ||U_{q,·}|| of the
    rank-r̂ truncated left singular vectors. The near-zero component's
    mixing weight π̂₀ estimates ρ^r, so ρ̂ = π̂₀^(1/r̂).

    Parameters
    ----------
    U : ndarray of shape (M, k)
        Left singular vectors from SVD of E.
    r_hat : int
        Estimated discriminative rank.

    Returns
    -------
    rho_hat : float
        Estimated zero-set probability.
    info : dict
        Diagnostic information:
        - 'pi0': mixing weight of the near-zero component
        - 'norms': array of loading norms
        - 'gmm': fitted GaussianMixture object
        - 'labels': component assignments (0=near-zero, 1=active)
        - 'per_direction': list of per-direction GMM fit dicts
    """
    U_r = U[:, :r_hat]

    # Aggregate loading norms
    norms = np.linalg.norm(U_r, axis=1)

    # Fit 2-component GMM to norms
    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(norms.reshape(-1, 1))

    # Identify the near-zero component (lower mean)
    means = gmm.means_.ravel()
    zero_comp = int(np.argmin(means))
    pi0 = gmm.weights_[zero_comp]

    # Component labels: 0=near-zero, 1=active
    raw_labels = gmm.predict(norms.reshape(-1, 1))
    labels = np.where(raw_labels == zero_comp, 0, 1)

    # Recover per-direction ρ̂
    rho_hat = pi0 ** (1.0 / r_hat) if r_hat > 0 else pi0

    # Per-direction diagnostic GMM fits
    per_direction = []
    for ell in range(r_hat):
        col = np.abs(U_r[:, ell])
        gmm_dir = GaussianMixture(n_components=2, random_state=0)
        gmm_dir.fit(col.reshape(-1, 1))
        dir_means = gmm_dir.means_.ravel()
        dir_zero_comp = int(np.argmin(dir_means))
        per_direction.append({
            'pi0': gmm_dir.weights_[dir_zero_comp],
            'means': dir_means,
            'gmm': gmm_dir,
            'values': col,
        })

    info = {
        'pi0': pi0,
        'norms': norms,
        'gmm': gmm,
        'labels': labels,
        'per_direction': per_direction,
    }
    return rho_hat, info


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
