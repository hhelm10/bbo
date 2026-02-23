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


def estimate_rho(U: np.ndarray, r_hat: int, K_max: int = 5):
    """Estimate ρ̂ via Gaussian mixture model on loading norms.

    Selects the number of components K by BIC (K=2,...,K_max), then
    estimates ρ̂ from the mixing weight of the component whose mean is
    closest to zero.

    Parameters
    ----------
    U : ndarray of shape (M, k)
        Left singular vectors from SVD of E.
    r_hat : int
        Estimated discriminative rank.
    K_max : int, default=5
        Maximum number of GMM components to consider.

    Returns
    -------
    rho_hat : float
        Estimated zero-set probability.
    info : dict
        Diagnostic information:
        - 'pi0': mixing weight of the near-zero component
        - 'norms': array of loading norms
        - 'gmm': fitted GaussianMixture object (best K)
        - 'gmm1': fitted 1-component GMM (for comparison)
        - 'K_best': selected number of components
        - 'bics': dict mapping K -> BIC value
        - 'labels': component assignments (0=near-zero, else=active)
    """
    U_r = U[:, :r_hat]

    # Aggregate loading norms
    norms = np.linalg.norm(U_r, axis=1)
    norms_col = norms.reshape(-1, 1)

    # Fit K=1 for comparison
    gmm1 = GaussianMixture(n_components=1, random_state=0).fit(norms_col)

    # Fit K=2,...,K_max and select by BIC
    bics = {1: gmm1.bic(norms_col)}
    best_gmm = None
    best_bic = np.inf
    best_K = 2

    for K in range(2, K_max + 1):
        gmm_k = GaussianMixture(n_components=K, random_state=0).fit(norms_col)
        bic_k = gmm_k.bic(norms_col)
        bics[K] = bic_k
        if bic_k < best_bic:
            best_bic = bic_k
            best_gmm = gmm_k
            best_K = K

    # Identify the near-zero component (lowest mean)
    means = best_gmm.means_.ravel()
    zero_comp = int(np.argmin(means))
    pi0 = best_gmm.weights_[zero_comp]

    # Component labels: 0=near-zero, else=active
    raw_labels = best_gmm.predict(norms_col)
    labels = np.where(raw_labels == zero_comp, 0, 1)

    # Recover per-direction ρ̂
    rho_hat = pi0 ** (1.0 / r_hat) if r_hat > 0 else pi0

    info = {
        'pi0': pi0,
        'norms': norms,
        'gmm': best_gmm,
        'gmm1': gmm1,
        'K_best': best_K,
        'bics': bics,
        'labels': labels,
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
