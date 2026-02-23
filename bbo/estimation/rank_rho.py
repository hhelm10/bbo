"""Estimate discriminative rank r̂ and zero-set probability ρ̂ from data.

Uses the per-query between-class excess B_q to estimate ρ̂ via GMM, and
the singular value spectrum of E to estimate r̂ via profile likelihood.

Steps:
1. Compute E (per-query pairwise dissimilarities for all model pairs)
2. Partition pairs into within-class and cross-class; compute B_q
   (per-query between-class excess = mean cross-class - mean within-class)
3. SVD of E → r̂ via spectral gap (1 elbow)
4. Two-component GMM on B_q → ρ̂ = π̂₀ (near-zero component weight)
"""

import numpy as np
from sklearn.mixture import GaussianMixture

from bbo.embedding.mds import select_dimension


def compute_E_disc(E: np.ndarray, pairs: np.ndarray, labels: np.ndarray):
    """Compute the between-class centered dissimilarity matrix and B_q.

    For each query q, subtracts the average within-class dissimilarity
    from each cross-class pair's dissimilarity, isolating the discriminative
    component.

    Parameters
    ----------
    E : ndarray of shape (M, n_pairs)
        Per-query energy tensor.
    pairs : ndarray of shape (n_pairs, 2)
        Model pair indices (i, j) for each column of E.
    labels : ndarray of shape (n_models,)
        Binary class labels.

    Returns
    -------
    E_disc : ndarray of shape (M, n_cross)
        Between-class centered dissimilarity matrix (cross-class pairs only).
    cross_mask : ndarray of shape (n_pairs,), bool
        Mask indicating which pairs are cross-class.
    B_q : ndarray of shape (M,)
        Per-query between-class excess (scalar summary).
    """
    y_i = labels[pairs[:, 0]]
    y_j = labels[pairs[:, 1]]

    # Partition masks
    within0 = (y_i == 0) & (y_j == 0)
    within1 = (y_i == 1) & (y_j == 1)
    cross = y_i != y_j

    # Per-query within-class means
    E_bar_0 = E[:, within0].mean(axis=1) if within0.any() else np.zeros(E.shape[0])
    E_bar_1 = E[:, within1].mean(axis=1) if within1.any() else np.zeros(E.shape[0])
    within_mean = 0.5 * (E_bar_0 + E_bar_1)  # (M,)

    # E^disc: cross-class pairs, centered by within-class mean
    E_cross = E[:, cross]  # (M, n_cross)
    E_disc = E_cross - within_mean[:, np.newaxis]

    # Per-query between-class excess (scalar)
    B_q = E_cross.mean(axis=1) - within_mean

    return E_disc, cross, B_q


def estimate_discriminative_rank(E: np.ndarray, n_elbows: int = 1):
    """Estimate r̂ from the singular value spectrum of E.

    Parameters
    ----------
    E : ndarray of shape (M, n_pairs)
        Per-query energy tensor (full, not between-class centered).
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


def estimate_rho(B_q: np.ndarray, k_max: int = 10):
    """Estimate ρ̂ via GMM on per-query between-class excess.

    Fits GMMs with K=1,...,k_max components and selects the best K by BIC.
    The near-zero component captures queries with no discriminative
    contribution (orthogonal queries), while positive components capture
    discriminative (signal) queries.

    ρ̂ = π̂₀, the mixing weight of the near-zero component.

    Parameters
    ----------
    B_q : ndarray of shape (M,)
        Per-query between-class excess values.
    k_max : int, default=10
        Maximum number of GMM components to search over.

    Returns
    -------
    rho_hat : float
        Estimated zero-set probability.
    info : dict
        Diagnostic information:
        - 'pi0': mixing weight of the near-zero component
        - 'B_q': the input B_q values
        - 'gmm': best GMM (K=K*)
        - 'gmm1': fitted 1-component GMM (for BIC comparison)
        - 'bic1': BIC of 1-component model
        - 'bic_best': BIC of best model
        - 'k_best': number of components in best model
        - 'all_bics': dict mapping K -> BIC
        - 'labels': component assignments (0=near-zero, 1=active)
    """
    B_col = B_q.reshape(-1, 1)

    # Fit K=1,...,k_max and select by BIC
    gmms = {}
    bics = {}
    for k in range(1, k_max + 1):
        gmm = GaussianMixture(n_components=k, random_state=0).fit(B_col)
        gmms[k] = gmm
        bics[k] = gmm.bic(B_col)

    k_best = min(bics, key=bics.get)
    gmm_best = gmms[k_best]
    gmm1 = gmms[1]

    if k_best == 1:
        # Single component: no separation, ρ̂ = 1
        rho_hat = 1.0
        pi0 = 1.0
        labels = np.zeros(len(B_q), dtype=int)
    else:
        # Identify the near-zero component (lowest mean)
        means = gmm_best.means_.ravel()
        zero_comp = int(np.argmin(means))
        pi0 = gmm_best.weights_[zero_comp]

        # Component labels: 0=near-zero, 1=active
        raw_labels = gmm_best.predict(B_col)
        labels = np.where(raw_labels == zero_comp, 0, 1)

        rho_hat = pi0

    info = {
        'pi0': pi0,
        'B_q': B_q,
        'gmm': gmm_best,
        'gmm1': gmm1,
        'bic1': bics[1],
        'bic_best': bics[k_best],
        'k_best': k_best,
        'all_bics': bics,
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
