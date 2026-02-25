"""Estimate discriminative rank r̂ and per-direction zero-set probabilities ρ̂_ℓ.

Steps:
1. Compute Ẽ (between-class centered per-query dissimilarities)
2. SVD of Ẽ → r̂ via spectral gap (profile likelihood)
3. For each direction ℓ = 1..r̂, fit a 2-component GMM to |Ũ_{q,ℓ}|
   (the left-singular-vector loadings). The near-zero component weight
   gives ρ̂_ℓ, the probability a random query carries no signal along ℓ.
4. Failure bound: Σ_ℓ ρ̂_ℓ^m. For r̂ = 1 this reduces to a single GMM on B_q.
"""

import numpy as np
from sklearn.mixture import GaussianMixture


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


def estimate_discriminative_rank(E: np.ndarray):
    """Estimate r̂ from the singular value spectrum of Ẽ via largest successive ratio.

    r̂ = argmax_r (σ_r / σ_{r+1}), i.e. the position of the largest spectral gap.

    Parameters
    ----------
    E : ndarray of shape (M, n_pairs)
        Between-class centered per-query energy matrix Ẽ.

    Returns
    -------
    r_hat : int
        Estimated discriminative rank (≥ 1).
    U : ndarray of shape (M, k)
        Left singular vectors (query loadings).
    s : ndarray of shape (k,)
        Singular values in descending order.
    """
    U, s, Vt = np.linalg.svd(E, full_matrices=False)
    # Largest successive ratio σ_r / σ_{r+1} in the leading half of the spectrum.
    # Restricting to the first half avoids spurious gaps at the numerical rank
    # boundary while still covering any plausible discriminative rank.
    k = max(2, len(s) // 2)
    ratios = s[:k - 1] / s[1:k]
    r_hat = int(np.argmax(ratios)) + 1  # 1-indexed
    return r_hat, U, s


def estimate_rho(U: np.ndarray, r_hat: int):
    """Estimate per-direction zero-set probabilities ρ̂_ℓ via GMM.

    For each direction ℓ = 1..r̂, fits K=1 and K=2 GMMs to the absolute
    left-singular-vector loadings |U_{q,ℓ}|. The near-zero component
    weight gives ρ̂_ℓ.

    For r̂ = 1, the loadings |U_{q,1}| are proportional to B_q, so the
    procedure reduces to fitting a GMM on B_q.

    Parameters
    ----------
    U : ndarray of shape (M, k)
        Left singular vectors from SVD of Ẽ.
    r_hat : int
        Estimated discriminative rank.

    Returns
    -------
    rho_hats : ndarray of shape (r_hat,)
        Per-direction zero-set probabilities.
    info : dict
        Diagnostic information:
        - 'per_direction': list of dicts (one per direction ℓ), each with:
            - 'loadings': |U_{q,ℓ}| values
            - 'gmm': fitted 2-component GaussianMixture
            - 'gmm1': fitted 1-component GMM
            - 'bic1', 'bic2': BIC values
            - 'rho_l': ρ̂_ℓ for this direction
            - 'labels': component assignments (0=near-zero, 1=active)
    """
    rho_hats = np.zeros(r_hat)
    per_direction = []

    for ell in range(r_hat):
        loadings = np.abs(U[:, ell])
        L_col = loadings.reshape(-1, 1)

        gmm1 = GaussianMixture(n_components=1, random_state=0).fit(L_col)
        gmm2 = GaussianMixture(n_components=2, random_state=0).fit(L_col)

        bic1 = gmm1.bic(L_col)
        bic2 = gmm2.bic(L_col)

        # Near-zero component = lower mean
        means = gmm2.means_.ravel()
        zero_comp = int(np.argmin(means))
        rho_l = gmm2.weights_[zero_comp]

        raw_labels = gmm2.predict(L_col)
        labels = np.where(raw_labels == zero_comp, 0, 1)

        rho_hats[ell] = rho_l
        per_direction.append({
            'loadings': loadings,
            'gmm': gmm2,
            'gmm1': gmm1,
            'bic1': bic1,
            'bic2': bic2,
            'rho_l': rho_l,
            'labels': labels,
        })

    info = {'per_direction': per_direction}
    return rho_hats, info


def predict_mstar(rho_hats, epsilon: float = 0.05):
    """Predict the query budget m* from the theoretical bound Σ_ℓ ρ̂_ℓ^m ≤ ε.

    Finds the smallest m such that Σ_ℓ ρ̂_ℓ^m ≤ ε.

    Parameters
    ----------
    rho_hats : float or array-like
        Per-direction zero-set probabilities. Scalar for r̂=1.
    epsilon : float, default=0.05
        Desired failure probability.

    Returns
    -------
    mstar : int or float
        Predicted query budget. Returns 1 if all ρ̂_ℓ = 0,
        np.inf if any ρ̂_ℓ ≥ 1.
    """
    rho_hats = np.atleast_1d(np.asarray(rho_hats, dtype=float))
    if np.any(rho_hats >= 1):
        return np.inf
    if np.all(rho_hats <= 0):
        return 1
    # Binary search for smallest m where sum(rho_l^m) <= epsilon
    for m in range(1, 10000):
        if np.sum(rho_hats ** m) <= epsilon:
            return m
    return np.inf
