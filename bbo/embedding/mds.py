"""Classical MDS implementation from scratch.

Implements multidimensional scaling via double-centering and eigendecomposition,
with support for out-of-sample projection (Gower's formula).

Includes automatic dimensionality selection via the profile likelihood method
of Zhu & Ghodsi (2006).
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.linalg import eigh


def _compute_profile_likelihood(values: np.ndarray) -> np.ndarray:
    """Compute profile log-likelihood for a sequence of sorted values.

    For each candidate split point, fits a two-segment constant-mean normal
    model with pooled variance and returns the log-likelihood.

    Parameters
    ----------
    values : ndarray of shape (n,)
        Sorted (descending) eigenvalues, all positive.

    Returns
    -------
    logliks : ndarray of shape (n-1,)
        Log-likelihood at each candidate split point i=1,...,n-1.

    References
    ----------
    Zhu, M. and Ghodsi, A. (2006). "Automatic dimensionality selection from
    the scree plot via the use of profile likelihood." Computational Statistics
    & Data Analysis, 51(2), pp. 918-930.
    """
    n = len(values)
    logliks = np.empty(n - 1)

    for i in range(1, n):
        group1 = values[:i]
        group2 = values[i:]

        mu1 = group1.mean()
        mu2 = group2.mean()

        # Pooled variance (MLE)
        ss = np.sum((group1 - mu1) ** 2) + np.sum((group2 - mu2) ** 2)
        sigma2 = ss / n

        if sigma2 < 1e-300:
            # Perfect fit — all values in each group identical
            logliks[i - 1] = 0.0
        else:
            logliks[i - 1] = -0.5 * n * np.log(2 * np.pi * sigma2) - ss / (2 * sigma2)

    return logliks


def select_dimension(eigenvalues: np.ndarray, n_elbows: int = 2) -> int:
    """Select embedding dimension via the profile likelihood method.

    Iteratively finds elbow points in the eigenvalue spectrum by maximizing
    the profile likelihood of a two-segment normal model. Returns the
    dimension at the last elbow.

    Parameters
    ----------
    eigenvalues : ndarray
        Eigenvalues in descending order (as returned by ClassicalMDS.fit).
    n_elbows : int, default=2
        Number of elbows to find. The dimension at the last elbow is returned.

    Returns
    -------
    d : int
        Selected number of dimensions (>= 1).

    References
    ----------
    Zhu, M. and Ghodsi, A. (2006). "Automatic dimensionality selection from
    the scree plot via the use of profile likelihood." Computational Statistics
    & Data Analysis, 51(2), pp. 918-930.
    """
    # Keep only positive eigenvalues
    pos = eigenvalues[eigenvalues > 0].copy()

    if len(pos) <= 1:
        return max(1, len(pos))

    elbows = []
    values = pos.copy()

    for _ in range(n_elbows):
        if len(values) <= 1:
            break

        logliks = _compute_profile_likelihood(values)
        # Best split point (1-indexed into values)
        elbow_idx = int(np.argmax(logliks)) + 1
        elbows.append(elbow_idx)

        # Truncate to the left segment for next iteration
        values = values[:elbow_idx]

    if not elbows:
        return 1

    # The last elbow found gives the selected dimension.
    # But since we're iteratively truncating, the cumulative dimension
    # is the *last* elbow_idx found (it indexes into an already-truncated array,
    # so it's already in terms of the original indexing up to previous elbow).
    return max(1, elbows[-1])


@dataclass
class ClassicalMDS:
    """Classical (metric) multidimensional scaling.

    Parameters
    ----------
    n_components : int or None
        Number of embedding dimensions to retain. If None, automatically
        selected via the Zhu & Ghodsi (2006) profile likelihood method.
    n_elbows : int
        Number of elbows for automatic dimension selection (only used when
        n_components is None).
    min_eig : float
        Eigenvalues below this threshold are zeroed (handles numerical noise).
    """

    n_components: Optional[int] = None
    n_elbows: int = 2
    min_eig: float = 1e-10

    # Fitted state (set by fit)
    eigenvalues_: np.ndarray = field(default=None, repr=False)
    eigenvectors_: np.ndarray = field(default=None, repr=False)
    mean_sq_dists_: np.ndarray = field(default=None, repr=False)
    grand_mean_sq_dist_: float = field(default=None, repr=False)
    n_: int = field(default=None, repr=False)
    n_components_: int = field(default=None, repr=False)  # actual dimension used

    def fit(self, D: np.ndarray) -> "ClassicalMDS":
        """Fit MDS from a symmetric distance matrix.

        Parameters
        ----------
        D : ndarray of shape (n, n)
            Pairwise distance matrix (not squared).

        Returns
        -------
        self
        """
        n = D.shape[0]
        self.n_ = n

        # Squared distance matrix
        D2 = D ** 2

        # Store centering stats for out-of-sample projection
        self.mean_sq_dists_ = D2.mean(axis=1)  # row means = col means (symmetric)
        self.grand_mean_sq_dist_ = D2.mean()

        # Double-centering: B = -1/2 * J * D^2 * J where J = I - 1/n * 11^T
        B = -0.5 * (D2 - self.mean_sq_dists_[:, None] - self.mean_sq_dists_[None, :] + self.grand_mean_sq_dist_)

        # Eigendecomposition (B is symmetric PSD in the ideal case)
        eigenvalues, eigenvectors = eigh(B)

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        self.eigenvalues_ = eigenvalues
        self.eigenvectors_ = eigenvectors

        return self

    def fit_transform(self, D: np.ndarray) -> np.ndarray:
        """Fit MDS and return the embedding coordinates.

        Parameters
        ----------
        D : ndarray of shape (n, n)
            Pairwise distance matrix.

        Returns
        -------
        X : ndarray of shape (n, n_components)
            Embedding coordinates.
        """
        self.fit(D)
        return self._compute_coordinates()

    def _compute_coordinates(self) -> np.ndarray:
        """Compute coordinates from fitted eigendecomposition."""
        if self.n_components is None:
            k = select_dimension(self.eigenvalues_, n_elbows=self.n_elbows)
        else:
            k = self.n_components
        k = min(k, len(self.eigenvalues_))
        self.n_components_ = k
        lam = self.eigenvalues_[:k].copy()
        lam[lam < self.min_eig] = 0.0
        X = self.eigenvectors_[:, :k] * np.sqrt(lam)[None, :]
        return X

    def project(self, D_new: np.ndarray) -> np.ndarray:
        """Project new points using Gower's out-of-sample formula.

        Parameters
        ----------
        D_new : ndarray of shape (m, n)
            Distances from m new points to the n training points.

        Returns
        -------
        X_new : ndarray of shape (m, n_components)
            Coordinates of new points in the fitted embedding.
        """
        D_new_sq = D_new ** 2

        # Gower's formula: x_new = -1/2 * L^{-1} * U^T * (d_new^2 - mean_sq_dists)
        # where the centering is: b_new = -1/2 * (d_new^2 - col_means - row_mean_new + grand_mean)
        # But simpler: project onto eigenvectors
        k = self.n_components_ if self.n_components_ is not None else min(self.n_components or len(self.eigenvalues_), len(self.eigenvalues_))
        lam = self.eigenvalues_[:k].copy()
        lam[lam < self.min_eig] = self.min_eig  # avoid division by zero
        U = self.eigenvectors_[:, :k]

        # Double-center the new distance row
        b_new = -0.5 * (D_new_sq - self.mean_sq_dists_[None, :] -
                        D_new_sq.mean(axis=1, keepdims=True) + self.grand_mean_sq_dist_)

        # Project: x_new = b_new @ U @ diag(1/sqrt(lam))
        X_new = b_new @ U / np.sqrt(lam)[None, :]

        return X_new

    @property
    def eigenvalues(self) -> np.ndarray:
        """All eigenvalues from the fitted decomposition (descending order)."""
        if self.eigenvalues_ is None:
            raise ValueError("MDS not fitted yet. Call fit() first.")
        return self.eigenvalues_

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Fraction of total variance explained by each component."""
        eigs = self.eigenvalues.copy()
        eigs[eigs < 0] = 0
        total = eigs.sum()
        if total == 0:
            return np.zeros_like(eigs)
        return eigs / total

    def effective_rank(self, threshold: float = 0.95) -> int:
        """Number of components needed to explain `threshold` fraction of variance."""
        cumvar = np.cumsum(self.explained_variance_ratio)
        return int(np.searchsorted(cumvar, threshold) + 1)
