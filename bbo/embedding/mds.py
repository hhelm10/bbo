"""Classical MDS implementation from scratch.

Implements multidimensional scaling via double-centering and eigendecomposition,
with support for out-of-sample projection (Gower's formula).
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.linalg import eigh


@dataclass
class ClassicalMDS:
    """Classical (metric) multidimensional scaling.

    Parameters
    ----------
    n_components : int
        Number of embedding dimensions to retain.
    min_eig : float
        Eigenvalues below this threshold are zeroed (handles numerical noise).
    """

    n_components: int = 2
    min_eig: float = 1e-10

    # Fitted state (set by fit)
    eigenvalues_: np.ndarray = field(default=None, repr=False)
    eigenvectors_: np.ndarray = field(default=None, repr=False)
    mean_sq_dists_: np.ndarray = field(default=None, repr=False)
    grand_mean_sq_dist_: float = field(default=None, repr=False)
    n_: int = field(default=None, repr=False)

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
        k = min(self.n_components, len(self.eigenvalues_))
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
        k = min(self.n_components, len(self.eigenvalues_))
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
