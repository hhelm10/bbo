"""Classical MDS via graspologic.

Wraps graspologic's ClassicalMDS (with dissimilarity='precomputed') to provide
the same interface used throughout the BBO codebase. Automatic dimensionality
selection uses the Zhu & Ghodsi (2006) profile likelihood method on the full
eigenvalue spectrum (not the truncated spectrum that graspologic uses internally).
"""

from typing import Optional

import numpy as np
from graspologic.embed import ClassicalMDS as _GraspologicMDS
from graspologic.embed.svd import _compute_likelihood


def select_dimension(eigenvalues: np.ndarray, n_elbows: int = 2) -> int:
    """Select embedding dimension via the profile likelihood method.

    Runs the Zhu & Ghodsi (2006) iterative elbow-finding algorithm on the
    full eigenvalue spectrum. Each subsequent elbow expands beyond the
    previous one.

    Parameters
    ----------
    eigenvalues : ndarray
        Eigenvalues in descending order.
    n_elbows : int, default=2
        Number of elbows to find. The dimension at the last elbow is returned.

    Returns
    -------
    d : int
        Selected number of dimensions (>= 1).
    """
    pos = eigenvalues[eigenvalues > 0]
    if len(pos) <= 1:
        return max(1, len(pos))

    idx = 0
    elbows = []
    for _ in range(n_elbows):
        arr = pos[idx:]
        if arr.size <= 1:
            break
        lq = _compute_likelihood(arr)
        idx += np.argmax(lq).item() + 1
        elbows.append(idx)

    if not elbows:
        return 1
    return elbows[-1]


class ClassicalMDS:
    """Classical (metric) multidimensional scaling on precomputed distances.

    When n_components is None, selects dimensionality by running the Zhu &
    Ghodsi profile likelihood method on the full eigenvalue spectrum of the
    double-centered distance matrix, then delegates to graspologic for the
    actual embedding.

    Parameters
    ----------
    n_components : int or None
        Number of embedding dimensions. If None, automatically selected via
        the Zhu & Ghodsi (2006) profile likelihood method.
    n_elbows : int
        Number of elbows for automatic dimension selection (only used when
        n_components is None).
    """

    def __init__(self, n_components: Optional[int] = None, n_elbows: int = 2):
        self.n_components = n_components
        self.n_elbows = n_elbows

    def _resolve_n_components(self, D: np.ndarray) -> int:
        """Determine n_components from the full eigenvalue spectrum."""
        n = D.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H @ (D ** 2) @ H
        eigvals = np.linalg.eigvalsh(B)[::-1]
        return select_dimension(eigvals, n_elbows=self.n_elbows)

    def _build_mds(self, D: np.ndarray) -> _GraspologicMDS:
        """Build graspologic MDS with resolved n_components."""
        if self.n_components is None:
            nc = self._resolve_n_components(D)
        else:
            nc = self.n_components
        return _GraspologicMDS(
            n_components=nc,
            dissimilarity="precomputed",
        )

    def fit(self, D: np.ndarray) -> "ClassicalMDS":
        """Fit MDS from a symmetric distance matrix."""
        self._mds = self._build_mds(D)
        self._mds.fit(D)
        return self

    def fit_transform(self, D: np.ndarray) -> np.ndarray:
        """Fit MDS and return the embedding coordinates."""
        self._mds = self._build_mds(D)
        return self._mds.fit_transform(D)

    @property
    def n_components_(self) -> int:
        """Actual number of components used (after auto-selection)."""
        return self._mds.n_components_

    @property
    def singular_values_(self) -> np.ndarray:
        """Singular values from the fitted decomposition."""
        return self._mds.singular_values_

    def __repr__(self):
        return f"ClassicalMDS(n_components={self.n_components}, n_elbows={self.n_elbows})"
