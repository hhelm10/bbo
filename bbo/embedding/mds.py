"""Classical MDS via graspologic.

Wraps graspologic's ClassicalMDS (with dissimilarity='precomputed') to provide
the same interface used throughout the BBO codebase. Automatic dimensionality
selection uses the Zhu & Ghodsi (2006) profile likelihood method.
"""

from typing import Optional

import numpy as np
from graspologic.embed import ClassicalMDS as _GraspologicMDS
from graspologic.embed import select_dimension as _graspologic_select_dimension


def select_dimension(eigenvalues: np.ndarray, n_elbows: int = 2) -> int:
    """Select embedding dimension via the profile likelihood method.

    Thin wrapper around graspologic.embed.select_dimension that accepts
    a 1-d eigenvalue array and returns the dimension at the last elbow.

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
    elbows, _ = _graspologic_select_dimension(pos, n_elbows=n_elbows)
    if not elbows:
        return 1
    return elbows[-1]


class ClassicalMDS:
    """Classical (metric) multidimensional scaling on precomputed distances.

    Delegates to graspologic.embed.ClassicalMDS with dissimilarity='precomputed'.

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
        self._mds = _GraspologicMDS(
            n_components=n_components,
            n_elbows=n_elbows,
            dissimilarity="precomputed",
        )

    def fit(self, D: np.ndarray) -> "ClassicalMDS":
        """Fit MDS from a symmetric distance matrix."""
        self._mds.fit(D)
        return self

    def fit_transform(self, D: np.ndarray) -> np.ndarray:
        """Fit MDS and return the embedding coordinates."""
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
