"""Query sampling distributions.

Defines distributions over the query set Q for controlling which queries
are selected during experiments.
"""

from abc import ABC, abstractmethod
import numpy as np


class QueryDistribution(ABC):
    """Abstract base for query sampling distributions."""

    @abstractmethod
    def probabilities(self, M: int) -> np.ndarray:
        """Return probability vector over M queries.

        Parameters
        ----------
        M : int
            Total number of queries.

        Returns
        -------
        probs : ndarray of shape (M,)
            Probability of selecting each query. Sums to 1.
        """


class UniformDistribution(QueryDistribution):
    """Uniform distribution over all queries."""

    def probabilities(self, M: int) -> np.ndarray:
        return np.ones(M) / M


class SubsetDistribution(QueryDistribution):
    """Distribution that concentrates mass on a subset of queries.

    Places total probability `mass` on `indices`, distributed uniformly
    within that subset, and spreads the remaining (1 - mass) uniformly
    over all other queries.

    Parameters
    ----------
    indices : array-like
        Indices of queries to concentrate on.
    mass : float
        Total probability mass on the subset (0 < mass <= 1).
    """

    def __init__(self, indices: np.ndarray, mass: float = 0.9):
        self.indices = np.asarray(indices)
        self.mass = mass

    def probabilities(self, M: int) -> np.ndarray:
        n_subset = len(self.indices)
        n_other = M - n_subset

        probs = np.zeros(M)
        if n_subset > 0:
            probs[self.indices] = self.mass / n_subset
        if n_other > 0:
            other_mask = np.ones(M, dtype=bool)
            other_mask[self.indices] = False
            probs[other_mask] = (1.0 - self.mass) / n_other

        return probs
