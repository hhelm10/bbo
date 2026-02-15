"""Abstract base class for black-box generative models."""

from abc import ABC, abstractmethod
import numpy as np


class Model(ABC):
    """A black-box generative model that responds to queries.

    At temperature 0, respond() returns deterministic embedded responses.
    """

    @abstractmethod
    def respond(self, query_indices: np.ndarray) -> np.ndarray:
        """Return embedded responses for given query indices.

        Parameters
        ----------
        query_indices : ndarray of shape (m,)
            Indices into the query set.

        Returns
        -------
        responses : ndarray of shape (m, p)
            Embedded responses g(f(q_k)) for each query.
        """

    @property
    @abstractmethod
    def label(self) -> int:
        """Class label for this model."""
