"""Synthetic problem generator for controlled experiments.

Creates classification problems with known discriminative rank r and
controlled noise levels. All responses are precomputed for speed.

Response model (additive):
    g(f_i(q)) = signal_strength * sum_l A[q,l] * z_i(l) * directions[l]

where:
    A[q,l] ~ Bernoulli(signal_prob)  (independent per query per dimension)
    z_i(l) in {+1, -1}              (per-model, per-dimension sign with noise)
    directions[l] in R^p             (unit direction vector for dimension l)

The per-query squared distance decomposes as:
    ||diff(q)||^2 = sum_l alpha_l(q) * phi_l(f_i, f_j)
with alpha_l(q) = s^2 * A[q,l]^2 >= 0 and phi_l(i,j) = (z_i(l) - z_j(l))^2 >= 0.

Under uniform Pi_Q, rho_l = 1 - signal_prob for each l.
"""

from dataclasses import dataclass
from typing import List
import numpy as np

from bbo.models.base import Model


class SyntheticModel(Model):
    """A synthetic model with precomputed embedded responses.

    Parameters
    ----------
    embedded_responses : ndarray of shape (M, p)
        Precomputed g(f(q)) for all M queries.
    label_ : int
        Class label.
    model_id : int
        Unique identifier.
    """

    def __init__(self, embedded_responses: np.ndarray, label_: int, model_id: int = 0):
        self._embedded_responses = embedded_responses
        self._label = label_
        self._model_id = model_id

    def respond(self, query_indices: np.ndarray) -> np.ndarray:
        return self._embedded_responses[query_indices]

    @property
    def label(self) -> int:
        return self._label

    @property
    def all_responses(self) -> np.ndarray:
        """All M embedded responses."""
        return self._embedded_responses


@dataclass
class SyntheticProblem:
    """A synthetic classification problem with controlled structure.

    Parameters
    ----------
    M : int
        Total number of queries.
    r : int
        Discriminative rank (number of signal dimensions).
    A : ndarray of shape (M, r)
        Sensitivity matrix. A[q,l] = 1 if query q carries signal along
        dimension l.
    directions : ndarray of shape (r, p)
        Unit direction vectors for each discriminative dimension.
    p : int
        Embedding dimension.
    noise_level : float
        Per-query per-dimension sign flip probability.
    signal_strength : float
        Amplitude of discriminative signal.
    """

    M: int
    r: int
    A: np.ndarray  # shape (M, r)
    directions: np.ndarray  # shape (r, p)
    p: int
    noise_level: float = 0.0
    signal_strength: float = 1.0

    @property
    def sensitivity_matrix(self) -> np.ndarray:
        """Binary matrix A[q, l]."""
        return self.A

    @property
    def orthogonal_queries(self) -> np.ndarray:
        """Indices of queries in Q_perp (not active for any dimension)."""
        return np.where(self.A.sum(axis=1) == 0)[0]

    @property
    def signal_queries(self) -> np.ndarray:
        """Indices of all signal queries (active for at least one dimension)."""
        return np.where(self.A.sum(axis=1) > 0)[0]

    @property
    def rho(self) -> float:
        """Maximum zero-set probability under uniform distribution.

        rho = max_l |Z_l| / M where Z_l = {q : A[q,l] = 0}
        """
        zero_counts = (self.A == 0).sum(axis=0)  # per dimension
        return zero_counts.max() / self.M

    def generate_models(self, n: int, rng: np.random.Generator = None) -> List[SyntheticModel]:
        """Generate n synthetic models (n/2 per class).

        For each model i with class label y_i:
          - base sign = +1 if y_i == 0, -1 if y_i == 1
          - For each dimension l:
            sign z_i(l) = base_sign, flipped independently with prob noise_level
          - response[q] = signal_strength * sum_l A[q,l] * z_i(l) * directions[l]

        Signs are per-model per-dimension (NOT per-query), so that
        phi_l(f, f') = (z_i(l) - z_j(l))^2 is query-independent,
        matching the paper's discriminative factorization.

        Parameters
        ----------
        n : int
            Total number of models (split evenly between classes).
        rng : numpy random Generator, optional

        Returns
        -------
        models : list of SyntheticModel
        """
        if rng is None:
            rng = np.random.default_rng()

        n_per_class = n // 2
        models = []

        for class_label in [0, 1]:
            base_sign = 1.0 if class_label == 0 else -1.0

            for i in range(n_per_class):
                # Per-dimension signs (query-independent)
                signs = np.full(self.r, base_sign)
                if self.noise_level > 0:
                    flip_mask = rng.random(self.r) < self.noise_level
                    signs[flip_mask] *= -1

                # (M, r) * (r,) -> (M, r) then @ (r, p) -> (M, p)
                signal = (self.A * signs) @ self.directions
                embedded = self.signal_strength * signal

                model = SyntheticModel(embedded, class_label,
                                       model_id=class_label * n_per_class + i)
                models.append(model)

        return models


def make_problem(M: int = 100, r: int = 5, signal_prob: float = 0.2,
                 noise_level: float = 0.0, p: int = 20,
                 rng: np.random.Generator = None,
                 signal_strength: float = 1.0) -> SyntheticProblem:
    """Create a standard synthetic problem.

    Each query independently contributes to each dimension with probability
    signal_prob. Under uniform Pi_Q, this gives rho = 1 - signal_prob.

    Parameters
    ----------
    M : int
        Number of queries.
    r : int
        Discriminative rank.
    signal_prob : float
        Probability each query is active for each dimension.
        rho = 1 - signal_prob under uniform distribution.
    noise_level : float
        Per-query per-dimension sign flip probability.
    p : int
        Embedding dimension.
    rng : numpy random Generator
    signal_strength : float
        Amplitude of discriminative signal.

    Returns
    -------
    problem : SyntheticProblem
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Random unit direction vectors for each discriminative dimension
    directions = rng.standard_normal((r, p))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    # Independent Bernoulli sensitivity matrix
    A = (rng.random((M, r)) < signal_prob).astype(float)

    return SyntheticProblem(
        M=M,
        r=r,
        A=A,
        directions=directions,
        p=p,
        noise_level=noise_level,
        signal_strength=signal_strength,
    )


def get_all_responses(models: List[SyntheticModel]) -> np.ndarray:
    """Stack all model responses into a single array.

    Returns
    -------
    responses : ndarray of shape (n_models, M, p)
    """
    return np.stack([m.all_responses for m in models])


def get_labels(models: List[SyntheticModel]) -> np.ndarray:
    """Extract labels from a list of models.

    Returns
    -------
    labels : ndarray of shape (n_models,)
    """
    return np.array([m.label for m in models])
