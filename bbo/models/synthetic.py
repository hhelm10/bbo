"""Synthetic problem generator for controlled experiments.

Bernoulli-Weight Model (from computational_model.md):

The discriminative field alpha_l(q) = xi_{ql} * w_{ql} where:
    xi_{ql} ~ Bernoulli(p)   (activation: does query q probe dimension l?)
    w_{ql}  ~ Uniform(0, 1)  (intensity: how strongly?)

Models:
    Each model f has a latent type vector theta_f in {0,1}^r.
    Class label y = theta_{f,1}. Dimensions 2,...,r are random (Bernoulli(0.5)).

Response model:
    g(f_i(q)) = sum_l sqrt(alpha_l(q)) * s_l(f_i) * directions[l]

where s_l(f) = 1 - 2*theta_{f,l} in {+1, -1}.

The per-query squared distance decomposes exactly (orthogonal directions):
    ||diff(q)||^2 = sum_l alpha_l(q) * phi_l(f_i, f_j)

with alpha_l(q) >= 0 (field magnitude) and
     phi_l(i,j) = (s_l(f_i) - s_l(f_j))^2 = 4 * 1[theta_{f,l} != theta_{f',l}].

Dimension 1 carries class signal (phi_1 = 4 for cross-class, 0 for within-class).
Dimensions 2,...,r carry noise (phi_l = 4 with prob 0.5 regardless of class).

Zero sets: Z_l = {q : alpha_l(q) = 0} = {q : xi_{ql} = 0}.
Under uniform Pi_Q: rho_l = Pi_Q(Z_l) = 1 - p.
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
    """A synthetic classification problem with controlled discriminative structure.

    Parameters
    ----------
    M : int
        Total number of queries.
    r : int
        Discriminative rank (number of latent dimensions).
    alpha : ndarray of shape (M, r)
        Field magnitudes. alpha[q,l] = xi[q,l] * w[q,l] >= 0.
        Zero iff xi[q,l] = 0 (query q does not activate dimension l).
    directions : ndarray of shape (r, p)
        Orthonormal direction vectors for each discriminative dimension.
    p : int
        Embedding dimension.
    """

    M: int
    r: int
    alpha: np.ndarray  # shape (M, r), field magnitudes
    directions: np.ndarray  # shape (r, p), orthonormal rows
    p: int

    @property
    def sensitivity_matrix(self) -> np.ndarray:
        """Binary activation matrix: 1 where alpha > 0."""
        return (self.alpha > 0).astype(float)

    @property
    def orthogonal_queries(self) -> np.ndarray:
        """Indices of queries in Q_perp (alpha_l(q) = 0 for all l)."""
        return np.where((self.alpha > 0).sum(axis=1) == 0)[0]

    @property
    def signal_queries(self) -> np.ndarray:
        """Indices of queries active for at least one dimension."""
        return np.where((self.alpha > 0).sum(axis=1) > 0)[0]

    @property
    def query_total_signal(self) -> np.ndarray:
        """Total signal intensity per query: sum_l alpha_l(q). Shape (M,)."""
        return self.alpha.sum(axis=1)

    @property
    def rho(self) -> float:
        """Maximum zero-set probability under uniform distribution.

        rho = max_l |Z_l| / M where Z_l = {q : alpha_l(q) = 0}.
        Since alpha_l(q) = 0 iff xi_{ql} = 0, rho ~ 1 - signal_prob.
        """
        zero_counts = (self.alpha == 0).sum(axis=0)  # per dimension
        return zero_counts.max() / self.M

    def generate_models(self, n: int, rng: np.random.Generator = None) -> List[SyntheticModel]:
        """Generate n synthetic models (n/2 per class).

        Each model f has latent type theta_f in {0,1}^r:
          - theta_{f,1} = class label (0 or 1)
          - theta_{f,l} ~ Bernoulli(0.5) for l >= 2 (random noise dimensions)

        Signs: s_l(f) = 1 - 2*theta_{f,l} in {+1, -1}.
        Response: g(f(q)) = sum_l sqrt(alpha[q,l]) * s_l(f) * directions[l]

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
        sqrt_alpha = np.sqrt(self.alpha)  # (M, r)

        for class_label in [0, 1]:
            for i in range(n_per_class):
                # Latent type vector theta in {0,1}^r
                theta = rng.integers(0, 2, size=self.r)
                theta[0] = class_label  # dimension 1 determines class

                # Signs: s_l = 1 - 2*theta_l
                signs = 1.0 - 2.0 * theta.astype(float)

                # g(f(q)) = sum_l sqrt(alpha[q,l]) * s_l * directions[l]
                embedded = (sqrt_alpha * signs) @ self.directions

                model = SyntheticModel(embedded, class_label,
                                       model_id=class_label * n_per_class + i)
                models.append(model)

        return models


def make_problem(M: int = 100, r: int = 5, signal_prob: float = 0.3,
                 p: int = 20,
                 rng: np.random.Generator = None) -> SyntheticProblem:
    """Create a Bernoulli-Weight synthetic problem.

    The field alpha_l(q) = xi_{ql} * w_{ql} where xi ~ Bern(signal_prob),
    w ~ Uniform(0,1). This gives rho = 1 - signal_prob under uniform Pi_Q.

    Parameters
    ----------
    M : int
        Number of queries.
    r : int
        Discriminative rank (number of latent dimensions).
    signal_prob : float
        Activation probability p. Each query activates each dimension
        independently with this probability. rho = 1 - signal_prob.
    p : int
        Embedding dimension (must be >= r for orthogonal directions).
    rng : numpy random Generator

    Returns
    -------
    problem : SyntheticProblem
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Orthogonal direction vectors via QR decomposition.
    if r > p:
        raise ValueError(f"Need r <= p for orthogonal directions, got r={r}, p={p}")
    random_matrix = rng.standard_normal((p, r))
    Q, _ = np.linalg.qr(random_matrix)
    directions = Q[:, :r].T  # (r, p), rows are orthonormal

    # Bernoulli-Weight field: alpha = xi * w
    xi = (rng.random((M, r)) < signal_prob).astype(float)
    w = rng.random((M, r))
    alpha = xi * w

    return SyntheticProblem(
        M=M,
        r=r,
        alpha=alpha,
        directions=directions,
        p=p,
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
