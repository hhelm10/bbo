"""Synthetic problem generator for controlled experiments.

Bernoulli-Weight Model:

The discriminative field alpha_l(q) = xi_{ql} * w_{ql} where:
    xi_{ql} ~ Bernoulli(p)   (activation: does query q probe dimension l?)
    w_{ql}  ~ Uniform(0, 1)  (intensity: how strongly?)

Models:
    Each model f has a latent type vector theta_f in {0,1}^r drawn uniformly.
    Class label y = parity(theta_f) XOR Bernoulli(eta).

    The parity label ensures ALL r dimensions must be activated to beat chance.
    No proper subset of dimensions suffices.

Response model (realizes the kernel via orthogonal embedding):
    g(f(q)) = sum_l sqrt(alpha_l(q)) / 2 * (1 - 2*theta_{f,l}) * directions[l]

This gives the exact decomposition:
    ||g(f_i(q)) - g(f_j(q))||^2 = sum_l alpha_l(q) * 1[theta_{i,l} != theta_{j,l}]

Zero sets: Z_l = {q : alpha_l(q) = 0} = {q : xi_{ql} = 0}.
Under uniform Pi_Q: rho_l = Pi_Q(Z_l) = 1 - p.

Theoretical bound: P[error >= 0.5] <= r * (1-p)^m.
This is tight: missing any single dimension makes the parity label unrecoverable.
"""

from dataclasses import dataclass
from typing import List
import numpy as np

from bbo.models.base import Model


class SyntheticModel(Model):
    """A synthetic model with precomputed embedded responses.

    Parameters
    ----------
    embedded_responses : ndarray of shape (M, p_embed)
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
    directions : ndarray of shape (r, p_embed)
        Orthonormal direction vectors for each discriminative dimension.
    p_embed : int
        Embedding dimension.
    """

    M: int
    r: int
    alpha: np.ndarray  # shape (M, r), field magnitudes
    directions: np.ndarray  # shape (r, p_embed), orthonormal rows
    p_embed: int

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

    def generate_models(self, n: int, eta: float = 0.0,
                         rng: np.random.Generator = None) -> List[SyntheticModel]:
        """Generate n synthetic models.

        Each model f has latent type theta_f in {0,1}^r drawn uniformly.
        Label: y = parity(theta_f) XOR Bernoulli(eta).

        The parity label ensures all r dimensions must be activated to beat
        chance. Missing any single dimension makes the label unrecoverable.

        Parameters
        ----------
        n : int
            Total number of models.
        eta : float
            Label noise probability. When eta=0, L*=0 (perfectly separable
            given all dimensions). When eta>0, L*=eta.
        rng : numpy random Generator, optional

        Returns
        -------
        models : list of SyntheticModel
        """
        if rng is None:
            rng = np.random.default_rng()

        models = []
        sqrt_alpha = np.sqrt(self.alpha)  # (M, r)

        for i in range(n):
            # Latent type vector theta in {0,1}^r, drawn uniformly
            theta = rng.integers(0, 2, size=self.r)

            # Parity label: XOR of all dimensions
            label = int(theta.sum() % 2)

            # Label noise
            if eta > 0 and rng.random() < eta:
                label = 1 - label

            # Signs: s_l = 1 - 2*theta_l
            signs = 1.0 - 2.0 * theta.astype(float)

            # g(f(q)) = sum_l sqrt(alpha[q,l]) * (1/2) * s_l * directions[l]
            embedded = (sqrt_alpha * 0.5 * signs) @ self.directions

            model = SyntheticModel(embedded, label, model_id=i)
            models.append(model)

        return models


def make_problem(M: int = 100, r: int = 5, signal_prob: float = 0.3,
                 p_embed: int = 20,
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
    p_embed : int
        Embedding dimension (must be >= r for orthogonal directions).
    rng : numpy random Generator

    Returns
    -------
    problem : SyntheticProblem
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Orthogonal direction vectors via QR decomposition.
    if r > p_embed:
        raise ValueError(
            f"Need r <= p_embed for orthogonal directions, got r={r}, p_embed={p_embed}"
        )
    random_matrix = rng.standard_normal((p_embed, r))
    Q, _ = np.linalg.qr(random_matrix)
    directions = Q[:, :r].T  # (r, p_embed), rows are orthonormal

    # Bernoulli-Weight field: alpha = xi * w
    xi = (rng.random((M, r)) < signal_prob).astype(float)
    w = rng.random((M, r))
    alpha = xi * w

    return SyntheticProblem(
        M=M,
        r=r,
        alpha=alpha,
        directions=directions,
        p_embed=p_embed,
    )


def get_all_responses(models: List[SyntheticModel]) -> np.ndarray:
    """Stack all model responses into a single array.

    Returns
    -------
    responses : ndarray of shape (n_models, M, p_embed)
    """
    return np.stack([m.all_responses for m in models])


def get_labels(models: List[SyntheticModel]) -> np.ndarray:
    """Extract labels from a list of models.

    Returns
    -------
    labels : ndarray of shape (n_models,)
    """
    return np.array([m.label for m in models])
