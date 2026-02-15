"""Query set sampling utilities."""

import numpy as np
from bbo.queries.distributions import QueryDistribution, UniformDistribution


def sample_queries(M: int, m: int, distribution: QueryDistribution = None,
                   rng: np.random.Generator = None,
                   replace: bool = True) -> np.ndarray:
    """Sample m query indices i.i.d. from {0, ..., M-1} according to distribution.

    The theory requires i.i.d. sampling (with replacement) from Pi_Q.

    Parameters
    ----------
    M : int
        Total number of queries.
    m : int
        Number of queries to sample.
    distribution : QueryDistribution, optional
        Sampling distribution. Defaults to uniform.
    rng : numpy random Generator, optional
    replace : bool
        Whether to sample with replacement. Default True (i.i.d.).

    Returns
    -------
    indices : ndarray of shape (m,)
        Sampled query indices.
    """
    if rng is None:
        rng = np.random.default_rng()
    if distribution is None:
        distribution = UniformDistribution()

    probs = distribution.probabilities(M)
    return rng.choice(M, size=m, replace=replace, p=probs)
