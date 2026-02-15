"""Exp 6: Effective rank of the energy distance tensor.

Compute SVD of the M x n_pairs energy distance tensor.
Expected: clear spectral gap with r_eff << M.
"""

import numpy as np
import pandas as pd

from bbo.distances.energy import per_query_energy_tensor


def run_exp6(responses: np.ndarray) -> dict:
    """Run Exp 6: compute energy distance tensor and its SVD.

    Parameters
    ----------
    responses : ndarray of shape (n_models, M, p)

    Returns
    -------
    result : dict with keys:
        - singular_values: ndarray
        - r90, r95, r99: effective rank at 90/95/99% variance
        - cumulative_variance: ndarray
    """
    T, pairs = per_query_energy_tensor(responses)

    # SVD of the energy distance tensor
    U, s, Vt = np.linalg.svd(T, full_matrices=False)

    # Cumulative explained variance
    sv_sq = s ** 2
    total = sv_sq.sum()
    cumvar = np.cumsum(sv_sq) / total if total > 0 else np.zeros_like(sv_sq)

    r90 = int(np.searchsorted(cumvar, 0.9) + 1)
    r95 = int(np.searchsorted(cumvar, 0.95) + 1)
    r99 = int(np.searchsorted(cumvar, 0.99) + 1)

    return {
        "singular_values": s,
        "cumulative_variance": cumvar,
        "r90": r90,
        "r95": r95,
        "r99": r99,
        "n_pairs": len(pairs),
        "M": responses.shape[1],
    }
