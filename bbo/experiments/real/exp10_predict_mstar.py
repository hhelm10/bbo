"""Exp 10: Estimating rho and predicting m*.

Fit the theoretical decay curve r*rho^m to observed error-vs-m data.
Use estimated rho to predict m* for target accuracy.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def _decay_model(m, r, rho):
    """Theoretical error decay: P[error >= 0.5] = r * rho^m."""
    return r * np.power(rho, m)


def fit_decay_curve(m_values: np.ndarray, error_probs: np.ndarray) -> dict:
    """Fit r*rho^m to observed P[error >= 0.5] vs m.

    Parameters
    ----------
    m_values : ndarray
    error_probs : ndarray
        Observed P[error >= 0.5] for each m.

    Returns
    -------
    result : dict with r_hat, rho_hat, and predicted m* for various targets.
    """
    # Filter positive values
    mask = error_probs > 0
    if mask.sum() < 2:
        return {"r_hat": np.nan, "rho_hat": np.nan}

    m_pos = m_values[mask]
    e_pos = error_probs[mask]

    try:
        popt, pcov = curve_fit(_decay_model, m_pos, e_pos,
                               p0=[5.0, 0.8],
                               bounds=([0.1, 0.01], [1000, 0.999]),
                               maxfev=10000)
        r_hat, rho_hat = popt
    except RuntimeError:
        return {"r_hat": np.nan, "rho_hat": np.nan}

    # Predict m* for target accuracies
    predictions = {}
    for target_acc in [0.8, 0.9, 0.95]:
        target_error_prob = 1.0 - target_acc
        if rho_hat > 0 and rho_hat < 1 and r_hat > 0:
            # r * rho^m = target => m = log(target/r) / log(rho)
            if target_error_prob / r_hat > 0:
                m_star = np.log(target_error_prob / r_hat) / np.log(rho_hat)
                predictions[f"m_star_{int(target_acc*100)}"] = max(1, int(np.ceil(m_star)))
            else:
                predictions[f"m_star_{int(target_acc*100)}"] = np.nan
        else:
            predictions[f"m_star_{int(target_acc*100)}"] = np.nan

    return {
        "r_hat": r_hat,
        "rho_hat": rho_hat,
        **predictions,
    }
