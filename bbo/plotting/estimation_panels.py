"""Shared 3-panel estimation figure: scree plot, GMM on B_q, failure probability.

Used by both the motivating example (Figure 3) and system prompt experiment
to validate the r̂, ρ̂ estimation procedure.
"""

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.stats import norm as scipy_norm
from scipy.optimize import curve_fit

from bbo.plotting.style import PALETTE
from bbo.distances.energy import per_query_energy_tensor
from bbo.estimation.rank_rho import (
    compute_E_disc,
    estimate_discriminative_rank,
    estimate_rho,
    predict_mstar,
)


def plot_estimation_panels(
    responses: np.ndarray,
    labels: np.ndarray,
    signal_indices: np.ndarray,
    orthogonal_indices: np.ndarray,
    ax_scree,
    ax_gmm,
    ax_fail,
    fail_csv_path=None,
):
    """Plot the 3-panel estimation figure on provided axes.

    (a) Scree plot: singular values of E (all queries), vertical line at r̂
    (b) GMM fit: histogram of B_q with 2-component GMM overlay
    (c) Failure probability: P[err >= 0.5] with r̂ρ̂^m theoretical overlay

    Parameters
    ----------
    responses : ndarray of shape (n_models, M, p)
    labels : ndarray of shape (n_models,)
    signal_indices, orthogonal_indices : ndarray
    ax_scree, ax_gmm, ax_fail : matplotlib Axes
    fail_csv_path : str, optional
        Path to failure_probs.csv with columns: query_set, n, m, failure_prob
    """
    # --- Compute E from ALL queries (signal + orthogonal) ---
    all_idx = np.concatenate([signal_indices, orthogonal_indices])
    E_all, pairs = per_query_energy_tensor(responses[:, all_idx, :])

    # r̂ from scree of full E
    r_hat, U, s = estimate_discriminative_rank(E_all, n_elbows=1)

    # B_q from between-class centering
    _, _, B_q = compute_E_disc(E_all, pairs, labels)

    # ρ̂ from BIC-optimal GMM on B_q (K=1,...,10)
    rho_hat, gmm_info = estimate_rho(B_q)

    n_signal = len(signal_indices)

    # --- Panel (a): Scree plot ---
    sv_norm = s / s[0]
    n_show = min(50, len(sv_norm))
    ax_scree.plot(np.arange(1, n_show + 1), sv_norm[:n_show],
                  color=PALETTE[0], linewidth=1.2, marker="o", markersize=2)

    ax_scree.axvline(x=r_hat, color="0.4", linestyle="--", linewidth=1.2, alpha=0.8)
    ax_scree.text(r_hat + 2.5, 0.85, f"$\\hat{{r}}={r_hat}$",
                  fontsize=5, color="0.3")

    ax_scree.set_xlabel("Component $r$")
    ax_scree.set_ylabel("$\\sigma_r / \\sigma_1$")
    ax_scree.set_title("(a) Singular values of $E$")

    # --- Panel (b): GMM on B_q ---
    gmm_best = gmm_info['gmm']
    gmm1 = gmm_info['gmm1']
    bic1 = gmm_info['bic1']
    bic_best = gmm_info['bic_best']
    k_best = gmm_info['k_best']

    B_signal = B_q[:n_signal]
    B_orth = B_q[n_signal:]

    bins = np.linspace(B_q.min(), B_q.max(), 30)
    ax_gmm.hist(B_signal, bins=bins, alpha=0.6, color=PALETTE[1],
                label="Signal", density=True, edgecolor="none")
    ax_gmm.hist(B_orth, bins=bins, alpha=0.6, color=PALETTE[2],
                label='"Orthogonal"', density=True, edgecolor="none")

    # Density curves
    x_plot = np.linspace(B_q.min() - 0.005, B_q.max() + 0.005, 300)

    # K=1 reference
    m1_mean = gmm1.means_[0, 0]
    m1_std = np.sqrt(gmm1.covariances_[0, 0, 0])
    ax_gmm.plot(x_plot, scipy_norm.pdf(x_plot, m1_mean, m1_std),
                color="0.3", linestyle="--", linewidth=0.8)

    # K=K* components (BIC-selected)
    means_best = gmm_best.means_.ravel()
    zero_comp = int(np.argmin(means_best))
    for k in range(k_best):
        w = gmm_best.weights_[k]
        mu = gmm_best.means_[k, 0]
        std = np.sqrt(gmm_best.covariances_[k, 0, 0])
        comp_density = w * scipy_norm.pdf(x_plot, mu, std)
        color = PALETTE[2] if k == zero_comp else PALETTE[1]
        ax_gmm.fill_between(x_plot, comp_density, alpha=0.2, color=color)
        ax_gmm.plot(x_plot, comp_density, color=color, linewidth=0.8)

    ax_gmm.set_xlabel("$B_q$ (between-class excess)")
    ax_gmm.set_ylabel("Density")
    ax_gmm.set_title("(b) GMM on $B_q$")

    # Main legend (upper right)
    leg_main = [Line2D([0], [0], color=PALETTE[1], lw=4, alpha=0.6, label="Signal"),
                Line2D([0], [0], color=PALETTE[2], lw=4, alpha=0.6, label='"Orthogonal"'),
                Line2D([0], [0], color="0.3", linestyle="--", lw=0.8,
                       label=f"$K\\!=\\!1$ (BIC={bic1:.0f})"),
                Line2D([0], [0], color=PALETTE[2], linestyle="-", lw=0.8,
                       label=f"$K^*\\!=\\!{k_best}$ Near-zero"),
                Line2D([0], [0], color=PALETTE[1], linestyle="-", lw=0.8,
                       label=f"$K^*\\!=\\!{k_best}$ Active (BIC={bic_best:.0f})")]
    leg1 = ax_gmm.legend(handles=leg_main, loc="upper right", fontsize=4)
    ax_gmm.add_artist(leg1)

    # ρ̂ legend (center right)
    leg_rho = [Line2D([], [], linestyle="none",
                      label=f"$\\hat{{\\rho}} = {rho_hat:.2f}$")]
    ax_gmm.legend(handles=leg_rho, loc="center right", fontsize=4,
                  handlelength=0, handletextpad=0)

    # --- Panel (c): Failure probability P[err >= 0.5] ---
    if fail_csv_path is not None and Path(fail_csv_path).exists():
        df_fail = pd.read_csv(fail_csv_path)

        n_colors = {80: PALETTE[0], 10: PALETTE[1]}
        has_n = "n" in df_fail.columns

        for n_val in [80, 10]:
            if has_n:
                sub = df_fail[(df_fail["query_set"] == "uniform") &
                              (df_fail["n"] == n_val)].sort_values("m")
            else:
                if n_val != 80:
                    continue
                sub = df_fail[df_fail["query_set"] == "uniform"].sort_values("m")
            if not sub.empty:
                ax_fail.plot(sub["m"], sub["failure_prob"],
                             marker="o", markersize=2, color=n_colors[n_val],
                             linestyle="-", linewidth=0.8)

        m_max = df_fail["m"].max()
        m_cont = np.linspace(1, m_max, 200)

        # Theoretical bound: r̂ · ρ̂^m
        if rho_hat > 0:
            bound = np.minimum(1.0, r_hat * rho_hat ** m_cont)
            ax_fail.plot(m_cont, bound, color="0.3", linestyle=":",
                         linewidth=0.8, alpha=0.7)

        # Fit a·ρ^m + γ for each n
        def _bound_model(m, a, rho, gamma):
            return a * rho ** m + gamma

        fit_results = {}
        for n_val in [80, 10]:
            if has_n:
                sub_n = df_fail[(df_fail["query_set"] == "uniform") &
                                (df_fail["n"] == n_val)].sort_values("m")
            else:
                if n_val != 80:
                    continue
                sub_n = df_fail[df_fail["query_set"] == "uniform"].sort_values("m")
            if sub_n.empty or len(sub_n) < 3:
                continue
            m_data = sub_n["m"].values.astype(float)
            y_data = sub_n["failure_prob"].values
            try:
                popt, _ = curve_fit(_bound_model, m_data, y_data,
                                    p0=[1.0, 0.5, 0.01],
                                    bounds=([0, 0, 0], [10, 1, 1]))
                a_fit, rho_fit, gamma_fit = popt
                y_fit = _bound_model(m_cont, a_fit, rho_fit, gamma_fit)
                ax_fail.plot(m_cont, y_fit, color=n_colors[n_val],
                             linestyle="--", linewidth=0.8, alpha=0.8)
                fit_results[n_val] = (a_fit, rho_fit, gamma_fit)
            except RuntimeError:
                pass

        # Legend
        leg = [Line2D([0], [0], color=n_colors[80], linestyle="-", lw=0.8,
                       marker="o", markersize=2, label="$n=80$"),
               Line2D([0], [0], color=n_colors[10], linestyle="-", lw=0.8,
                       marker="o", markersize=2, label="$n=10$")]
        if rho_hat > 0:
            leg.append(Line2D([0], [0], color="0.3", linestyle=":", lw=0.8,
                              label=f"$\\hat{{r}}\\hat{{\\rho}}^m$"
                                    f"\n($\\hat{{r}}\\!={r_hat},"
                                    f"\\,\\hat{{\\rho}}\\!={rho_hat:.2f}$)"))
        if fit_results:
            leg.append(Line2D([0], [0], color="0.5", linestyle="--", lw=0.8,
                              label="Fit ($r\\rho^m\\!+\\!\\gamma$)"))
        ax_fail.legend(handles=leg, loc="upper right", fontsize=4)

        # Annotate fit expressions near curves
        annot_cfg = {80: {"m": 6, "va": "top", "offset": -0.04},
                     10: {"m": 6, "va": "bottom", "offset": 0.04}}
        for n_val in [80, 10]:
            if n_val not in fit_results:
                continue
            a_f, rho_f, gamma_f = fit_results[n_val]
            txt = f"${a_f:.2f}\\!\\cdot\\!{rho_f:.2f}^m\\!+\\!{gamma_f:.2f}$"
            cfg = annot_cfg[n_val]
            y_annot = _bound_model(cfg["m"], a_f, rho_f, gamma_f)
            ax_fail.text(cfg["m"], y_annot + cfg["offset"], txt,
                         fontsize=4, color=n_colors[n_val],
                         ha="center", va=cfg["va"],
                         bbox=dict(boxstyle="round,pad=0.15",
                                   facecolor="white", edgecolor="none",
                                   alpha=0.7))

        ax_fail.set_xscale("log")
        ax_fail.set_ylim(-0.02, 1.05)
        ax_fail.set_xlabel("Number of queries $m$")
        ax_fail.set_ylabel(r"$\mathbb{P}[\mathrm{err} \geq 0.5]$")
        ax_fail.set_title("(c) Failure probability")
