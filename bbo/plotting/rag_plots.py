"""3-panel figure for the RAG compliance auditing experiment.

(a) Scree plot of Ẽ with dashed line at r̂
(b) Stacked histograms of |Ũ_{q,ℓ}| for ℓ=1,2, colored by signal/control, with K=1,2 GMM
(c) Failure probability vs m: empirical (n=10,80), fitted, theoretical bound
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.stats import norm as scipy_norm
from scipy.optimize import curve_fit

from bbo.plotting.style import set_paper_style, PALETTE
from bbo.distances.energy import per_query_energy_tensor
from bbo.estimation.rank_rho import (
    compute_E_disc,
    estimate_discriminative_rank,
    estimate_rho,
    predict_mstar,
)


def plot_rag_figure(npz_path, fail_csv_path, output_path="figures/figure_rag.pdf"):
    """Generate the 3-panel RAG experiment figure."""

    set_paper_style()

    data = np.load(str(npz_path), allow_pickle=True)
    responses = data["responses"]
    labels = data["labels"]
    signal_indices = data["signal_indices"]
    control_indices = data["control_indices"]

    n_signal = len(signal_indices)
    n_control = len(control_indices)

    # --- Estimation ---
    all_idx = np.concatenate([signal_indices, control_indices])
    E_all, pairs = per_query_energy_tensor(responses[:, all_idx, :])
    E_disc, _, B_q = compute_E_disc(E_all, pairs, labels)

    r_hat, U, s = estimate_discriminative_rank(E_disc)
    rho_hats, gmm_info = estimate_rho(U, r_hat)

    # --- Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.8))
    ax_scree, ax_gmm, ax_fail = axes

    # =====================================================================
    # Panel (a): Scree plot
    # =====================================================================
    sv_norm = s / s[0]
    n_show = min(20, len(sv_norm))
    ax_scree.plot(
        np.arange(1, n_show + 1), sv_norm[:n_show],
        color=PALETTE[0], linewidth=1.2, marker="o", markersize=2,
    )
    ax_scree.axvline(
        x=r_hat, color="0.4", linestyle="--", linewidth=1.2, alpha=0.8,
    )
    ax_scree.text(
        r_hat + 2.0, 0.82, f"$\\hat{{r}}={r_hat}$",
        fontsize=5, color="0.3",
    )
    ax_scree.set_xlabel("Component $r$")
    ax_scree.set_ylabel("$\\sigma_r / \\sigma_1$")
    ax_scree.set_title("(a) Singular values of $\\tilde{E}$")

    # =====================================================================
    # Panel (b): Stacked histograms for directions ℓ=1 and ℓ=2
    # =====================================================================
    # Two rows inside one axes using manual positioning is complex;
    # instead, use two vertically-stacked sub-histograms via inset axes.
    ax_gmm.set_visible(False)

    gs_inner = ax_gmm.get_subplotspec().subgridspec(2, 1, hspace=0.45)
    ax_top = fig.add_subplot(gs_inner[0])
    ax_bot = fig.add_subplot(gs_inner[1])

    for ell, ax_h in enumerate([ax_top, ax_bot]):
        dir_info = gmm_info["per_direction"][ell]
        loadings = dir_info["loadings"]
        gmm2 = dir_info["gmm"]
        gmm1 = dir_info["gmm1"]
        bic1 = dir_info["bic1"]
        bic2 = dir_info["bic2"]
        rho_l = dir_info["rho_l"]

        L_signal = loadings[:n_signal]
        L_control = loadings[n_signal:]

        bins = np.linspace(loadings.min(), loadings.max(), 28)

        ax_h.hist(
            L_signal, bins=bins, alpha=0.6, color=PALETTE[1],
            label="Signal", density=True, edgecolor="none",
        )
        ax_h.hist(
            L_control, bins=bins, alpha=0.6, color=PALETTE[2],
            label="Control", density=True, edgecolor="none",
        )

        x_plot = np.linspace(
            loadings.min() - 0.005, loadings.max() + 0.005, 300,
        )

        # K=1 density
        m1_mean = gmm1.means_[0, 0]
        m1_std = np.sqrt(gmm1.covariances_[0, 0, 0])
        ax_h.plot(
            x_plot, scipy_norm.pdf(x_plot, m1_mean, m1_std),
            color="0.3", linestyle="--", linewidth=0.8,
        )

        # K=2 components
        comp_colors = [PALETTE[2], PALETTE[1]]
        means_k2 = gmm2.means_.ravel()
        zero_comp = int(np.argmin(means_k2))
        for k in range(2):
            w = gmm2.weights_[k]
            mu = gmm2.means_[k, 0]
            std = np.sqrt(gmm2.covariances_[k, 0, 0])
            comp_density = w * scipy_norm.pdf(x_plot, mu, std)
            cidx = 0 if k == zero_comp else 1
            ax_h.fill_between(
                x_plot, comp_density, alpha=0.2, color=comp_colors[cidx],
            )
            ax_h.plot(
                x_plot, comp_density, color=comp_colors[cidx], linewidth=0.8,
            )

        ax_h.set_xlabel(f"$|\\tilde{{U}}_{{q,{ell+1}}}|$")
        if ell == 0:
            ax_h.set_ylabel("Density")

        # Compact rho annotation
        ax_h.text(
            0.97, 0.92,
            f"$\\hat{{\\rho}}_{ell+1}\\!=\\!{rho_l:.2f}$",
            transform=ax_h.transAxes, fontsize=4.5,
            ha="right", va="top",
            bbox=dict(
                boxstyle="round,pad=0.15", facecolor="white",
                edgecolor="none", alpha=0.7,
            ),
        )

    # Title spanning both sub-rows
    ax_top.set_title("(b) GMM on $|\\tilde{U}_{q,\\ell}|$")

    # Single legend on top axes
    leg_handles = [
        Line2D([0], [0], color=PALETTE[1], lw=4, alpha=0.6, label="Signal"),
        Line2D([0], [0], color=PALETTE[2], lw=4, alpha=0.6, label="Control"),
        Line2D([0], [0], color="0.3", linestyle="--", lw=0.8, label="$K\\!=\\!1$"),
        Line2D([0], [0], color="0.3", linestyle="-", lw=0.8, label="$K\\!=\\!2$"),
    ]
    ax_top.legend(handles=leg_handles, loc="upper right", fontsize=3.5, ncol=2)

    # =====================================================================
    # Panel (c): Failure probability
    # =====================================================================
    df_fail = pd.read_csv(fail_csv_path)
    df_sig = df_fail[df_fail["query_set"] == "signal"]

    n_colors = {80: PALETTE[0], 10: PALETTE[1]}
    m_cont = np.linspace(1, 100, 300)

    # Theoretical bound: Σ_ℓ ρ̂_ℓ^m
    bound = np.minimum(
        1.0,
        np.sum([rho_l ** m_cont for rho_l in rho_hats], axis=0),
    )
    ax_fail.plot(
        m_cont, bound, color="0.3", linestyle=":", linewidth=0.8, alpha=0.7,
    )

    def _bound_model(m, a, rho, gamma):
        return a * rho ** m + gamma

    fit_results = {}
    for n_val in [80, 10]:
        sub = df_sig[df_sig["n"] == n_val].sort_values("m")
        if sub.empty:
            continue

        # Empirical points
        ax_fail.plot(
            sub["m"], sub["failure_prob"],
            marker="o", markersize=2, color=n_colors[n_val],
            linestyle="-", linewidth=0.8,
        )

        # Fit a·ρ^m + γ
        m_data = sub["m"].values.astype(float)
        y_data = sub["failure_prob"].values
        if len(m_data) >= 3:
            try:
                popt, _ = curve_fit(
                    _bound_model, m_data, y_data,
                    p0=[1.0, 0.5, 0.01],
                    bounds=([0, 0, 0], [10, 1, 1]),
                )
                a_fit, rho_fit, gamma_fit = popt
                y_fit = _bound_model(m_cont, a_fit, rho_fit, gamma_fit)
                ax_fail.plot(
                    m_cont, y_fit, color=n_colors[n_val],
                    linestyle="--", linewidth=0.8, alpha=0.8,
                )
                fit_results[n_val] = (a_fit, rho_fit, gamma_fit)
            except RuntimeError:
                pass

    # Legend
    leg = [
        Line2D([0], [0], color=n_colors[80], linestyle="-", lw=0.8,
               marker="o", markersize=2, label="$n=80$"),
        Line2D([0], [0], color=n_colors[10], linestyle="-", lw=0.8,
               marker="o", markersize=2, label="$n=10$"),
        Line2D([0], [0], color="0.3", linestyle=":", lw=0.8,
               label="$\\sum_\\ell \\hat{\\rho}_\\ell^m$"),
    ]
    if fit_results:
        leg.append(
            Line2D([0], [0], color="0.5", linestyle="--", lw=0.8,
                   label="Fit ($a\\rho^m\\!+\\!\\gamma$)"),
        )
    ax_fail.legend(handles=leg, loc="upper right", fontsize=4)

    # Annotate fit expressions
    annot_cfg = {
        80: {"m": 8, "va": "top", "offset": -0.04},
        10: {"m": 8, "va": "bottom", "offset": 0.04},
    }
    for n_val, (a_f, rho_f, gamma_f) in fit_results.items():
        txt = f"${a_f:.1f}\\!\\cdot\\!{rho_f:.2f}^m\\!+\\!{gamma_f:.2f}$"
        cfg = annot_cfg[n_val]
        y_annot = _bound_model(cfg["m"], a_f, rho_f, gamma_f)
        ax_fail.text(
            cfg["m"], y_annot + cfg["offset"], txt,
            fontsize=4, color=n_colors[n_val],
            ha="center", va=cfg["va"],
            bbox=dict(
                boxstyle="round,pad=0.15", facecolor="white",
                edgecolor="none", alpha=0.7,
            ),
        )

    ax_fail.set_xscale("log")
    ax_fail.set_ylim(-0.02, 1.05)
    ax_fail.set_xlabel("Number of queries $m$")
    ax_fail.set_ylabel(r"$\mathbb{P}[\mathrm{err} \geq 0.5]$")
    ax_fail.set_title("(c) Failure probability")

    # --- Save ---
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved figure to {output_path}")

    # Print summary
    rho_str = ", ".join(f"{r:.2f}" for r in rho_hats)
    print(f"  r_hat={r_hat}, rho_hats=[{rho_str}]")
    print(f"  m*(95%)={predict_mstar(rho_hats, 0.05)}")
    for n_val, (a_f, rho_f, gamma_f) in fit_results.items():
        print(f"  Fit n={n_val}: {a_f:.2f}*{rho_f:.3f}^m + {gamma_f:.3f}")
