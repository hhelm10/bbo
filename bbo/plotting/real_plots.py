"""Plotting functions for real LLM experiments (Exp 6-10)."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.stats import norm as scipy_norm
from scipy.optimize import curve_fit

from bbo.plotting.style import set_paper_style, PALETTE
from bbo.distances.energy import per_query_energy_tensor
from bbo.estimation.rank_rho import (
    estimate_discriminative_rank,
    estimate_rho,
)


def plot_exp6_scree(singular_values: np.ndarray, output_dir: str = "results/figures"):
    """Plot Exp 6: scree plot of singular value spectrum."""
    set_paper_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Raw singular values
    ax1.plot(range(1, len(singular_values) + 1), singular_values,
             marker="o", markersize=3, color=PALETTE[0])
    ax1.set_xlabel("Component index")
    ax1.set_ylabel("Singular value")
    ax1.set_title("Singular value spectrum")

    # Cumulative explained variance
    sv_sq = singular_values ** 2
    cumvar = np.cumsum(sv_sq) / sv_sq.sum()
    ax2.plot(range(1, len(cumvar) + 1), cumvar,
             marker="o", markersize=3, color=PALETTE[1])
    ax2.axhline(y=0.9, color="gray", linestyle=":", alpha=0.5)
    ax2.axhline(y=0.95, color="gray", linestyle=":", alpha=0.5)
    ax2.set_xlabel("Number of components")
    ax2.set_ylabel("Cumulative explained variance")
    ax2.set_title("Effective rank")

    # Mark 90% and 95% thresholds
    r90 = np.searchsorted(cumvar, 0.9) + 1
    r95 = np.searchsorted(cumvar, 0.95) + 1
    ax2.annotate(f"90%: r={r90}", xy=(r90, 0.9), fontsize=10)
    ax2.annotate(f"95%: r={r95}", xy=(r95, 0.95), fontsize=10)

    fig.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/exp6_scree_plot.pdf")
    plt.close(fig)


def plot_exp7(df: pd.DataFrame, output_dir: str = "results/figures"):
    """Plot Exp 7: mean accuracy +/- std vs m."""
    set_paper_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.errorbar(df["m"], df["mean_accuracy"], yerr=df["std_accuracy"],
                marker="o", capsize=3, color=PALETTE[0])
    ax.set_xlabel("Number of queries $m$")
    ax.set_ylabel("Classification accuracy")
    ax.set_title("Accuracy vs number of queries (real data)")
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylim(0.4, 1.05)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/exp7_accuracy_vs_m.pdf")
    plt.close(fig)


def plot_exp8(df: pd.DataFrame, output_dir: str = "results/figures"):
    """Plot Exp 8: accuracy vs m for relevant/orthogonal/uniform."""
    set_paper_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    dist_labels = {"uniform": "Uniform", "relevant": "Relevant",
                   "orthogonal": "Orthogonal"}

    for i, (dist_name, label) in enumerate(dist_labels.items()):
        sub = df[df["distribution"] == dist_name]
        ax.plot(sub["m"], sub["mean_accuracy"],
                marker="o", markersize=3, color=PALETTE[i], label=label)

    ax.set_xlabel("Number of queries $m$")
    ax.set_ylabel("Classification accuracy")
    ax.set_title("Relevant vs orthogonal queries (real data)")
    ax.legend()
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/exp8_relevant_vs_orthogonal.pdf")
    plt.close(fig)


def plot_exp9(df: pd.DataFrame, output_dir: str = "results/figures"):
    """Plot Exp 9: accuracy vs m for different methods (baselines)."""
    set_paper_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    methods = df["method"].unique()
    for i, method in enumerate(methods):
        sub = df[df["method"] == method]
        ax.plot(sub["m"], sub["mean_accuracy"],
                marker="o", markersize=3, color=PALETTE[i], label=method)

    ax.set_xlabel("Number of queries $m$")
    ax.set_ylabel("Classification accuracy")
    ax.set_title("MDS embedding vs baselines")
    ax.legend()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/exp9_baselines.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3×3 Real Data Figure
# ---------------------------------------------------------------------------

def _plot_scree(ax, responses, labels, signal_indices, orthogonal_indices):
    """Col 1: Scree plot of E (raw) with r̂ dashed line."""
    all_idx = np.concatenate([signal_indices, orthogonal_indices])
    E_all, pairs = per_query_energy_tensor(responses[:, all_idx, :])
    r_hat, U, s = estimate_discriminative_rank(E_all)
    rho_hats, gmm_info = estimate_rho(U, r_hat)

    sv_norm = s / s[0]
    n_show = min(20, len(sv_norm))
    ax.plot(np.arange(1, n_show + 1), sv_norm[:n_show],
            color=PALETTE[0], linewidth=1.2, marker="o", markersize=2)
    ax.axvline(x=r_hat, color="0.4", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(r_hat + 1.5, 0.82, f"$\\hat{{r}}={r_hat}$",
            fontsize=5, color="0.3")
    ax.set_xlabel("Component $r$")
    ax.set_ylabel("$\\sigma_r / \\sigma_1$")

    return r_hat, U, s, rho_hats, gmm_info, len(signal_indices)


def _plot_gmm_single(ax, gmm_info, r_hat, rho_hats, n_signal, direction=0):
    """Plot a single-direction GMM histogram on ax."""
    dir_info = gmm_info["per_direction"][direction]
    loadings = dir_info["loadings"]
    gmm2 = dir_info["gmm"]
    gmm1 = dir_info["gmm1"]
    bic1 = dir_info["bic1"]
    bic2 = dir_info["bic2"]
    rho_l = dir_info["rho_l"]

    L_signal = loadings[:n_signal]
    L_orth = loadings[n_signal:]

    bins = np.linspace(loadings.min(), loadings.max(), 28)
    ax.hist(L_signal, bins=bins, alpha=0.6, color=PALETTE[1],
            label="Signal", density=True, edgecolor="none")
    ax.hist(L_orth, bins=bins, alpha=0.6, color=PALETTE[2],
            label="Orthogonal", density=True, edgecolor="none")

    x_plot = np.linspace(loadings.min() - 0.005, loadings.max() + 0.005, 300)

    # K=1 density
    m1_mean = gmm1.means_[0, 0]
    m1_std = np.sqrt(gmm1.covariances_[0, 0, 0])
    ax.plot(x_plot, scipy_norm.pdf(x_plot, m1_mean, m1_std),
            color="0.3", linestyle="--", linewidth=0.8)

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
        ax.fill_between(x_plot, comp_density, alpha=0.2, color=comp_colors[cidx])
        ax.plot(x_plot, comp_density, color=comp_colors[cidx], linewidth=0.8)

    ax.set_xlabel("$|\\hat{\\alpha}_{q,\\ell}|$")
    ax.set_ylabel("Density")

    # ρ̂ annotation
    ax.text(0.97, 0.12,
            f"$\\hat{{\\rho}}_{direction+1}\\!=\\!{rho_l:.2f}$",
            transform=ax.transAxes, fontsize=4.5,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.7))

    # K+BIC legend
    leg_k = [
        Line2D([0], [0], color="0.3", linestyle="--", lw=0.8,
               label=f"$K\\!=\\!1$ (BIC$={bic1:.0f}$)"),
        Line2D([0], [0], color="0.3", linestyle="-", lw=0.8,
               label=f"$K\\!=\\!2$ (BIC$={bic2:.0f}$)"),
    ]
    ax.legend(handles=leg_k, loc="upper right", fontsize=3.5)


def _plot_failure_prob(ax, fail_csv, rho_hats=None, query_set="uniform"):
    """Col 3: P[err >= 0.5] vs m with fitted curves and theory bound."""
    df = pd.read_csv(fail_csv)

    n_colors = {80: PALETTE[0], 10: PALETTE[1]}
    m_cont = np.linspace(1, 100, 300)

    def _bound_model(m, a, rho, gamma):
        return a * rho ** m + gamma

    fit_results = {}
    for n_val in [80, 10]:
        sub = df[(df["query_set"] == query_set) & (df["n"] == n_val)].sort_values("m")
        if sub.empty:
            continue
        m_data = sub["m"].values.astype(float)
        y_data = sub["failure_prob"].values

        ax.plot(m_data, y_data, marker="o", markersize=2,
                color=n_colors[n_val], linestyle="-", linewidth=0.8)

        if len(m_data) >= 3:
            try:
                popt, _ = curve_fit(
                    _bound_model, m_data, y_data,
                    p0=[1.0, 0.5, 0.01],
                    bounds=([0, 0, 0], [10, 1, 1]),
                )
                y_fit = _bound_model(m_cont, *popt)
                ax.plot(m_cont, y_fit, color=n_colors[n_val],
                        linestyle="--", linewidth=0.8, alpha=0.8)
                fit_results[n_val] = popt
            except RuntimeError:
                pass

    # Theory bound: Σ_ℓ ρ̂_ℓ^m
    if rho_hats is not None and np.any(np.array(rho_hats) > 0):
        theory = np.minimum(
            1.0, np.sum([rho_l ** m_cont for rho_l in rho_hats], axis=0))
        ax.plot(m_cont, theory, color="0.3", linestyle=":", linewidth=0.8,
                alpha=0.7)

    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Queries $m$")
    ax.set_ylabel("$\\mathbb{P}[\\mathrm{err} \\geq 0.5]$")

    # Legend
    leg = [Line2D([0], [0], color=n_colors[n], linestyle="-", lw=0.8,
                  marker="o", markersize=2, label=f"$n={n}$")
           for n in [80, 10]]
    if rho_hats is not None:
        leg.append(Line2D([0], [0], color="0.3", linestyle=":", lw=0.8,
                          label="$\\sum_\\ell \\hat{\\rho}_\\ell^m$"))
    if fit_results:
        leg.append(Line2D([0], [0], color="0.5", linestyle="--", lw=0.8,
                          label="Fit ($a\\rho^m\\!+\\!\\gamma$)"))
    ax.legend(handles=leg, loc="upper right", fontsize=3.5)

    # Annotate fit parameters
    sorted_n = sorted(fit_results.keys(), reverse=True)
    for i, n_val in enumerate(sorted_n):
        a_f, rho_f, gamma_f = fit_results[n_val]
        txt = (f"${a_f:.2f}\\!\\cdot\\!{rho_f:.2f}^m"
               f"\\!+\\!{gamma_f:.2f}$")
        offset = -0.04 if i == 0 else 0.04
        va = "top" if i == 0 else "bottom"
        m_annot = 6
        y_annot = _bound_model(m_annot, a_f, rho_f, gamma_f)
        ax.text(m_annot, y_annot + offset, txt,
                fontsize=4, color=n_colors[n_val],
                ha="center", va=va,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.7))


def _plot_mean_error(ax, classification_csv, rho_hats=None,
                     method="mds", dist="signal"):
    """Col 3 (RAG): Mean error vs m with fitted a·ρ^m + γ and theory bound."""
    df = pd.read_csv(classification_csv)

    if "method" in df.columns:
        df_plot = df[(df["method"] == method) & (df["distribution"] == dist)]
    else:
        df_plot = df[df["distribution"] == dist]

    n_values = sorted(df_plot["n"].unique())
    n_colors = {n: PALETTE[i] for i, n in enumerate(n_values)}
    m_cont = np.linspace(1, 100, 300)

    def _error_model(m, a, rho, gamma):
        return a * rho ** m + gamma

    fit_results = {}
    for n_val in n_values:
        sub = df_plot[df_plot["n"] == n_val].sort_values("m")
        if sub.empty:
            continue
        m_data = sub["m"].values.astype(float)
        err = 1.0 - sub["mean_accuracy"].values
        ax.plot(m_data, err, marker="o", markersize=2,
                color=n_colors[n_val], linestyle="-", linewidth=0.8)

        if len(m_data) >= 3:
            try:
                popt, _ = curve_fit(
                    _error_model, m_data, err,
                    p0=[0.3, 0.5, 0.01],
                    bounds=([0, 0, 0], [2, 1, 0.5]),
                )
                y_fit = _error_model(m_cont, *popt)
                ax.plot(m_cont, y_fit, color=n_colors[n_val],
                        linestyle="--", linewidth=0.8, alpha=0.8)
                fit_results[n_val] = popt
            except RuntimeError:
                pass

    # Theory rate: Σ_ℓ ρ̂_ℓ^m
    if rho_hats is not None and np.any(np.array(rho_hats) > 0):
        theory = np.sum([rho_l ** m_cont for rho_l in rho_hats], axis=0)
        ax.plot(m_cont, theory, color="0.3", linestyle=":", linewidth=0.8,
                alpha=0.7)

    ax.set_xscale("log")
    ax.set_ylim(-0.02, 0.45)
    ax.set_xlabel("Queries $m$")
    ax.set_ylabel("Mean error")

    # Legend
    leg = [Line2D([0], [0], color=n_colors[n], linestyle="-", lw=0.8,
                  marker="o", markersize=2, label=f"$n={n}$")
           for n in n_values]
    if fit_results:
        leg.append(Line2D([0], [0], color="0.5", linestyle="--", lw=0.8,
                          label="Fit ($a\\rho^m\\!+\\!\\gamma$)"))
    if rho_hats is not None:
        leg.append(Line2D([0], [0], color="0.3", linestyle=":", lw=0.8,
                          label="$\\sum_\\ell \\hat{\\rho}_\\ell^m$"))
    ax.legend(handles=leg, loc="upper right", fontsize=3.5)

    # Annotate fit parameters
    sorted_n = sorted(fit_results.keys(), reverse=True)
    for i, n_val in enumerate(sorted_n):
        a_f, rho_f, gamma_f = fit_results[n_val]
        txt = (f"${a_f:.2f}\\!\\cdot\\!{rho_f:.2f}^m"
               f"\\!+\\!{gamma_f:.3f}$")
        offset = -0.02 if i == 0 else 0.02
        va = "top" if i == 0 else "bottom"
        m_annot = 8
        y_annot = _error_model(m_annot, a_f, rho_f, gamma_f)
        ax.text(m_annot, y_annot + offset, txt,
                fontsize=4, color=n_colors[n_val],
                ha="center", va=va,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.7))


def plot_real_data_3x3(
    motivating_npz: str,
    motivating_fail_csv: str,
    system_prompt_npz: str,
    system_prompt_fail_csv: str,
    rag_npz: str,
    rag_classification_csv: str,
    output_path: str = "figures/figure_real_3x3.pdf",
):
    """3×3 real data figure.

    Rows: Motivating, System Prompt, RAG
    Cols: Scree plot, GMM on |Ũ_{q,ℓ}|, col 3 metric
      - Motivating & SysPrompt col 3: P[err >= 0.5] (Theorem 1)
      - RAG col 3: Mean error (Theorem 2)

    RAG row col 2 uses subgridspec for two stacked GMM histograms (ℓ=1, ℓ=2).
    """
    set_paper_style()
    from matplotlib.legend import Legend

    fig = plt.figure(figsize=(5.5, 3.8))
    gs = gridspec.GridSpec(3, 3, figure=fig,
                           left=0.10, right=0.98, bottom=0.08, top=0.94,
                           wspace=0.40, hspace=0.20)

    row_labels = ["Motivating", "System Prompt", "RAG"]
    datasets = [
        {
            "npz": motivating_npz,
            "signal_key": "sensitive_indices",
            "orth_key": "orthogonal_indices",
            "col3": "fail",
            "col3_path": motivating_fail_csv,
            "col3_kw": {"query_set": "uniform"},
        },
        {
            "npz": system_prompt_npz,
            "signal_key": "signal_indices",
            "orth_key": "orthogonal_indices",
            "col3": "fail",
            "col3_path": system_prompt_fail_csv,
            "col3_kw": {"query_set": "uniform"},
        },
        {
            "npz": rag_npz,
            "signal_key": "signal_indices",
            "orth_key": "control_indices",
            "col3": "mean_error",
            "col3_path": rag_classification_csv,
            "col3_kw": {"dist": "signal"},
        },
    ]

    n_rows = len(row_labels)

    for row_idx, (label, ds) in enumerate(zip(row_labels, datasets)):
        is_last = row_idx == n_rows - 1
        data = np.load(ds["npz"], allow_pickle=True)
        responses = data["responses"]
        labels = data["labels"]
        signal_idx = data[ds["signal_key"]]
        orth_idx = data[ds["orth_key"]]

        # --- Col 1: Scree plot ---
        ax_scree = fig.add_subplot(gs[row_idx, 0])
        r_hat, U, s, rho_hats, gmm_info, n_signal = _plot_scree(
            ax_scree, responses, labels, signal_idx, orth_idx,
        )
        if row_idx == 0:
            ax_scree.set_title("Singular values of $E$")
        if not is_last:
            ax_scree.set_xlabel("")
            ax_scree.set_xticklabels([])

        # Row label
        ax_scree.annotate(
            label, xy=(-0.50, 0.5), xycoords="axes fraction",
            fontsize=7, fontweight="bold", rotation=90,
            ha="center", va="center",
        )

        # --- Col 2: GMM (single direction ℓ=1 for all rows) ---
        ax_gmm = fig.add_subplot(gs[row_idx, 1])
        _plot_gmm_single(ax_gmm, gmm_info, r_hat, rho_hats, n_signal,
                         direction=0)
        if row_idx == 0:
            ax_gmm.set_title("GMM on $|\\hat{\\alpha}_{q,\\ell}|$")
        if not is_last:
            ax_gmm.set_xlabel("")
            ax_gmm.set_xticklabels([])
        ax_gmm.set_ylabel("Density")

        leg_sc = [
            Line2D([0], [0], color=PALETTE[1], lw=4, alpha=0.6, label="Signal"),
            Line2D([0], [0], color=PALETTE[2], lw=4, alpha=0.6, label="Orthogonal"),
        ]
        leg2 = Legend(ax_gmm, leg_sc, ["Signal", "Orthogonal"],
                      loc="upper left", fontsize=3.5)
        ax_gmm.add_artist(leg2)

        # --- Col 3 ---
        ax_c3 = fig.add_subplot(gs[row_idx, 2])
        if ds["col3"] == "fail":
            _plot_failure_prob(ax_c3, ds["col3_path"], rho_hats=rho_hats,
                               **ds["col3_kw"])
            if row_idx == 0:
                ax_c3.set_title("$\\mathbb{P}[\\mathrm{err} \\geq 0.5]$ vs $m$")
        else:
            _plot_mean_error(ax_c3, ds["col3_path"], rho_hats=rho_hats,
                             **ds["col3_kw"])
            if row_idx == 0:
                ax_c3.set_title("Mean error vs $m$")

        if not is_last:
            ax_c3.set_xlabel("")
            ax_c3.set_xticklabels([])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved 3×3 real data figure to {output_path}")
