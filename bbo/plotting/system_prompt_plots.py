"""Plotting functions for the system prompt auditing experiment.

Figure 3 panels (a,b,c): Estimation and theory validation
  (a) Scree plot of E (all queries)
  (b) GMM fit on B_q (per-query between-class excess)
  (c) P[error >= 0.5] with theoretical overlay

Figure 4 row 1 (a,b,c): Quantitative results (error vs m)

Style matches motivating_plots.py (Figure 3) and synthetic_plots.py (Figure 4).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.stats import norm as scipy_norm

from bbo.plotting.style import set_paper_style, PALETTE
from bbo.distances.energy import pairwise_energy_distances_t0, per_query_energy_tensor
from bbo.embedding.mds import ClassicalMDS
from bbo.estimation.rank_rho import (
    compute_E_disc,
    estimate_discriminative_rank,
    estimate_rho,
    predict_mstar,
)


def plot_figure3_system_prompt(
    responses: np.ndarray,
    labels: np.ndarray,
    signal_indices: np.ndarray,
    orthogonal_indices: np.ndarray,
    ax_scree,
    ax_gmm,
    ax_fail,
    fail_csv_path=None,
):
    """Plot Figure 3 panels (a), (b), (c) for the system prompt experiment.

    (a) Scree plot: singular values of E (all queries), vertical line at r̂
    (b) GMM fit: histogram of B_q with 2-component GMM overlay
    (c) Failure probability: P[err >= 0.5] with r̂ρ̂^m theoretical overlay
    """
    # --- Compute E from ALL queries (signal + orthogonal) ---
    all_idx = np.concatenate([signal_indices, orthogonal_indices])
    E_all, pairs = per_query_energy_tensor(responses[:, all_idx, :])

    # r̂ from scree of full E
    r_hat, U, s = estimate_discriminative_rank(E_all, n_elbows=1)

    # B_q from between-class centering
    _, _, B_q = compute_E_disc(E_all, pairs, labels)

    # ρ̂ from 2-component GMM on B_q
    rho_hat, gmm_info = estimate_rho(B_q)

    n_signal = len(signal_indices)

    # --- Panel (a): Scree plot ---
    sv_norm = s / s[0]
    n_show = min(50, len(sv_norm))
    ax_scree.plot(np.arange(1, n_show + 1), sv_norm[:n_show],
                  color=PALETTE[0], linewidth=1.2,
                  marker="o", markersize=2)

    # Vertical line at r̂
    ax_scree.axvline(x=r_hat, color="0.4", linestyle="--", linewidth=1.2, alpha=0.8)
    ax_scree.text(r_hat + 2.5, 0.85, f"$\\hat{{r}}={r_hat}$",
                  fontsize=5, color="0.3")

    ax_scree.set_xlabel("Component $r$")
    ax_scree.set_ylabel("$\\sigma_r / \\sigma_1$")
    ax_scree.set_title("(a) Singular values of $E$")

    # --- Panel (b): GMM fit on B_q ---
    gmm2 = gmm_info['gmm']
    gmm1 = gmm_info['gmm1']
    bic1 = gmm_info['bic1']
    bic2 = gmm_info['bic2']

    # Histogram colored by signal vs orthogonal
    B_signal = B_q[:n_signal]
    B_orth = B_q[n_signal:]

    bins = np.linspace(B_q.min(), B_q.max(), 30)
    ax_gmm.hist(B_signal, bins=bins, alpha=0.6, color=PALETTE[1],
                label="Signal", density=True, edgecolor="none")
    ax_gmm.hist(B_orth, bins=bins, alpha=0.6, color=PALETTE[2],
                label='"Orthogonal"', density=True, edgecolor="none")

    # Overlay density curves
    x_plot = np.linspace(B_q.min() - 0.002, B_q.max() + 0.002, 300)
    x_col = x_plot.reshape(-1, 1)

    # K=1: single Gaussian
    m1_mean = gmm1.means_[0, 0]
    m1_std = np.sqrt(gmm1.covariances_[0, 0, 0])
    ax_gmm.plot(x_plot, scipy_norm.pdf(x_plot, m1_mean, m1_std),
                color="0.3", linestyle="--", linewidth=0.8)

    # K=2: plot each component as weighted density with shaded fill
    comp_colors = [PALETTE[2], PALETTE[1]]  # near-zero, active
    means_k2 = gmm2.means_.ravel()
    zero_comp = int(np.argmin(means_k2))
    for k in range(2):
        w = gmm2.weights_[k]
        mu = gmm2.means_[k, 0]
        std = np.sqrt(gmm2.covariances_[k, 0, 0])
        comp_density = w * scipy_norm.pdf(x_plot, mu, std)
        cidx = 0 if k == zero_comp else 1
        ax_gmm.fill_between(x_plot, comp_density, alpha=0.2,
                            color=comp_colors[cidx])
        ax_gmm.plot(x_plot, comp_density, color=comp_colors[cidx],
                    linewidth=0.8)

    # Annotate ρ̂
    ax_gmm.text(0.97, 0.95,
                f"$\\hat{{\\rho}} = {rho_hat:.2f}$",
                transform=ax_gmm.transAxes, fontsize=5, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="0.7", alpha=0.8))

    ax_gmm.set_xlabel("$B_q$ (between-class excess)")
    ax_gmm.set_ylabel("Density")
    ax_gmm.set_title("(b) GMM on $B_q$")

    # Legend: histograms + GMM fits with BIC
    leg_hist = [Line2D([0], [0], color=PALETTE[1], lw=4, alpha=0.6, label="Signal"),
                Line2D([0], [0], color=PALETTE[2], lw=4, alpha=0.6, label='"Orthogonal"')]
    leg_gmm = [Line2D([0], [0], color="0.3", linestyle="--", lw=0.8,
                       label=f"$K\\!=\\!1$ (BIC={bic1:.0f})"),
               Line2D([0], [0], color=PALETTE[2], linestyle="-", lw=0.8,
                       label=f"Near-zero ($\\pi_0\\!={rho_hat:.2f}$)"),
               Line2D([0], [0], color=PALETTE[1], linestyle="-", lw=0.8,
                       label=f"Active ($\\pi_1\\!={1-rho_hat:.2f}$)")]
    ax_gmm.legend(handles=leg_hist + leg_gmm, loc="upper left", fontsize=4)

    # --- Panel (c): Failure probability P[err >= 0.5] ---
    if fail_csv_path is not None and Path(fail_csv_path).exists():
        from scipy.optimize import curve_fit

        df_fail = pd.read_csv(fail_csv_path)

        # Plot empirical curves for each n value
        n_colors = {80: PALETTE[0], 10: PALETTE[1]}

        # Check if CSV has 'n' column (new format) or not (old format)
        has_n = "n" in df_fail.columns

        for n_val in [80, 10]:
            if has_n:
                sub = df_fail[(df_fail["query_set"] == "uniform") &
                              (df_fail["n"] == n_val)].sort_values("m")
            else:
                # Old format: only n=80
                if n_val != 80:
                    continue
                sub = df_fail[df_fail["query_set"] == "uniform"].sort_values("m")
            if not sub.empty:
                ax_fail.plot(sub["m"], sub["failure_prob"],
                             marker="o", markersize=2, color=n_colors[n_val],
                             linestyle="-", linewidth=0.8)

        m_max = df_fail["m"].max()
        m_cont = np.linspace(1, m_max, 200)

        # Theoretical bound: r̂ · ρ̂^m (from B_q GMM estimate)
        if rho_hat > 0:
            bound = np.minimum(1.0, r_hat * rho_hat ** m_cont)
            ax_fail.plot(m_cont, bound, color="0.3", linestyle=":",
                         linewidth=0.8, alpha=0.7,
                         label=f"$\\hat{{r}}\\hat{{\\rho}}^m$"
                               f" ($\\hat{{\\rho}}\\!={rho_hat:.2f}$)")

        # Fit a·ρ^m + γ to empirical curves for each n
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

        # Legend (short labels)
        leg = [Line2D([0], [0], color=n_colors[80], linestyle="-", lw=0.8,
                       marker="o", markersize=2, label="$n=80$"),
               Line2D([0], [0], color=n_colors[10], linestyle="-", lw=0.8,
                       marker="o", markersize=2, label="$n=10$")]
        if rho_hat > 0:
            leg.append(Line2D([0], [0], color="0.3", linestyle=":", lw=0.8,
                              label=f"$\\hat{{r}}\\hat{{\\rho}}^m$"
                                    f" ($\\hat{{\\rho}}\\!={rho_hat:.2f}$)"))
        # Single "Fit" entry since both use dashed version of their empirical color
        if fit_results:
            leg.append(Line2D([0], [0], color="0.5", linestyle="--", lw=0.8,
                              label="Fit"))
        ax_fail.legend(handles=leg, loc="upper right", fontsize=4)

        # Annotate fit expressions near the curves
        annot_cfg = {80: {"m": 4, "va": "top", "offset": -0.04},
                     10: {"m": 4, "va": "bottom", "offset": 0.04}}
        for n_val in [80, 10]:
            if n_val not in fit_results:
                continue
            a_f, rho_f, gamma_f = fit_results[n_val]
            txt = f"${a_f:.2f}\\!\\cdot\\!{rho_f:.2f}^m\\!+\\!{gamma_f:.3f}$"
            cfg = annot_cfg[n_val]
            y_annot = _bound_model(cfg["m"], a_f, rho_f, gamma_f)
            ax_fail.text(cfg["m"], y_annot + cfg["offset"], txt,
                         fontsize=3.5, color=n_colors[n_val],
                         ha="center", va=cfg["va"])
        ax_fail.set_xscale("log")
        ax_fail.set_ylim(-0.02, 1.05)
        ax_fail.set_xlabel("Number of queries $m$")
        ax_fail.set_ylabel(r"$\mathbb{P}[\mathrm{err} \geq 0.5]$")
        ax_fail.set_title("(c) Failure probability")


def plot_figure4_row1(
    config,
    ax_a,
    ax_b,
    ax_c,
    default_base_model: str = "ministral-8b",
    default_embedding_model: str = "nomic-embed-text-v1.5",
):
    """Plot Figure 4 row 1: quantitative results for system prompt experiment.

    Style matches motivating panel (b) and figure_combined conventions:
    - n-value as color, query distribution as linestyle
    - Consistent PALETTE usage
    - Legend with ncol=2
    """
    # --- Panel (a): Error vs m by query type + baselines ---
    # Style: match motivating panel (b) — n as color, dist as linestyle
    csv_path = config.classification_csv(default_base_model, default_embedding_model)
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df_mds = df[df["method"] == "mds"]

        available_n = sorted(df_mds["n"].unique())
        n_colors = {n: PALETTE[i] for i, n in enumerate(available_n)}

        dist_styles = {"relevant": "-", "orthogonal": "--"}
        dist_labels = {"relevant": "Signal", "orthogonal": '"Orthogonal"'}

        for n in available_n:
            sub_n = df_mds[df_mds["n"] == n]
            for dist_name, ls in dist_styles.items():
                sub = sub_n[sub_n["distribution"] == dist_name].sort_values("m")
                if sub.empty:
                    continue
                ax_a.plot(sub["m"], 1 - sub["mean_accuracy"], marker="o",
                          markersize=2, color=n_colors[n], linestyle=ls,
                          linewidth=0.8)

        # Concat baseline (largest n, signal queries)
        n_plot = available_n[-1]
        df_concat = df[(df["method"] == "concat") & (df["n"] == n_plot)]
        if not df_concat.empty:
            ax_a.plot(df_concat.sort_values("m")["m"],
                      1 - df_concat.sort_values("m")["mean_accuracy"],
                      marker="^", markersize=2, color="0.4", linestyle="-.",
                      linewidth=0.8, label="Concat")

    ax_a.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, linewidth=0.5)
    ax_a.set_xscale("log")
    ax_a.set_ylim(0, 0.55)
    ax_a.set_xlabel("Number of queries $m$")
    ax_a.set_ylabel("Mean error")
    ax_a.set_title("(a) Error vs $m$")

    # Legend: n colors + dist linestyles + concat
    leg_n = [Line2D([0], [0], color=n_colors[n], lw=1.0, label=f"$n={n}$")
             for n in available_n]
    leg_dist = [Line2D([0], [0], color="0.4", linestyle=ls, lw=1.0,
                        label=dist_labels[name])
                for name, ls in dist_styles.items()]
    leg_base = []
    if csv_path.exists() and not df_concat.empty:
        leg_base = [Line2D([0], [0], color="0.4", marker="^", linestyle="-.",
                           markersize=3, lw=0.8, label="Concat")]
    ax_a.legend(handles=leg_n + leg_dist + leg_base, loc="upper right",
                ncol=2, fontsize=4)

    # --- Panel (b): Error vs m across base models ---
    all_n = sorted(df["n"].unique()) if csv_path.exists() else sorted(config.n_values)
    n_max, n_min = all_n[-1], all_n[0]
    n_line_styles = {n_max: "-", n_min: "--"} if len(all_n) >= 2 else {n_max: "-"}

    for i, bm in enumerate(config.base_models):
        bm_csv = config.classification_csv(bm, default_embedding_model)
        if not bm_csv.exists():
            continue
        bm_df = pd.read_csv(bm_csv)
        bm_mds = bm_df[(bm_df["method"] == "mds") & (bm_df["distribution"] == "relevant")]
        color = PALETTE[i % len(PALETTE)]

        for n_val, ls in n_line_styles.items():
            sub = bm_mds[bm_mds["n"] == n_val].sort_values("m")
            if sub.empty:
                continue
            ax_b.plot(sub["m"], 1 - sub["mean_accuracy"], marker="o",
                      markersize=2, color=color, linestyle=ls, linewidth=0.8)

    ax_b.set_xscale("log")
    ax_b.set_ylim(0, 0.55)
    ax_b.set_xlabel("Number of queries $m$")
    ax_b.set_yticklabels([])
    ax_b.set_title("(b) Across base models")

    # Legend: model colors + n line styles
    handles_b = [Line2D([0], [0], color=PALETTE[i], lw=1, label=bm)
                 for i, bm in enumerate(config.base_models)
                 if config.classification_csv(bm, default_embedding_model).exists()]
    handles_b += [Line2D([0], [0], color="0.4", linestyle=ls, lw=1,
                          label=f"$n={n}$")
                  for n, ls in n_line_styles.items()]
    ax_b.legend(handles=handles_b, loc="upper right", fontsize=4, ncol=2)

    # --- Panel (c): Error vs m across embedding models ---
    for i, em in enumerate(config.embedding_models):
        em_csv = config.classification_csv(default_base_model, em)
        if not em_csv.exists():
            continue
        em_df = pd.read_csv(em_csv)
        em_mds = em_df[(em_df["method"] == "mds") & (em_df["distribution"] == "relevant")]
        color = PALETTE[i % len(PALETTE)]

        for n_val, ls in n_line_styles.items():
            sub = em_mds[em_mds["n"] == n_val].sort_values("m")
            if sub.empty:
                continue
            ax_c.plot(sub["m"], 1 - sub["mean_accuracy"], marker="o",
                      markersize=2, color=color, linestyle=ls, linewidth=0.8)

    ax_c.set_xscale("log")
    ax_c.set_ylim(0, 0.55)
    ax_c.set_xlabel("Number of queries $m$")
    ax_c.set_yticklabels([])
    ax_c.set_title("(c) Across embeddings")

    handles_c = [Line2D([0], [0], color=PALETTE[i], lw=1,
                         label=em.split("/")[-1])
                 for i, em in enumerate(config.embedding_models)
                 if config.classification_csv(default_base_model, em).exists()]
    handles_c += [Line2D([0], [0], color="0.4", linestyle=ls, lw=1,
                          label=f"$n={n}$")
                  for n, ls in n_line_styles.items()]
    ax_c.legend(handles=handles_c, loc="upper right", fontsize=4, ncol=2)


def plot_system_prompt_figures(config, output_dir: str = "figures"):
    """Generate Figure 3 (a,b,c) and Figure 4 row 1 (a,b,c) for system prompt experiment."""
    set_paper_style()

    # Load default data (ministral-8b + nomic)
    default_bm = "ministral-8b"
    default_em = "nomic-embed-text-v1.5"
    npz_path = config.npz_path(default_bm, default_em)

    if not npz_path.exists():
        print(f"Embeddings not found at {npz_path}. Run embed step first.")
        return

    data = np.load(str(npz_path), allow_pickle=True)
    responses = data["responses"]
    labels = data["labels"]
    signal_indices = data["signal_indices"]
    orthogonal_indices = data["orthogonal_indices"]

    # --- Figure 3: Estimation and theory validation (3 panels) ---
    fail_csv = Path("results/system_prompt/failure_probs.csv")

    fig3 = plt.figure(figsize=(5.5, 1.65))
    gs3 = GridSpec(1, 3, figure=fig3,
                   left=0.08, right=0.97, bottom=0.22, top=0.82, wspace=0.45)

    ax3_a = fig3.add_subplot(gs3[0, 0])
    ax3_b = fig3.add_subplot(gs3[0, 1])
    ax3_c = fig3.add_subplot(gs3[0, 2])

    plot_figure3_system_prompt(
        responses, labels, signal_indices, orthogonal_indices,
        ax3_a, ax3_b, ax3_c,
        fail_csv_path=str(fail_csv),
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig3.savefig(f"{output_dir}/figure3_system_prompt.pdf")
    fig3.savefig(f"{output_dir}/figure3_system_prompt.png", dpi=200)
    plt.close(fig3)
    print(f"Saved Figure 3 (system prompt) to {output_dir}/figure3_system_prompt.pdf")

    # --- Figure 4 row 1: Quantitative ---
    # Layout matches figure_combined: 12-column grid with explicit margins
    fig4 = plt.figure(figsize=(5.5, 1.65))
    gs4 = GridSpec(1, 12, figure=fig4,
                   left=0.07, right=0.99, bottom=0.18, top=0.85,
                   wspace=0.4)

    ax4_a = fig4.add_subplot(gs4[0, 0:4])
    ax4_b = fig4.add_subplot(gs4[0, 4:8])
    ax4_c = fig4.add_subplot(gs4[0, 8:12])

    plot_figure4_row1(config, ax4_a, ax4_b, ax4_c,
                      default_base_model=default_bm,
                      default_embedding_model=default_em)

    fig4.savefig(f"{output_dir}/figure4_row1_system_prompt.pdf")
    plt.close(fig4)
    print(f"Saved Figure 4 row 1 to {output_dir}/figure4_row1_system_prompt.pdf")
