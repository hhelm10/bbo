"""Plotting functions for the motivating example figure."""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
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
from bbo.experiments.real.exp6_effective_rank import run_exp6


def plot_motivating_figure(
    responses: np.ndarray,
    labels: np.ndarray,
    sensitive_indices: np.ndarray,
    orthogonal_indices: np.ndarray,
    metadata_path: str,
    classification_csv: str,
    model_names: np.ndarray = None,
    output_dir: str = "figures",
):
    """Create the 3-panel motivating figure.

    Layout (GridSpec 2x3):
        gs[0, 0] = (a) Sensitive MDS scatter
        gs[1, 0] = Orthogonal MDS scatter
        gs[:, 1] = (b) Error vs m
        gs[:, 2] = (c) Singular value spectrum
    """
    set_paper_style()
    plt.rcParams.update({
        "font.size": 6,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
    })

    # Load adapter metadata for sensitive_frac, matching to actual adapters
    with open(metadata_path) as f:
        metadata = json.load(f)
    meta_by_id = {m["adapter_id"]: m for m in metadata}
    if model_names is not None:
        valid_ids = [int(n.split("_")[1]) for n in model_names]
    else:
        valid_ids = [m["adapter_id"] for m in metadata[:len(labels)]]
    sensitive_fracs = np.array([meta_by_id[i]["sensitive_frac"] for i in valid_ids])

    # --- Layout ---
    fig = plt.figure(figsize=(5.5, 1.65))
    gs = GridSpec(2, 3, figure=fig, wspace=0.55, hspace=0.55)

    ax_a_top = fig.add_subplot(gs[0, 0])
    ax_a_bot = fig.add_subplot(gs[1, 0])
    ax_b = fig.add_subplot(gs[:, 1])
    ax_c = fig.add_subplot(gs[:, 2])

    # --- Orange gradient colormap for class-1 adapters ---
    light_orange = (1.0, 0.85, 0.6)
    orange_cmap = LinearSegmentedColormap.from_list(
        "orange_grad", [light_orange, PALETTE[1]]
    )

    # --- Panel (a): MDS scatter plots (stacked), m=50 queries ---
    rng = np.random.RandomState(0)
    m_mds = 5
    sens_sub = rng.choice(sensitive_indices, size=m_mds, replace=False)
    orth_sub = rng.choice(orthogonal_indices, size=m_mds, replace=False)

    D_sens = pairwise_energy_distances_t0(responses, sens_sub)
    X_sens = ClassicalMDS(n_components=2).fit_transform(D_sens)

    D_orth = pairwise_energy_distances_t0(responses, orth_sub)
    X_orth = ClassicalMDS(n_components=2).fit_transform(D_orth)

    class0_mask = labels == 0
    class1_mask = labels == 1

    fracs_1 = sensitive_fracs[class1_mask]
    frac_norm = (fracs_1 - fracs_1.min()) / (fracs_1.max() - fracs_1.min() + 1e-12)
    colors_1 = orange_cmap(frac_norm)

    for ax, X, is_top, title in [
        (ax_a_top, X_sens, True, f'(a) Signal queries, $m={m_mds}$'),
        (ax_a_bot, X_orth, False, f'"Orthogonal" queries, $m={m_mds}$'),
    ]:
        ax.scatter(
            X[class0_mask, 0], X[class0_mask, 1],
            c=[PALETTE[0]], marker="o", s=8, alpha=0.7, zorder=2,
        )
        ax.scatter(
            X[class1_mask, 0], X[class1_mask, 1],
            c=colors_1, marker="s", s=8, alpha=0.7, zorder=2,
        )
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(title)
        ax.set_ylabel("MDS 2")

        if is_top:
            legend_elements = [
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=PALETTE[0], markersize=4, label="Class 0"),
                Line2D([0], [0], marker="s", color="w",
                       markerfacecolor=PALETTE[1], markersize=4, label="Class 1"),
            ]
            ax.legend(handles=legend_elements, loc="best", fontsize=4)
        else:
            ax.set_xlabel("MDS 1")

    # --- Panel (b): Mean error vs m, colored by n ---
    df = pd.read_csv(classification_csv)

    n_values_plot = [10, 20, 80]
    n_colors = {n: PALETTE[i] for i, n in enumerate(n_values_plot)}
    dist_styles = {"relevant": "-", "orthogonal": "--"}
    dist_labels = {"relevant": "Signal", "orthogonal": '"Orthogonal"'}

    for n in n_values_plot:
        sub_n = df[df["n"] == n]
        for dist_name, ls in dist_styles.items():
            sub = sub_n[sub_n["distribution"] == dist_name].sort_values("m")
            if sub.empty:
                continue
            mean_err = 1.0 - sub["mean_accuracy"]
            ax_b.plot(sub["m"], mean_err, marker="o", markersize=2,
                      color=n_colors[n], linestyle=ls, linewidth=0.8)

    ax_b.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, linewidth=0.5)
    ax_b.set_xscale("log")
    ax_b.set_ylim(0.1, 0.55)
    ax_b.set_xlabel("Number of queries $m$")
    ax_b.set_ylabel("Mean error")
    ax_b.set_title("(b) Error vs $m$")

    leg_n = [Line2D([0], [0], color=n_colors[n], lw=1.0, label=f"$n={n}$")
             for n in n_values_plot]
    leg_dist = [Line2D([0], [0], color="0.4", linestyle=ls, lw=1.0,
                        label=dist_labels[name])
                for name, ls in dist_styles.items()]
    ax_b.legend(handles=leg_n + leg_dist, loc="upper right", ncol=2, fontsize=4)

    # --- Panel (c): Singular value spectrum of E (per-query energy tensor) ---
    query_sets = [
        (sensitive_indices, "Signal", PALETTE[1], "-"),
        (orthogonal_indices, '"Orthogonal"', PALETTE[2], "--"),
    ]

    n_show = 50
    r_hat_signal = None
    for q_idx, label, color, ls in query_sets:
        E, _ = per_query_energy_tensor(responses[:, q_idx, :])
        r_hat, _, sv = estimate_discriminative_rank(E, n_elbows=1)
        sv = sv / sv[0]
        k = min(n_show, len(sv))
        ax_c.plot(np.arange(1, k + 1), sv[:k],
                  color=color, linestyle=ls, linewidth=1.2,
                  marker="o", markersize=2, label=label)
        if label == "Signal":
            r_hat_signal = r_hat

    # Annotate r̂ on the spectrum
    if r_hat_signal is not None:
        ax_c.axvline(x=r_hat_signal, color="0.4", linestyle=":", linewidth=0.8,
                     alpha=0.7)
        ax_c.text(r_hat_signal + 0.5, 0.85, f"$\\hat{{r}}={r_hat_signal}$",
                  fontsize=5, color="0.3")

    ax_c.set_xlabel("Component $r$")
    ax_c.set_ylabel("$\\sigma_r / \\sigma_1$")
    ax_c.set_title("(c) Singular values of $E$")
    ax_c.legend(loc="upper right", fontsize=4)

    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/motivating_figure.pdf")
    plt.close(fig)


def plot_figure3_estimation(
    responses: np.ndarray,
    labels: np.ndarray,
    signal_indices: np.ndarray,
    orthogonal_indices: np.ndarray,
    fail_csv_path: str = None,
    output_dir: str = "figures",
):
    """Figure 3: Estimation validation on the motivating example.

    Three panels:
      (a) Scree plot of E (all queries), vertical line at r̂
      (b) GMM on B_q (per-query between-class excess)
      (c) Failure probability P[err >= 0.5] with theoretical overlay

    Uses the motivating example data (100 LoRA adapters, 200 queries)
    so the reader validates estimation on data already introduced.
    """
    from scipy.optimize import curve_fit

    set_paper_style()

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

    # --- Figure layout ---
    fig = plt.figure(figsize=(5.5, 1.65))
    gs = GridSpec(1, 3, figure=fig,
                  left=0.08, right=0.97, bottom=0.22, top=0.82, wspace=0.45)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    # --- Panel (a): Scree plot ---
    sv_norm = s / s[0]
    n_show = min(50, len(sv_norm))
    ax_a.plot(np.arange(1, n_show + 1), sv_norm[:n_show],
              color=PALETTE[0], linewidth=1.2, marker="o", markersize=2)

    ax_a.axvline(x=r_hat, color="0.4", linestyle="--", linewidth=1.2, alpha=0.8)
    ax_a.text(r_hat + 2.5, 0.85, f"$\\hat{{r}}={r_hat}$",
              fontsize=5, color="0.3")

    ax_a.set_xlabel("Component $r$")
    ax_a.set_ylabel("$\\sigma_r / \\sigma_1$")
    ax_a.set_title("(a) Singular values of $E$")

    # --- Panel (b): GMM on B_q ---
    gmm2 = gmm_info['gmm']
    gmm1 = gmm_info['gmm1']
    bic1 = gmm_info['bic1']
    bic2 = gmm_info['bic2']

    B_signal = B_q[:n_signal]
    B_orth = B_q[n_signal:]

    bins = np.linspace(B_q.min(), B_q.max(), 30)
    ax_b.hist(B_signal, bins=bins, alpha=0.6, color=PALETTE[1],
              label="Signal", density=True, edgecolor="none")
    ax_b.hist(B_orth, bins=bins, alpha=0.6, color=PALETTE[2],
              label='"Orthogonal"', density=True, edgecolor="none")

    # Density curves
    x_plot = np.linspace(B_q.min() - 0.005, B_q.max() + 0.005, 300)
    x_col = x_plot.reshape(-1, 1)

    # K=1
    m1_mean = gmm1.means_[0, 0]
    m1_std = np.sqrt(gmm1.covariances_[0, 0, 0])
    ax_b.plot(x_plot, scipy_norm.pdf(x_plot, m1_mean, m1_std),
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
        ax_b.fill_between(x_plot, comp_density, alpha=0.2,
                          color=comp_colors[cidx])
        ax_b.plot(x_plot, comp_density, color=comp_colors[cidx], linewidth=0.8)

    ax_b.set_xlabel("$B_q$ (between-class excess)")
    ax_b.set_ylabel("Density")
    ax_b.set_title("(b) GMM on $B_q$")

    # Main legend (upper right)
    leg_main = [Line2D([0], [0], color=PALETTE[1], lw=4, alpha=0.6, label="Signal"),
                Line2D([0], [0], color=PALETTE[2], lw=4, alpha=0.6, label='"Orthogonal"'),
                Line2D([0], [0], color="0.3", linestyle="--", lw=0.8,
                       label=f"$K\\!=\\!1$ (BIC={bic1:.0f})"),
                Line2D([0], [0], color=PALETTE[2], linestyle="-", lw=0.8,
                       label=f"$K\\!=\\!2$ Near-zero"),
                Line2D([0], [0], color=PALETTE[1], linestyle="-", lw=0.8,
                       label=f"$K\\!=\\!2$ Active (BIC={bic2:.0f})")]
    leg1 = ax_b.legend(handles=leg_main, loc="upper right", fontsize=4)
    ax_b.add_artist(leg1)

    # ρ̂ legend (center right)
    leg_rho = [Line2D([], [], linestyle="none",
                      label=f"$\\hat{{\\rho}} = {rho_hat:.2f}$")]
    ax_b.legend(handles=leg_rho, loc="center right", fontsize=4,
                handlelength=0, handletextpad=0)

    # --- Panel (c): Failure probability ---
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
                ax_c.plot(sub["m"], sub["failure_prob"],
                          marker="o", markersize=2, color=n_colors[n_val],
                          linestyle="-", linewidth=0.8)

        m_max = df_fail["m"].max()
        m_cont = np.linspace(1, m_max, 200)

        # Theoretical bound: r̂ · ρ̂^m
        if rho_hat > 0:
            bound = np.minimum(1.0, r_hat * rho_hat ** m_cont)
            ax_c.plot(m_cont, bound, color="0.3", linestyle=":",
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
                ax_c.plot(m_cont, y_fit, color=n_colors[n_val],
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
        ax_c.legend(handles=leg, loc="upper right", fontsize=4)

        # Annotate fit expressions
        annot_cfg = {80: {"m": 6, "va": "top", "offset": -0.04},
                     10: {"m": 6, "va": "bottom", "offset": 0.04}}
        for n_val in [80, 10]:
            if n_val not in fit_results:
                continue
            a_f, rho_f, gamma_f = fit_results[n_val]
            txt = f"${a_f:.2f}\\!\\cdot\\!{rho_f:.2f}^m\\!+\\!{gamma_f:.2f}$"
            cfg = annot_cfg[n_val]
            y_annot = _bound_model(cfg["m"], a_f, rho_f, gamma_f)
            ax_c.text(cfg["m"], y_annot + cfg["offset"], txt,
                      fontsize=4, color=n_colors[n_val],
                      ha="center", va=cfg["va"],
                      bbox=dict(boxstyle="round,pad=0.15",
                                facecolor="white", edgecolor="none",
                                alpha=0.7))

        ax_c.set_xscale("log")
        ax_c.set_ylim(-0.02, 1.05)
        ax_c.set_xlabel("Number of queries $m$")
        ax_c.set_ylabel(r"$\mathbb{P}[\mathrm{err} \geq 0.5]$")
        ax_c.set_title("(c) Failure probability")

    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/figure3_estimation.pdf")
    fig.savefig(f"{output_dir}/figure3_estimation.png", dpi=200)
    plt.close(fig)
    print(f"Saved Figure 3 (estimation) to {output_dir}/figure3_estimation.pdf")
