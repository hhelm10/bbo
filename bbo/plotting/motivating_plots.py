"""Plotting functions for the motivating example figure."""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from pathlib import Path

from bbo.plotting.style import set_paper_style, PALETTE
from bbo.plotting.estimation_panels import plot_estimation_panels
from bbo.distances.energy import pairwise_energy_distances_t0, per_query_energy_tensor
from bbo.embedding.mds import ClassicalMDS
from bbo.estimation.rank_rho import compute_E_disc, estimate_discriminative_rank


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
    """Create the 3-column motivating figure.

    Layout (GridSpec 2x3):
        gs[0, 0] = (a) Signal MDS scatter
        gs[1, 0] = Orthogonal MDS scatter
        gs[:, 1] = (b) Error vs m
        gs[0, 2] = (c) Scree plot of E (all queries)
        gs[1, 2] = B_q histogram (signal vs orthogonal)
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
    fig = plt.figure(figsize=(5.5, 1.35))
    gs = GridSpec(2, 3, figure=fig, wspace=0.55, hspace=0.65)

    ax_a_top = fig.add_subplot(gs[0, 0])
    ax_a_bot = fig.add_subplot(gs[1, 0])
    ax_b = fig.add_subplot(gs[:, 1])
    ax_c_top = fig.add_subplot(gs[0, 2])
    ax_c_bot = fig.add_subplot(gs[1, 2])

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

    # --- Panel (c): Discriminative structure ---
    # Compute Ẽ (between-class centered) from ALL queries
    all_idx = np.concatenate([sensitive_indices, orthogonal_indices])
    E_all, pairs = per_query_energy_tensor(responses[:, all_idx, :])
    E_disc, _, B_q = compute_E_disc(E_all, pairs, labels)

    # r̂ from scree of Ẽ
    r_hat, U, s = estimate_discriminative_rank(E_disc)
    n_signal = len(sensitive_indices)

    # Top: Scree plot of Ẽ
    sv_norm = s / s[0]
    n_show = min(30, len(sv_norm))
    ax_c_top.plot(np.arange(1, n_show + 1), sv_norm[:n_show],
                  color=PALETTE[0], linewidth=1.2, marker="o", markersize=1.5)
    ax_c_top.axvline(x=r_hat, color="0.4", linestyle="--", linewidth=1.2, alpha=0.8)
    ax_c_top.text(r_hat + 1.5, 0.85, f"$\\hat{{r}}={r_hat}$",
                  fontsize=5, color="0.3")
    ax_c_top.set_xticklabels([])
    ax_c_top.set_ylabel("$\\sigma_r / \\sigma_1$")
    ax_c_top.set_title("(c) Singular values of $\\tilde{E}$")

    # Bottom: B_q histogram
    B_signal = B_q[:n_signal]
    B_orth = B_q[n_signal:]
    bins = np.linspace(B_q.min(), B_q.max(), 30)
    ax_c_bot.hist(B_signal, bins=bins, alpha=0.6, color=PALETTE[1],
                  label="Signal", edgecolor="none")
    ax_c_bot.hist(B_orth, bins=bins, alpha=0.6, color=PALETTE[2],
                  label='"Orthogonal"', edgecolor="none")
    ax_c_bot.set_xlabel("$B_q$ (between-class excess)")
    ax_c_bot.set_ylabel("Count")
    ax_c_bot.set_title("(d) Per-query signal")
    ax_c_bot.legend(loc="upper right", fontsize=4)

    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/motivating_figure.pdf", bbox_inches="tight")
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

    Thin wrapper around plot_estimation_panels() that creates the figure layout.
    """
    set_paper_style()

    fig = plt.figure(figsize=(5.5, 1.41))
    gs = GridSpec(1, 3, figure=fig,
                  left=0.08, right=0.97, bottom=0.22, top=0.82, wspace=0.45)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    plot_estimation_panels(
        responses, labels, signal_indices, orthogonal_indices,
        ax_a, ax_b, ax_c, fail_csv_path=fail_csv_path,
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/figure3_estimation.pdf", bbox_inches="tight")
    fig.savefig(f"{output_dir}/figure3_estimation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 3 (estimation) to {output_dir}/figure3_estimation.pdf")
