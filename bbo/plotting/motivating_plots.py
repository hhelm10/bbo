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
from bbo.distances.energy import pairwise_energy_distances_t0
from bbo.embedding.mds import ClassicalMDS
from bbo.experiments.real.exp6_effective_rank import run_exp6


def plot_motivating_figure(
    responses: np.ndarray,
    labels: np.ndarray,
    sensitive_indices: np.ndarray,
    orthogonal_indices: np.ndarray,
    metadata_path: str,
    classification_csv: str,
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

    # Load adapter metadata for sensitive_frac
    with open(metadata_path) as f:
        metadata = json.load(f)
    sensitive_fracs = np.array([m["sensitive_frac"] for m in metadata])

    # --- Layout ---
    fig = plt.figure(figsize=(5.5, 2.2))
    gs = GridSpec(2, 3, figure=fig, wspace=0.5, hspace=0.45)

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
    m_mds = 50
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
        (ax_a_top, X_sens, True, "(a) Sensitive queries"),
        (ax_a_bot, X_orth, False, "Orthogonal queries"),
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
            ax.legend(handles=legend_elements, loc="best")
        else:
            ax.set_xlabel("MDS 1")

    # --- Panel (b): Mean error vs m, multiple n ---
    df = pd.read_csv(classification_csv)

    has_n_col = "n" in df.columns
    if has_n_col:
        n_values = sorted(df["n"].unique())
    else:
        n_values = [int(df["m"].max())]

    n_colors = {n: PALETTE[i] for i, n in enumerate(n_values)}
    dist_styles = {"relevant": "-", "orthogonal": "--", "uniform": ":"}

    for n in n_values:
        sub_n = df[df["n"] == n] if has_n_col else df
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
             for n in n_values]
    leg_dist = [Line2D([0], [0], color="0.4", linestyle=ls, lw=1.0,
                        label=name.capitalize())
                for name, ls in dist_styles.items()]
    ax_b.legend(handles=leg_n + leg_dist, loc="upper right", ncol=2)

    # --- Panel (c): Singular value spectrum ---
    query_sets = [
        (sensitive_indices, "Sensitive", PALETTE[1], "-"),
        (orthogonal_indices, "Orthogonal", PALETTE[2], "--"),
        (None, "All queries", PALETTE[0], ":"),
    ]

    n_show = 50
    for q_idx, label, color, ls in query_sets:
        if q_idx is not None:
            resp_sub = responses[:, q_idx, :]
        else:
            resp_sub = responses
        result = run_exp6(resp_sub)
        sv = result["singular_values"]
        # Normalize by leading SV for comparability
        sv_norm = sv / sv[0]
        k = min(n_show, len(sv_norm))
        ax_c.plot(np.arange(1, k + 1), sv_norm[:k],
                  color=color, linestyle=ls, linewidth=0.8, label=label)

    ax_c.set_yscale("log")
    ax_c.set_xlabel("Component $r$")
    ax_c.set_ylabel("$\\sigma_r / \\sigma_1$")
    ax_c.set_title("(c) Singular values of $D$")
    ax_c.legend(loc="upper right", fontsize=4)

    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/motivating_figure.pdf")
    plt.close(fig)
