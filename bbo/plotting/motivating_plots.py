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


def plot_figure1_motivating(
    responses: np.ndarray,
    labels: np.ndarray,
    sensitive_indices: np.ndarray,
    orthogonal_indices: np.ndarray,
    metadata_path: str,
    classification_csv: str,
    model_names: np.ndarray = None,
    output_dir: str = "figures",
):
    """Figure 1: Motivating example (3 columns).

    Layout (GridSpec 2x3):
        gs[0, 0] = (a) MDS scatter — signal queries, m=5
        gs[1, 0] =     MDS scatter — orthogonal queries, m=5
        gs[:, 1] = (b) Error vs m for n=80 (signal, orthogonal, uniform)
        gs[:, 2] = (c) Error vs n for m=10 (signal, orthogonal, uniform)
    """
    set_paper_style()
    plt.rcParams.update({
        "font.size": 6,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
    })

    import json
    from matplotlib.colors import LinearSegmentedColormap

    # Load metadata for class-1 coloring
    with open(metadata_path) as f:
        metadata = json.load(f)
    meta_by_id = {m["adapter_id"]: m for m in metadata}
    if model_names is not None:
        valid_ids = [int(n.split("_")[1]) for n in model_names]
    else:
        valid_ids = [m["adapter_id"] for m in metadata[:len(labels)]]
    sensitive_fracs = np.array([meta_by_id[i]["sensitive_frac"] for i in valid_ids])

    # Load classification CSV
    df = pd.read_csv(classification_csv)

    # --- Layout ---
    fig = plt.figure(figsize=(5.5, 1.6))
    gs = GridSpec(2, 3, figure=fig, wspace=0.55, hspace=0.65)

    ax_a_top = fig.add_subplot(gs[0, 0])
    ax_a_bot = fig.add_subplot(gs[1, 0])
    ax_b = fig.add_subplot(gs[:, 1])
    ax_c = fig.add_subplot(gs[:, 2])

    # --- Orange gradient colormap ---
    light_orange = (1.0, 0.85, 0.6)
    orange_cmap = LinearSegmentedColormap.from_list(
        "orange_grad", [light_orange, PALETTE[1]]
    )

    class0_mask = labels == 0
    class1_mask = labels == 1
    fracs_1 = sensitive_fracs[class1_mask]
    frac_norm = (fracs_1 - fracs_1.min()) / (fracs_1.max() - fracs_1.min() + 1e-12)
    colors_1 = orange_cmap(frac_norm)

    # --- Panel (a): MDS scatter at m=5 — signal vs orthogonal ---
    rng = np.random.RandomState(0)
    m_mds = 5
    sig_sub = rng.choice(sensitive_indices, size=m_mds, replace=False)
    orth_sub = rng.choice(orthogonal_indices, size=m_mds, replace=False)

    for ax, qi, title in [
        (ax_a_top, sig_sub, "(a) Signal queries, $m\\!=\\!5$"),
        (ax_a_bot, orth_sub, "Orthogonal queries, $m\\!=\\!5$"),
    ]:
        D = pairwise_energy_distances_t0(responses, qi)
        X = ClassicalMDS(n_components=2).fit_transform(D)
        ax.scatter(X[class0_mask, 0], X[class0_mask, 1],
                   c=[PALETTE[0]], marker="o", s=8, alpha=0.7, zorder=2)
        ax.scatter(X[class1_mask, 0], X[class1_mask, 1],
                   c=colors_1, marker="s", s=8, alpha=0.7, zorder=2)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(title)
        ax.set_ylabel("MDS 2")

    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=PALETTE[0], markersize=4, label="Class 0"),
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=PALETTE[1], markersize=4, label="Class 1"),
    ]
    ax_a_top.legend(handles=legend_elements, loc="best", fontsize=4)
    ax_a_bot.set_xlabel("MDS 1")

    # --- Panels (b) and (c): Signal vs Orthogonal vs Uniform ---
    df_mds = df[df["method"] == "mds"]

    dist_config = [
        ("relevant",   "-",  PALETTE[0], "Signal"),
        ("orthogonal", "--", PALETTE[1], "Orthogonal"),
        ("uniform",    ":",  PALETTE[2], "Uniform"),
    ]

    # Panel (b): Error vs m for n=80
    n_plot = 80
    for dist_name, ls, color, label in dist_config:
        sub = df_mds[(df_mds["distribution"] == dist_name) &
                     (df_mds["n"] == n_plot)].sort_values("m")
        if not sub.empty:
            ax_b.plot(sub["m"], 1 - sub["mean_accuracy"],
                      marker="o", markersize=2, color=color,
                      linestyle=ls, linewidth=0.8, label=label)

    ax_b.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3, linewidth=0.5)
    ax_b.set_xscale("log")
    ax_b.set_ylim(-0.02, 0.55)
    ax_b.set_xlabel("Queries $m$")
    ax_b.set_ylabel("Mean error")
    ax_b.set_title(f"(b) Error vs $m$ ($n\\!=\\!{n_plot}$)")
    ax_b.legend(loc="upper right", fontsize=4)

    # Panel (c): Error vs n for m=10
    m_plot = 10
    all_n = sorted(df_mds["n"].unique())

    for dist_name, ls, color, label in dist_config:
        sub = df_mds[(df_mds["distribution"] == dist_name) &
                     (df_mds["m"] == m_plot)].sort_values("n")
        if not sub.empty:
            ax_c.plot(sub["n"], 1 - sub["mean_accuracy"],
                      marker="o", markersize=2, color=color,
                      linestyle=ls, linewidth=0.8, label=label)

    ax_c.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3, linewidth=0.5)
    ax_c.set_xscale("log")
    ax_c.set_ylim(-0.02, 0.55)
    ax_c.set_xticks(all_n)
    ax_c.set_xticklabels([str(n) for n in all_n])
    ax_c.xaxis.set_minor_locator(plt.NullLocator())
    ax_c.set_xlabel("Models $n$")
    ax_c.set_ylabel("Mean error")
    ax_c.set_title(f"(c) Error vs $n$ ($m\\!=\\!{m_plot}$)")
    ax_c.legend(loc="upper right", fontsize=4)

    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/figure1_motivating.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 1 to {output_dir}/figure1_motivating.pdf")
