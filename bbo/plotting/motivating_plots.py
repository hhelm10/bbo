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
    per_trial_npz: str,
    model_names: np.ndarray = None,
    output_dir: str = "figures",
):
    """Figure 1: Motivating example (3 columns).

    Layout (GridSpec 2x3):
        gs[0, 0] = (a) MDS scatter — worst query set, m=10
        gs[1, 0] =     MDS scatter — best query set, m=10
        gs[:, 1] = (b) Error vs m for n=80 (mean DKPS, concat, best, worst)
        gs[:, 2] = (c) Error vs n for m=10 (same 4 methods)
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

    # Load per-trial data for MDS scatter
    per_trial = np.load(per_trial_npz, allow_pickle=True)
    qi_best = per_trial["query_indices_best"]
    qi_worst = per_trial["query_indices_worst"]

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

    # --- Panel (a): MDS scatter at m=10 — worst and best ---
    for ax, qi, title in [
        (ax_a_top, qi_worst, "(a) Worst query set, $m\\!=\\!10$"),
        (ax_a_bot, qi_best, "Best query set, $m\\!=\\!10$"),
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

    # Legend on top
    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=PALETTE[0], markersize=4, label="Class 0"),
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=PALETTE[1], markersize=4, label="Class 1"),
    ]
    ax_a_top.legend(handles=legend_elements, loc="best", fontsize=4)
    ax_a_bot.set_xlabel("MDS 1")

    # --- Panel (b): Error vs m for n=80 ---
    n_plot = 80
    dist_name = "relevant"

    df_mds = df[(df["method"] == "mds") & (df["distribution"] == dist_name)]
    df_cat = df[(df["method"] == "concat") & (df["distribution"] == dist_name)]

    sub_mds = df_mds[df_mds["n"] == n_plot].sort_values("m")
    sub_cat = df_cat[df_cat["n"] == n_plot].sort_values("m")

    if not sub_mds.empty:
        ax_b.plot(sub_mds["m"], 1 - sub_mds["mean_accuracy"],
                  marker="o", markersize=2, color=PALETTE[0],
                  linestyle="-", linewidth=0.8, label="Mean DKPS")
        ax_b.plot(sub_mds["m"], 1 - sub_mds["max_accuracy"],
                  marker="^", markersize=2, color=PALETTE[2],
                  linestyle="-", linewidth=0.8, label="Best DKPS")
        ax_b.plot(sub_mds["m"], 1 - sub_mds["min_accuracy"],
                  marker="v", markersize=2, color=PALETTE[3],
                  linestyle="-", linewidth=0.8, label="Worst DKPS")
    if not sub_cat.empty:
        ax_b.plot(sub_cat["m"], 1 - sub_cat["mean_accuracy"],
                  marker="D", markersize=2, color=PALETTE[1],
                  linestyle="-.", linewidth=0.8, label="Concat")

    ax_b.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, linewidth=0.5)
    ax_b.set_xscale("log")
    ax_b.set_ylim(0.05, 0.55)
    ax_b.set_xlabel("Queries $m$")
    ax_b.set_ylabel("Mean error")
    ax_b.set_title(f"(b) Error vs $m$ ($n\\!=\\!{n_plot}$)")
    ax_b.legend(loc="upper right", fontsize=4)

    # --- Panel (c): Error vs n for m=10 ---
    m_plot = 10
    all_n = sorted(df_mds["n"].unique())

    sub_mds_n = df_mds[df_mds["m"] == m_plot].sort_values("n")
    sub_cat_n = df_cat[df_cat["m"] == m_plot].sort_values("n")

    if not sub_mds_n.empty:
        ax_c.plot(sub_mds_n["n"], 1 - sub_mds_n["mean_accuracy"],
                  marker="o", markersize=2, color=PALETTE[0],
                  linestyle="-", linewidth=0.8, label="Mean DKPS")
        ax_c.plot(sub_mds_n["n"], 1 - sub_mds_n["max_accuracy"],
                  marker="^", markersize=2, color=PALETTE[2],
                  linestyle="-", linewidth=0.8, label="Best DKPS")
        ax_c.plot(sub_mds_n["n"], 1 - sub_mds_n["min_accuracy"],
                  marker="v", markersize=2, color=PALETTE[3],
                  linestyle="-", linewidth=0.8, label="Worst DKPS")
    if not sub_cat_n.empty:
        ax_c.plot(sub_cat_n["n"], 1 - sub_cat_n["mean_accuracy"],
                  marker="D", markersize=2, color=PALETTE[1],
                  linestyle="-.", linewidth=0.8, label="Concat")

    ax_c.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, linewidth=0.5)
    ax_c.set_xscale("log")
    ax_c.set_ylim(0.05, 0.55)
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
