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
        gs[0, 0] = (a) MDS scatter — median-accuracy query set, m=10
        gs[1, 0] =     MDS scatter — worst-accuracy query set, m=10
        gs[:, 1] = (b) Error vs m (avg, best, worst, concat) for n=20,80
        gs[:, 2] = (c) Error vs n (avg, best, worst, concat) for m=10,50
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
    qi_median = per_trial["query_indices_median"]
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

    # --- Panel (a): MDS scatter at m=10 —-- median and worst ---
    for ax, qi, title in [
        (ax_a_top, qi_median, "(a) Median query set, $m\\!=\\!10$"),
        (ax_a_bot, qi_worst, "Worst query set, $m\\!=\\!10$"),
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

    # --- Panel (b): Error vs m for n=20, n=80 ---
    n_vals_b = [20, 80]
    n_colors = {20: PALETTE[0], 80: PALETTE[1]}
    dist_name = "relevant"

    df_mds = df[(df["method"] == "mds") & (df["distribution"] == dist_name)]
    df_cat = df[(df["method"] == "concat") & (df["distribution"] == dist_name)]

    for n_val in n_vals_b:
        sub = df_mds[df_mds["n"] == n_val].sort_values("m")
        if sub.empty:
            continue
        c = n_colors[n_val]
        # Average (mean error)
        ax_b.plot(sub["m"], 1 - sub["mean_accuracy"], marker="o", markersize=2,
                  color=c, linestyle="-", linewidth=0.8,
                  label=f"Avg ($n\\!=\\!{n_val}$)")
        # Best (min error = max accuracy)
        ax_b.plot(sub["m"], 1 - sub["max_accuracy"], marker="^", markersize=2,
                  color=c, linestyle=":", linewidth=0.6, alpha=0.7)
        # Worst (max error = min accuracy)
        ax_b.plot(sub["m"], 1 - sub["min_accuracy"], marker="v", markersize=2,
                  color=c, linestyle=":", linewidth=0.6, alpha=0.7)

    # Concat for largest n
    n_concat = n_vals_b[-1]
    sub_c = df_cat[df_cat["n"] == n_concat].sort_values("m")
    if not sub_c.empty:
        ax_b.plot(sub_c["m"], 1 - sub_c["mean_accuracy"], marker="D", markersize=2,
                  color="0.4", linestyle="-.", linewidth=0.8)

    ax_b.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, linewidth=0.5)
    ax_b.set_xscale("log")
    ax_b.set_ylim(0.1, 0.55)
    ax_b.set_xlabel("Queries $m$")
    ax_b.set_ylabel("Mean error")
    ax_b.set_title("(b) Error vs $m$")

    # Legend
    leg_b = []
    for n_val in n_vals_b:
        leg_b.append(Line2D([0], [0], color=n_colors[n_val], lw=1.0,
                            label=f"$n={n_val}$"))
    leg_b += [
        Line2D([0], [0], color="0.4", linestyle="-", marker="o", markersize=2,
               lw=0.8, label="Avg"),
        Line2D([0], [0], color="0.4", linestyle=":", marker="^", markersize=2,
               lw=0.6, label="Best"),
        Line2D([0], [0], color="0.4", linestyle=":", marker="v", markersize=2,
               lw=0.6, label="Worst"),
        Line2D([0], [0], color="0.4", linestyle="-.", marker="D", markersize=2,
               lw=0.8, label="Concat"),
    ]
    ax_b.legend(handles=leg_b, loc="upper right", ncol=2, fontsize=4)

    # --- Panel (c): Error vs n for m=10, m=50 ---
    m_vals_c = [10, 50]
    m_colors = {10: PALETTE[0], 50: PALETTE[1]}

    all_n = sorted(df_mds["n"].unique())

    for m_val in m_vals_c:
        sub = df_mds[df_mds["m"] == m_val].sort_values("n")
        if sub.empty:
            continue
        c = m_colors[m_val]
        ax_c.plot(sub["n"], 1 - sub["mean_accuracy"], marker="o", markersize=2,
                  color=c, linestyle="-", linewidth=0.8,
                  label=f"Avg ($m\\!=\\!{m_val}$)")
        ax_c.plot(sub["n"], 1 - sub["max_accuracy"], marker="^", markersize=2,
                  color=c, linestyle=":", linewidth=0.6, alpha=0.7)
        ax_c.plot(sub["n"], 1 - sub["min_accuracy"], marker="v", markersize=2,
                  color=c, linestyle=":", linewidth=0.6, alpha=0.7)

    # Concat for each m at all n
    for m_val in m_vals_c:
        sub_cc = df_cat[df_cat["m"] == m_val].sort_values("n")
        if not sub_cc.empty:
            ax_c.plot(sub_cc["n"], 1 - sub_cc["mean_accuracy"], marker="D",
                      markersize=2, color=m_colors[m_val], linestyle="-.",
                      linewidth=0.6, alpha=0.7)

    ax_c.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, linewidth=0.5)
    ax_c.set_xscale("log")
    ax_c.set_ylim(0.1, 0.55)
    ax_c.set_xticks(all_n)
    ax_c.set_xticklabels([str(n) for n in all_n])
    ax_c.xaxis.set_minor_locator(plt.NullLocator())
    ax_c.set_xlabel("Models $n$")
    ax_c.set_ylabel("Mean error")
    ax_c.set_title("(c) Error vs $n$")

    leg_c = []
    for m_val in m_vals_c:
        leg_c.append(Line2D([0], [0], color=m_colors[m_val], lw=1.0,
                            label=f"$m={m_val}$"))
    leg_c += [
        Line2D([0], [0], color="0.4", linestyle="-", marker="o", markersize=2,
               lw=0.8, label="Avg"),
        Line2D([0], [0], color="0.4", linestyle=":", marker="^", markersize=2,
               lw=0.6, label="Best"),
        Line2D([0], [0], color="0.4", linestyle=":", marker="v", markersize=2,
               lw=0.6, label="Worst"),
        Line2D([0], [0], color="0.4", linestyle="-.", marker="D", markersize=2,
               lw=0.8, label="Concat"),
    ]
    ax_c.legend(handles=leg_c, loc="upper right", ncol=2, fontsize=4)

    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/figure1_motivating.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 1 to {output_dir}/figure1_motivating.pdf")
