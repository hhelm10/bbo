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
        gs[0, 0] = (a) Sensitive MDS scatter (top-left)
        gs[1, 0] = Orthogonal MDS scatter (bottom-left)
        gs[:, 1] = (b) Accuracy vs m (spans both rows)
        gs[:, 2] = (c) Cumulative variance / effective rank (spans both rows)

    Parameters
    ----------
    responses : ndarray of shape (n_models, M, p)
    labels : ndarray of shape (n_models,)
    sensitive_indices, orthogonal_indices : ndarray
    metadata_path : str, path to adapter_metadata.json
    classification_csv : str, path to classification_results.csv
    output_dir : str
    """
    set_paper_style()

    # Load adapter metadata for sensitive_frac
    with open(metadata_path) as f:
        metadata = json.load(f)
    sensitive_fracs = np.array([m["sensitive_frac"] for m in metadata])

    # --- Layout ---
    fig = plt.figure(figsize=(5.5, 2.8))
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

    # --- Panel (a): MDS scatter plots (stacked) ---
    D_sens = pairwise_energy_distances_t0(responses, sensitive_indices)
    X_sens = ClassicalMDS(n_components=2).fit_transform(D_sens)

    D_orth = pairwise_energy_distances_t0(responses, orthogonal_indices)
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
        ax.set_ylabel("MDS dim 2")

        if is_top:
            # Legend on top panel only
            legend_elements = [
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=PALETTE[0], markersize=4, label="Class 0"),
                Line2D([0], [0], marker="s", color="w",
                       markerfacecolor=PALETTE[1], markersize=4, label="Class 1"),
            ]
            ax.legend(handles=legend_elements, loc="best")
        else:
            ax.set_xlabel("MDS dim 1")

    # --- Panel (b): Classification accuracy vs m ---
    df = pd.read_csv(classification_csv)

    line_styles = ["-", "--", ":"]
    markers = ["o", "s", "^"]
    dist_config = [
        ("relevant", "Relevant", PALETTE[1]),
        ("orthogonal", "Orthogonal", PALETTE[2]),
        ("uniform", "Uniform", PALETTE[0]),
    ]

    for (dist_name, label, color), ls, mk in zip(dist_config, line_styles, markers):
        sub = df[df["distribution"] == dist_name].sort_values("m")
        ax_b.plot(sub["m"], sub["mean_accuracy"], marker=mk, markersize=3,
                  color=color, label=label, linestyle=ls, linewidth=1.0)
        ax_b.fill_between(
            sub["m"],
            sub["mean_accuracy"] - sub["std_accuracy"],
            sub["mean_accuracy"] + sub["std_accuracy"],
            color=color, alpha=0.10,
        )

    ax_b.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, linewidth=0.5)
    ax_b.set_xscale("log")
    ax_b.set_ylim(0.55, 0.95)
    ax_b.set_xlabel("Number of queries $m$")
    ax_b.set_ylabel("Accuracy")
    ax_b.set_title("(b) Accuracy vs $m$")
    ax_b.legend(loc="lower right")

    # --- Panel (c): Cumulative explained variance ---
    result = run_exp6(responses)
    cumvar = result["cumulative_variance"]
    r90 = result["r90"]
    r95 = result["r95"]

    n_show = min(50, len(cumvar))
    components = np.arange(1, n_show + 1)

    ax_c.plot(components, cumvar[:n_show], color=PALETTE[0], linewidth=1.0)
    ax_c.set_xlabel("Number of components $r$")
    ax_c.set_ylabel("Cumulative variance explained")
    ax_c.set_title("(c) Effective rank")
    ax_c.set_ylim(0, 1.05)

    # Threshold lines
    ax_c.axhline(y=0.9, color="gray", linestyle="--", linewidth=0.5, alpha=0.6)
    ax_c.axhline(y=0.95, color="gray", linestyle="--", linewidth=0.5, alpha=0.6)

    # Vertical drop-lines at r_90 and r_95
    ax_c.axvline(x=r90, color=PALETTE[1], linestyle=":", linewidth=0.7, alpha=0.7)
    ax_c.axvline(x=r95, color=PALETTE[3], linestyle=":", linewidth=0.7, alpha=0.7)

    ax_c.annotate(
        f"$r_{{90}}={r90}$", xy=(r90, 0.9), fontsize=5.5,
        xytext=(r90 + 3, 0.75),
        arrowprops=dict(arrowstyle="->", lw=0.5, color="0.4"),
        color=PALETTE[1],
    )
    ax_c.annotate(
        f"$r_{{95}}={r95}$", xy=(r95, 0.95), fontsize=5.5,
        xytext=(r95 + 2, 0.82),
        arrowprops=dict(arrowstyle="->", lw=0.5, color="0.4"),
        color=PALETTE[3],
    )

    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/motivating_figure.pdf")
    plt.close(fig)
