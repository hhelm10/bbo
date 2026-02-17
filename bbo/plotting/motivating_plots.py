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
    output_dir: str = "results/motivating/figures",
):
    """Create the 3-panel (4 sub-panel) motivating figure.

    Parameters
    ----------
    responses : ndarray of shape (n_models, M, p)
        Embedded responses for all adapters and queries.
    labels : ndarray of shape (n_models,)
        Binary class labels (0 or 1) per adapter.
    sensitive_indices : ndarray
        Query indices for sensitive queries.
    orthogonal_indices : ndarray
        Query indices for orthogonal queries.
    metadata_path : str
        Path to adapter_metadata.json (contains sensitive_frac per adapter).
    classification_csv : str
        Path to classification_results.csv.
    output_dir : str
        Directory to save the figure.
    """
    set_paper_style()

    # Load adapter metadata for sensitive_frac
    with open(metadata_path) as f:
        metadata = json.load(f)
    sensitive_fracs = np.array([m["sensitive_frac"] for m in metadata])

    # Create figure with GridSpec
    fig = plt.figure(figsize=(5.5, 1.8))
    gs = GridSpec(1, 12, figure=fig, wspace=0.55)

    ax_a_left = fig.add_subplot(gs[0, 0:3])
    ax_a_right = fig.add_subplot(gs[0, 3:6])
    ax_b = fig.add_subplot(gs[0, 6:9])
    ax_c = fig.add_subplot(gs[0, 9:12])

    # --- Orange gradient colormap for class-1 adapters ---
    light_orange = (1.0, 0.85, 0.6)
    dark_orange = PALETTE[1]
    orange_cmap = LinearSegmentedColormap.from_list(
        "orange_grad", [light_orange, dark_orange]
    )

    # --- Panel (a): MDS scatter plots ---
    # Compute MDS embeddings
    D_sens = pairwise_energy_distances_t0(responses, sensitive_indices)
    X_sens = ClassicalMDS(n_components=2).fit_transform(D_sens)

    D_orth = pairwise_energy_distances_t0(responses, orthogonal_indices)
    X_orth = ClassicalMDS(n_components=2).fit_transform(D_orth)

    class0_mask = labels == 0
    class1_mask = labels == 1

    # Normalize sensitive_frac for class-1 adapters to [0, 1] for colormap
    fracs_1 = sensitive_fracs[class1_mask]
    frac_norm = (fracs_1 - fracs_1.min()) / (fracs_1.max() - fracs_1.min() + 1e-12)
    colors_1 = orange_cmap(frac_norm)

    for ax, X, is_left, title in [
        (ax_a_left, X_sens, True, "(a) Sensitive"),
        (ax_a_right, X_orth, False, "Orthogonal"),
    ]:
        # Class 0: blue circles
        ax.scatter(
            X[class0_mask, 0], X[class0_mask, 1],
            c=[PALETTE[0]], marker="o", s=10, alpha=0.7,
            zorder=2,
        )
        # Class 1: orange gradient by sensitive_frac, squares
        ax.scatter(
            X[class1_mask, 0], X[class1_mask, 1],
            c=colors_1, marker="s", s=10, alpha=0.7,
            zorder=2,
        )
        # Remove tick labels (MDS coords are arbitrary)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(title)

        if is_left:
            ax.set_xlabel("MDS dim 1")
            ax.set_ylabel("MDS dim 2")
            # Legend on left panel only
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

    dist_config = [
        ("relevant", "Relevant", PALETTE[1]),
        ("orthogonal", "Orthogonal", PALETTE[2]),
        ("uniform", "Uniform", PALETTE[0]),
    ]

    for dist_name, label, color in dist_config:
        sub = df[df["distribution"] == dist_name].sort_values("m")
        ax_b.plot(sub["m"], sub["mean_accuracy"], marker="o", markersize=2,
                  color=color, label=label)
        ax_b.fill_between(
            sub["m"],
            sub["mean_accuracy"] - sub["std_accuracy"],
            sub["mean_accuracy"] + sub["std_accuracy"],
            color=color, alpha=0.15,
        )

    ax_b.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, linewidth=0.5)
    ax_b.set_xscale("log")
    ax_b.set_ylim(0.5, 1.0)
    ax_b.set_xlabel("Number of queries $m$")
    ax_b.set_ylabel("Accuracy")
    ax_b.set_title("(b) Accuracy vs $m$")
    ax_b.legend(loc="lower right")

    # --- Panel (c): SVD spectrum + cumulative variance ---
    result = run_exp6(responses)
    svs = result["singular_values"]
    cumvar = result["cumulative_variance"]

    n_show = min(50, len(svs))
    svs_norm = svs[:n_show] / svs[0]  # normalize to first SV
    components = np.arange(1, n_show + 1)

    ax_c.plot(components, svs_norm, color=PALETTE[0], marker="o", markersize=1.5)
    ax_c.set_yscale("log")
    ax_c.set_xlabel("Component")
    ax_c.set_ylabel("Normalized SV", color=PALETTE[0])
    ax_c.tick_params(axis="y", labelcolor=PALETTE[0])
    ax_c.set_title("(c) SVD spectrum")

    # Right y-axis: cumulative variance
    ax_c2 = ax_c.twinx()
    ax_c2.plot(components, cumvar[:n_show], color=PALETTE[1], linewidth=0.8)
    ax_c2.set_ylabel("Cumul. var.", color=PALETTE[1])
    ax_c2.tick_params(axis="y", labelcolor=PALETTE[1])
    ax_c2.set_ylim(0, 1.05)

    # Threshold lines
    for thresh in [0.9, 0.95]:
        ax_c2.axhline(y=thresh, color="gray", linestyle="--", linewidth=0.5,
                      alpha=0.6)

    # Annotate r_90
    r90 = result["r90"]
    if r90 <= n_show:
        ax_c2.annotate(
            f"$r_{{90}}={r90}$",
            xy=(r90, 0.9), fontsize=5,
            xytext=(r90 + 5, 0.80),
            arrowprops=dict(arrowstyle="->", lw=0.5),
        )

    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/motivating_figure.pdf")
    plt.close(fig)
