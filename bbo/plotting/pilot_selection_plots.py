"""Figures for the pilot query selection experiment.

- plot_pilot_selection: 1×3 figure (one panel per dataset)
- plot_embed_robustness_3x3: 3×3 figure (rows=embed models, cols=datasets)
- plot_threshold_sensitivity: 1×3 figure (error vs loading percentile threshold)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

from bbo.plotting.style import set_paper_style, PALETTE


# Shared selector styling
SEL_CONFIG = [
    ("uniform",        PALETTE[2], "-", "s", "Uniform"),
    ("uniform_signal", PALETTE[0], "-", "o", "Est. signal"),
    ("stepwise",       PALETTE[3], "-", "^", "Stepwise"),
]

DATASET_TITLES = ["Motivating", "System Prompt", "RAG"]


def plot_pilot_selection(
    motivating_csv: str,
    system_prompt_csv: str,
    rag_csv: str,
    output_path: str = "figures/figure_query_selection_methods.pdf",
):
    """1×3 figure: pilot query selection results."""
    set_paper_style()

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.6), sharey=True)

    datasets = list(zip(DATASET_TITLES,
                        [motivating_csv, system_prompt_csv, rag_csv]))

    n_plot = 80

    for ax, (title, csv_path) in zip(axes, datasets):
        df = pd.read_csv(csv_path)

        for sel_name, color, ls, marker, label in SEL_CONFIG:
            sub = df[(df["selector"] == sel_name) &
                     (df["n_train"] == n_plot)].sort_values("m")
            if sub.empty:
                continue
            ax.plot(sub["m"], sub["mean_error"],
                    marker=marker, markersize=2, color=color,
                    linestyle=ls, linewidth=0.8)

        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3, linewidth=0.5)
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 0.55)
        ax.set_xlabel("Queries $m$")
        ax.set_title(title)

    axes[0].set_ylabel("Mean error")

    # Shared legend
    leg = []
    for sel_name, color, ls, marker, label in SEL_CONFIG:
        leg.append(Line2D([0], [0], color=color, linestyle=ls, lw=0.8,
                          marker=marker, markersize=2, label=label))
    axes[1].legend(handles=leg, loc="upper right", fontsize=6, ncol=1)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved pilot selection figure to {output_path}")


def plot_embed_robustness_3x3(
    csv_paths: dict,
    output_path: str = "figures/figure_embed_robustness.pdf",
):
    """3×3 figure: embedding model robustness.

    Parameters
    ----------
    csv_paths : dict
        {(dataset_name, embed_model): csv_path} for all 9 combos.
        dataset_name in {"motivating", "system_prompt", "rag"}
        embed_model in {"nomic-embed-text-v1.5", "all-MiniLM-L6-v2", "text-embedding-3-small"}
    """
    set_paper_style()

    embed_models = [
        ("nomic-embed-text-v1.5", "nomic-embed"),
        ("all-MiniLM-L6-v2", "all-MiniLM"),
        ("text-embedding-3-small", "text-embed-3-s"),
    ]
    dataset_keys = ["motivating", "system_prompt", "rag"]

    sel_config = [
        ("uniform",        PALETTE[2], "-", "s", "Uniform"),
        ("uniform_signal", PALETTE[0], "-", "o", "Est. signal"),
    ]

    n_plot = 80
    n_rows = len(embed_models)
    n_cols = len(dataset_keys)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5, 4.0), sharey=True)

    for row, (em_key, em_label) in enumerate(embed_models):
        for col, (ds_key, ds_title) in enumerate(zip(dataset_keys, DATASET_TITLES)):
            ax = axes[row, col]
            csv_path = csv_paths.get((ds_key, em_key))
            if csv_path and Path(csv_path).exists():
                df = pd.read_csv(csv_path)

                for sel_name, color, ls, marker, label in sel_config:
                    sub = df[(df["selector"] == sel_name) &
                             (df["n_train"] == n_plot)].sort_values("m")
                    if sub.empty:
                        continue
                    ax.plot(sub["m"], sub["mean_error"],
                            marker=marker, markersize=2, color=color,
                            linestyle=ls, linewidth=0.8)

            ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3, linewidth=0.5)
            ax.set_xscale("log")
            ax.set_ylim(-0.02, 0.55)

            if row == n_rows - 1:
                ax.set_xlabel("Queries $m$")
            if row == 0:
                ax.set_title(ds_title)
            if col == 0:
                ax.set_ylabel(em_label)

    # Shared legend
    leg = [Line2D([0], [0], color=c, linestyle=ls, lw=0.8,
                  marker=mk, markersize=2, label=lbl)
           for _, c, ls, mk, lbl in sel_config]
    axes[0, 1].legend(handles=leg, loc="upper right", fontsize=6, ncol=1)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved embed robustness figure to {output_path}")


def plot_threshold_sensitivity(
    motivating_csv: str,
    system_prompt_csv: str,
    rag_csv: str,
    output_path: str = "figures/figure_threshold_sensitivity.pdf",
):
    """1×3 figure: classification error vs loading percentile threshold.

    Each panel shows error at m=5 as a function of the percentile threshold
    used to define the signal set, plus a horizontal uniform reference line.
    """
    set_paper_style()

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.6), sharey=True)

    datasets = list(zip(DATASET_TITLES,
                        [motivating_csv, system_prompt_csv, rag_csv]))

    for ax, (title, csv_path) in zip(axes, datasets):
        df = pd.read_csv(csv_path)

        # Threshold curve
        thresh = df[df["selector"] == "threshold"].sort_values("threshold")
        ax.plot(thresh["threshold"], thresh["mean_error"],
                marker="o", markersize=2, color=PALETTE[0],
                linestyle="-", linewidth=0.8, label="Threshold sampling")

        # Uniform reference line
        unif = df[df["selector"] == "uniform"]
        if not unif.empty:
            ax.axhline(y=unif["mean_error"].values[0], color=PALETTE[2],
                       linestyle="--", linewidth=0.8, label="Uniform")

        # GMM reference point
        gmm = df[df["selector"] == "gmm_signal"]
        if not gmm.empty:
            ax.axhline(y=gmm["mean_error"].values[0], color=PALETTE[3],
                       linestyle=":", linewidth=0.8, label="GMM signal")

        ax.set_xlim(5, 95)
        ax.set_ylim(-0.02, 0.55)
        ax.set_xlabel("Percentile threshold $\\tau$")
        ax.set_title(title)

    axes[0].set_ylabel("Mean error ($m{=}5$)")

    # Shared legend
    leg = [
        Line2D([0], [0], color=PALETTE[0], linestyle="-", lw=0.8,
               marker="o", markersize=2, label="Threshold"),
        Line2D([0], [0], color=PALETTE[2], linestyle="--", lw=0.8,
               label="Uniform"),
        Line2D([0], [0], color=PALETTE[3], linestyle=":", lw=0.8,
               label="GMM signal"),
    ]
    axes[1].legend(handles=leg, loc="upper left", fontsize=6, ncol=1)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved threshold sensitivity figure to {output_path}")
