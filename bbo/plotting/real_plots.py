"""Plotting functions for real LLM experiments (Exp 6-10)."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from bbo.plotting.style import set_paper_style, PALETTE


def plot_exp6_scree(singular_values: np.ndarray, output_dir: str = "results/figures"):
    """Plot Exp 6: scree plot of singular value spectrum."""
    set_paper_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Raw singular values
    ax1.plot(range(1, len(singular_values) + 1), singular_values,
             marker="o", markersize=3, color=PALETTE[0])
    ax1.set_xlabel("Component index")
    ax1.set_ylabel("Singular value")
    ax1.set_title("Singular value spectrum")

    # Cumulative explained variance
    sv_sq = singular_values ** 2
    cumvar = np.cumsum(sv_sq) / sv_sq.sum()
    ax2.plot(range(1, len(cumvar) + 1), cumvar,
             marker="o", markersize=3, color=PALETTE[1])
    ax2.axhline(y=0.9, color="gray", linestyle=":", alpha=0.5)
    ax2.axhline(y=0.95, color="gray", linestyle=":", alpha=0.5)
    ax2.set_xlabel("Number of components")
    ax2.set_ylabel("Cumulative explained variance")
    ax2.set_title("Effective rank")

    # Mark 90% and 95% thresholds
    r90 = np.searchsorted(cumvar, 0.9) + 1
    r95 = np.searchsorted(cumvar, 0.95) + 1
    ax2.annotate(f"90%: r={r90}", xy=(r90, 0.9), fontsize=10)
    ax2.annotate(f"95%: r={r95}", xy=(r95, 0.95), fontsize=10)

    fig.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/exp6_scree_plot.pdf")
    plt.close(fig)


def plot_exp7(df: pd.DataFrame, output_dir: str = "results/figures"):
    """Plot Exp 7: mean accuracy +/- std vs m."""
    set_paper_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.errorbar(df["m"], df["mean_accuracy"], yerr=df["std_accuracy"],
                marker="o", capsize=3, color=PALETTE[0])
    ax.set_xlabel("Number of queries $m$")
    ax.set_ylabel("Classification accuracy")
    ax.set_title("Accuracy vs number of queries (real data)")
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylim(0.4, 1.05)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/exp7_accuracy_vs_m.pdf")
    plt.close(fig)


def plot_exp8(df: pd.DataFrame, output_dir: str = "results/figures"):
    """Plot Exp 8: accuracy vs m for relevant/orthogonal/uniform."""
    set_paper_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    dist_labels = {"uniform": "Uniform", "relevant": "Relevant",
                   "orthogonal": "Orthogonal"}

    for i, (dist_name, label) in enumerate(dist_labels.items()):
        sub = df[df["distribution"] == dist_name]
        ax.plot(sub["m"], sub["mean_accuracy"],
                marker="o", markersize=3, color=PALETTE[i], label=label)

    ax.set_xlabel("Number of queries $m$")
    ax.set_ylabel("Classification accuracy")
    ax.set_title("Relevant vs orthogonal queries (real data)")
    ax.legend()
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/exp8_relevant_vs_orthogonal.pdf")
    plt.close(fig)


def plot_exp9(df: pd.DataFrame, output_dir: str = "results/figures"):
    """Plot Exp 9: accuracy vs m for different methods (baselines)."""
    set_paper_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    methods = df["method"].unique()
    for i, method in enumerate(methods):
        sub = df[df["method"] == method]
        ax.plot(sub["m"], sub["mean_accuracy"],
                marker="o", markersize=3, color=PALETTE[i], label=method)

    ax.set_xlabel("Number of queries $m$")
    ax.set_ylabel("Classification accuracy")
    ax.set_title("MDS embedding vs baselines")
    ax.legend()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/exp9_baselines.pdf")
    plt.close(fig)
