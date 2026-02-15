"""Plotting functions for synthetic experiments (Exp 1-5)."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from pathlib import Path

from bbo.plotting.style import set_paper_style, PALETTE


def plot_exp1(df: pd.DataFrame, output_dir: str = "results/figures"):
    """Plot Exp 1: log P[error >= 0.5] vs m, color=r, alpha=noise_level."""
    set_paper_style()

    r_values = sorted(df["r"].unique())
    noise_levels = sorted(df["noise_level"].unique())
    alpha_map = _noise_alpha_map(noise_levels)

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, r in enumerate(r_values):
        color = PALETTE[i % len(PALETTE)]
        for nl in noise_levels:
            sub = df[(df["r"] == r) & (df["noise_level"] == nl)]
            mask = sub["prob_high_error"] > 0
            sub_pos = sub[mask]
            if len(sub_pos) > 0:
                ax.plot(sub_pos["m"], sub_pos["prob_high_error"],
                        marker="o", markersize=3, color=color,
                        alpha=alpha_map[nl], linewidth=1.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of queries $m$")
    ax.set_ylabel("$P[\\mathrm{error} \\geq 0.5]$")
    ax.set_ylim(bottom=1e-3, top=1.5)

    # Color legend for r
    r_handles = [mlines.Line2D([], [], color=PALETTE[i % len(PALETTE)],
                               label=f"$r = {r}$", linewidth=2)
                 for i, r in enumerate(r_values)]
    # Alpha legend for noise
    noise_handles = [mlines.Line2D([], [], color="gray",
                                    alpha=alpha_map[nl],
                                    label=f"$\\sigma = {nl}$", linewidth=2)
                     for nl in noise_levels]

    leg1 = ax.legend(handles=r_handles, title="Rank $r$",
                     loc="upper right", fontsize=9, title_fontsize=10)
    ax.add_artist(leg1)
    ax.legend(handles=noise_handles, title="Noise $\\sigma$",
              loc="center right", fontsize=9, title_fontsize=10)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/exp1_error_vs_m_rank.pdf")
    plt.close(fig)


def plot_exp2(df: pd.DataFrame, output_dir: str = "results/figures"):
    """Plot Exp 2: log P[error >= 0.5] vs m, color=rho, alpha=noise_level."""
    set_paper_style()

    rho_values = sorted(df["rho"].unique())
    noise_levels = sorted(df["noise_level"].unique())
    alpha_map = _noise_alpha_map(noise_levels)

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, rho in enumerate(rho_values):
        color = PALETTE[i % len(PALETTE)]
        for nl in noise_levels:
            sub = df[(df["rho"] == rho) & (df["noise_level"] == nl)]
            mask = sub["prob_high_error"] > 0
            sub_pos = sub[mask]
            if len(sub_pos) > 0:
                ax.plot(sub_pos["m"], sub_pos["prob_high_error"],
                        marker="o", markersize=3, color=color,
                        alpha=alpha_map[nl], linewidth=1.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of queries $m$")
    ax.set_ylabel("$P[\\mathrm{error} \\geq 0.5]$")
    ax.set_ylim(bottom=1e-3, top=1.5)

    # Color legend for rho
    rho_handles = [mlines.Line2D([], [], color=PALETTE[i % len(PALETTE)],
                                  label=f"$\\rho = {rho:.1f}$", linewidth=2)
                   for i, rho in enumerate(rho_values)]
    noise_handles = [mlines.Line2D([], [], color="gray",
                                    alpha=alpha_map[nl],
                                    label=f"$\\sigma = {nl}$", linewidth=2)
                     for nl in noise_levels]

    leg1 = ax.legend(handles=rho_handles, title="$\\rho$",
                     loc="upper right", fontsize=9, title_fontsize=10)
    ax.add_artist(leg1)
    ax.legend(handles=noise_handles, title="Noise $\\sigma$",
              loc="center right", fontsize=9, title_fontsize=10)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/exp2_error_vs_m_rho.pdf")
    plt.close(fig)


def plot_figure1(df_exp1: pd.DataFrame, df_exp2: pd.DataFrame,
                 output_dir: str = "results/figures"):
    """Combined Figure 1: two panels for Exp 1 and Exp 2."""
    set_paper_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    # --- Panel A: vary r ---
    r_values = sorted(df_exp1["r"].unique())
    noise_levels = sorted(df_exp1["noise_level"].unique())
    alpha_map = _noise_alpha_map(noise_levels)

    for i, r in enumerate(r_values):
        color = PALETTE[i % len(PALETTE)]
        for nl in noise_levels:
            sub = df_exp1[(df_exp1["r"] == r) & (df_exp1["noise_level"] == nl)]
            mask = sub["prob_high_error"] > 0
            sub_pos = sub[mask]
            if len(sub_pos) > 0:
                ax1.plot(sub_pos["m"], sub_pos["prob_high_error"],
                         marker="o", markersize=3, color=color,
                         alpha=alpha_map[nl], linewidth=1.5)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Number of queries $m$")
    ax1.set_ylabel("$P[\\mathrm{error} \\geq 0.5]$")
    ax1.set_ylim(bottom=1e-3, top=1.5)
    ax1.set_title("(a) Varying rank $r$", fontsize=12)

    r_handles = [mlines.Line2D([], [], color=PALETTE[i % len(PALETTE)],
                               label=f"$r = {r}$", linewidth=2)
                 for i, r in enumerate(r_values)]
    ax1.legend(handles=r_handles, fontsize=8, loc="upper right")

    # --- Panel B: vary rho ---
    rho_values = sorted(df_exp2["rho"].unique())
    noise_levels_2 = sorted(df_exp2["noise_level"].unique())
    alpha_map_2 = _noise_alpha_map(noise_levels_2)

    for i, rho in enumerate(rho_values):
        color = PALETTE[i % len(PALETTE)]
        for nl in noise_levels_2:
            sub = df_exp2[(df_exp2["rho"] == rho) & (df_exp2["noise_level"] == nl)]
            mask = sub["prob_high_error"] > 0
            sub_pos = sub[mask]
            if len(sub_pos) > 0:
                ax2.plot(sub_pos["m"], sub_pos["prob_high_error"],
                         marker="o", markersize=3, color=color,
                         alpha=alpha_map_2[nl], linewidth=1.5)

    ax2.set_xscale("log")
    ax2.set_xlabel("Number of queries $m$")
    ax2.set_title("(b) Varying $\\rho$", fontsize=12)

    rho_handles = [mlines.Line2D([], [], color=PALETTE[i % len(PALETTE)],
                                  label=f"$\\rho = {rho:.1f}$", linewidth=2)
                   for i, rho in enumerate(rho_values)]
    ax2.legend(handles=rho_handles, fontsize=8, loc="upper right")

    # Shared noise legend below
    noise_handles = [mlines.Line2D([], [], color="gray",
                                    alpha=alpha_map[nl],
                                    label=f"$\\sigma = {nl}$", linewidth=2)
                     for nl in noise_levels]
    fig.legend(handles=noise_handles, title="Noise $\\sigma$",
               loc="lower center", ncol=len(noise_levels),
               fontsize=9, title_fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/figure1_error_vs_m.pdf")
    plt.close(fig)


def plot_exp3(df: pd.DataFrame, output_dir: str = "results/figures"):
    """Plot Exp 3: accuracy vs m for three query distributions."""
    set_paper_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    dist_labels = {"uniform": "Uniform", "signal": "Signal-concentrated",
                   "orthogonal": "Orthogonal-concentrated"}

    for i, (dist_name, label) in enumerate(dist_labels.items()):
        sub = df[df["distribution"] == dist_name]
        ax.plot(sub["m"], sub["accuracy"],
                marker="o", markersize=4, color=PALETTE[i],
                label=label, linewidth=2)

    ax.set_xlabel("Number of queries $m$")
    ax.set_ylabel("Classification accuracy")
    ax.legend(fontsize=9)
    ax.set_ylim(0.4, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/exp3_query_distribution.pdf")
    plt.close(fig)


def plot_exp4(df: pd.DataFrame, output_dir: str = "results/figures"):
    """Plot Exp 4: classification error vs n."""
    set_paper_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(df["n_models"], df["mean_error"],
            marker="o", color=PALETTE[0])
    ax.set_xlabel("Number of training models $n$")
    ax.set_ylabel("Mean classification error")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/exp4_error_vs_n.pdf")
    plt.close(fig)


def plot_exp5(df: pd.DataFrame, output_dir: str = "results/figures"):
    """Plot Exp 5: inf_h L vs n with horizontal line at L*."""
    set_paper_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(df["n_models"], df["mean_error"],
            marker="o", color=PALETTE[0], label="Observed error")

    if "bayes_error" in df.columns:
        bayes = df["bayes_error"].iloc[0]
        ax.axhline(y=bayes, color="red", linestyle="--",
                   label=f"$L^*$ = {bayes:.2f}")

    ax.set_xlabel("Number of training models $n$")
    ax.set_ylabel("Classification error")
    ax.legend()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/exp5_bayes_convergence.pdf")
    plt.close(fig)


def _noise_alpha_map(noise_levels):
    """Map noise levels to alpha values (higher noise = more transparent)."""
    n = len(noise_levels)
    # Range from 1.0 (lowest noise) to 0.3 (highest noise)
    alphas = np.linspace(1.0, 0.3, n)
    return dict(zip(noise_levels, alphas))
