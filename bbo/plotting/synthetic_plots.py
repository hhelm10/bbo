"""Plotting functions for synthetic experiments (Exp 1-5)."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from pathlib import Path

from bbo.plotting.style import set_paper_style, PALETTE


def plot_exp1(df: pd.DataFrame, output_dir: str = "results/figures"):
    """Plot Exp 1: log P[error >= 0.5] vs m, one curve per r."""
    set_paper_style()

    r_values = sorted(df["r"].unique())

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, r in enumerate(r_values):
        color = PALETTE[i % len(PALETTE)]
        sub = df[df["r"] == r]
        mask = sub["prob_high_error"] > 0
        sub_pos = sub[mask]
        if len(sub_pos) > 0:
            ax.plot(sub_pos["m"], sub_pos["prob_high_error"],
                    marker="o", markersize=4, color=color,
                    label=f"$r = {r}$", linewidth=1.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of queries $m$")
    ax.set_ylabel("$P[\\mathrm{error} \\geq 0.5]$")
    ax.set_ylim(bottom=1e-3, top=1.5)
    ax.legend(title="Rank $r$", fontsize=9, title_fontsize=10)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/exp1_error_vs_m_rank.pdf")
    plt.close(fig)


def plot_exp2(df: pd.DataFrame, output_dir: str = "results/figures"):
    """Plot Exp 2: log P[error >= 0.5] vs m, one curve per rho."""
    set_paper_style()

    rho_values = sorted(df["rho"].unique())

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, rho in enumerate(rho_values):
        color = PALETTE[i % len(PALETTE)]
        sub = df[df["rho"] == rho]
        mask = sub["prob_high_error"] > 0
        sub_pos = sub[mask]
        if len(sub_pos) > 0:
            ax.plot(sub_pos["m"], sub_pos["prob_high_error"],
                    marker="o", markersize=4, color=color,
                    label=f"$\\rho = {rho:.1f}$", linewidth=1.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of queries $m$")
    ax.set_ylabel("$P[\\mathrm{error} \\geq 0.5]$")
    ax.set_ylim(bottom=1e-3, top=1.5)
    ax.legend(title="$\\rho$", fontsize=9, title_fontsize=10)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/exp2_error_vs_m_rho.pdf")
    plt.close(fig)


def plot_figure1(df_exp1: pd.DataFrame, df_exp2: pd.DataFrame,
                 df_exp3: pd.DataFrame, output_dir: str = "results/figures"):
    """Combined Figure 1: three panels for Exp 1, 2, and 3."""
    set_paper_style()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5))

    # --- Panel A: vary r ---
    r_values = sorted(df_exp1["r"].unique())

    for i, r in enumerate(r_values):
        color = PALETTE[i % len(PALETTE)]
        sub = df_exp1[df_exp1["r"] == r]
        mask = sub["prob_high_error"] > 0
        sub_pos = sub[mask]
        if len(sub_pos) > 0:
            ax1.plot(sub_pos["m"], sub_pos["prob_high_error"],
                     marker="o", markersize=3, color=color,
                     label=f"$r = {r}$", linewidth=1.5)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Number of queries $m$")
    ax1.set_ylabel("$P[\\mathrm{error} \\geq 0.5]$")
    ax1.set_ylim(bottom=1e-3, top=1.5)
    ax1.set_title("(a) Varying rank $r$  ($p = 0.3$)", fontsize=11)
    ax1.legend(fontsize=7, loc="upper right", ncol=2)

    # --- Panel B: vary rho ---
    rho_values = sorted(df_exp2["rho"].unique())

    for i, rho in enumerate(rho_values):
        color = PALETTE[i % len(PALETTE)]
        sub = df_exp2[df_exp2["rho"] == rho]
        mask = sub["prob_high_error"] > 0
        sub_pos = sub[mask]
        if len(sub_pos) > 0:
            ax2.plot(sub_pos["m"], sub_pos["prob_high_error"],
                     marker="o", markersize=3, color=color,
                     label=f"$\\rho = {rho:.1f}$", linewidth=1.5)

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Number of queries $m$")
    ax2.set_ylim(bottom=1e-3, top=1.5)
    ax2.set_title("(b) Varying $\\rho$  ($r = 5$)", fontsize=11)
    ax2.legend(fontsize=8, loc="upper right")

    # --- Panel C: query distribution ---
    dist_labels = {"uniform": "Uniform", "signal": "Signal-concentrated",
                   "orthogonal": "Orthogonal-concentrated"}

    for i, (dist_name, label) in enumerate(dist_labels.items()):
        sub = df_exp3[df_exp3["distribution"] == dist_name]
        ax3.plot(sub["m"], sub["accuracy"],
                 marker="o", markersize=3, color=PALETTE[i],
                 label=label, linewidth=1.5)

    ax3.set_xscale("log")
    ax3.set_xlabel("Number of queries $m$")
    ax3.set_ylabel("Classification accuracy")
    ax3.set_ylim(0.4, 1.05)
    ax3.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax3.set_title("(c) Query distribution  ($r = 5, p = 0.3$)", fontsize=11)
    ax3.legend(fontsize=8, loc="lower right")

    fig.tight_layout()

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

    ax.set_xscale("log")
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
