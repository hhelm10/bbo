"""Plotting functions for synthetic experiments (Exp 1-5)."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.lines as mlines
import seaborn as sns
from pathlib import Path

from bbo.plotting.style import set_paper_style, PALETTE


# ---------------------------------------------------------------------------
# Helpers for log y-axis with a break to display y = 0
# ---------------------------------------------------------------------------

def _zero_pos(n_reps, min_power=None):
    """Position on log axis where y=0 is displayed (below smallest visible tick)."""
    if min_power is not None:
        return 10.0 ** min_power / 3.0
    return 1.0 / (n_reps * 3)


def _map_zeros(y, n_reps, min_power=None):
    """Replace 0 values with zero_pos for log-scale plotting."""
    zp = _zero_pos(n_reps, min_power)
    return np.where(np.asarray(y, dtype=float) > 0, y, zp)


def _smooth_floor(y, n_reps):
    """Smooth floor for continuous theory curves on log scale.

    Uses a temperature-scaled soft-max in log space so the theory line
    visibly curves over ~2 decades before levelling off at zero_pos,
    instead of bending only in the narrow break zone.
    """
    zp = _zero_pos(n_reps)
    y = np.asarray(y, dtype=float)
    log_y = np.log(np.maximum(y, 1e-30))
    log_zp = np.log(zp)
    T = 3.0  # wider T → wider visible transition
    smooth_log = T * np.logaddexp(log_y / T, log_zp / T)
    return np.minimum(np.exp(smooth_log), 1.0)


def _setup_broken_log_y(ax, n_reps, min_power=None):
    """Configure log y-axis with a break to show y = 0.

    Adds:
      - "0" tick at zero_pos (below smallest visible tick)
      - Two diagonal slash marks indicating the axis break
      - Standard log ticks above the break

    Parameters
    ----------
    min_power : int, optional
        Lowest power of 10 to show (e.g., -3 for 10^-3).
        If None, uses floor(log10(1/n_reps)).
    """
    if min_power is not None:
        threshold = 10.0 ** min_power
    else:
        threshold = 1.0 / n_reps
    zp = _zero_pos(n_reps, min_power)

    ax.set_yscale("log")
    ax.set_ylim(bottom=zp * 0.5, top=1.5)

    # --- Custom ticks: "0" plus standard powers of 10 ---
    log_ticks = [10.0 ** k for k in range(0, -5, -1) if 10.0 ** k >= threshold]
    ax.set_yticks([zp] + log_ticks)
    ax.set_yticklabels(
        ["0"] + [f"$10^{{{int(np.log10(t))}}}$" for t in log_ticks]
    )
    ax.yaxis.set_minor_locator(plt.NullLocator())

    # --- Break indicator: two diagonal slashes on the y-axis ---
    break_mid = np.sqrt(zp * threshold)  # geometric midpoint of gap
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for ratio in [0.78, 1.28]:
        by = break_mid * ratio
        ax.plot(
            [-0.02, 0.02],
            [by * 0.85, by / 0.85],
            transform=trans, color="k", lw=0.8, clip_on=False,
        )


# ---------------------------------------------------------------------------
# Individual experiment plots
# ---------------------------------------------------------------------------

def _theory_bound(m_values, r, rho, clip_min=None):
    """Theoretical bound: min(r * rho^m, 1), clipped below at clip_min."""
    m = np.asarray(m_values, dtype=float)
    bound = np.minimum(r * rho ** m, 1.0)
    if clip_min is not None:
        bound = np.clip(bound, clip_min, None)
    return bound


def _gamma_bound(n_values, r, clip_min=None):
    """Sample complexity bound: gamma(n) <= min(1, 2^r * exp(-n / 2^r)).

    This is the coupon-collector-style bound capturing the requirement
    that each of the 2^r hypercube vertices needs at least one model.
    """
    n = np.asarray(n_values, dtype=float)
    two_r = 2.0 ** r
    bound = np.minimum(two_r * np.exp(-n / two_r), 1.0)
    if clip_min is not None:
        bound = np.clip(bound, clip_min, None)
    return bound


def plot_exp1(df: pd.DataFrame, output_dir: str = "results/figures"):
    """Plot Exp 1: log P[error >= 0.5] vs m, one curve per r."""
    set_paper_style()
    n_reps = int(df["n_reps"].iloc[0])
    r_values = sorted(df["r"].unique())
    clip_min = 1.0 / n_reps  # don't let bounds cross the break

    fig, ax = plt.subplots(figsize=(7, 5))

    m_dense = np.logspace(np.log10(1), np.log10(100), 200)

    for i, r in enumerate(r_values):
        color = PALETTE[i % len(PALETTE)]
        sub = df[df["r"] == r]
        rho = sub["rho"].iloc[0]
        y = _map_zeros(sub["prob_high_error"].values, n_reps)
        ax.plot(sub["m"], y, marker="o", markersize=4, color=color,
                label=f"$r = {r}$", linewidth=1.5)
        # Theoretical bound as dashed line (clipped above 1/n_reps)
        bound = _theory_bound(m_dense, r, rho, clip_min=clip_min)
        ax.plot(m_dense, bound, color=color, linestyle="--", linewidth=1.0,
                alpha=0.6)

    # Single legend entry for theory
    theory_handle = mlines.Line2D([], [], color="gray", linestyle="--",
                                  linewidth=1.8, alpha=0.8, label="$r\\rho^m$")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(theory_handle)
    labels.append("$r\\rho^m$")
    ax.legend(handles, labels, title="Rank $r$", fontsize=9, title_fontsize=10)

    ax.set_xscale("log")
    _setup_broken_log_y(ax, n_reps)
    ax.set_xlabel("Number of queries $m$")
    ax.set_ylabel("$P[\\mathrm{error} \\geq 0.5]$")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/exp1_error_vs_m_rank.pdf")
    plt.close(fig)


def plot_exp2(df: pd.DataFrame, output_dir: str = "results/figures"):
    """Plot Exp 2: log P[error >= 0.5] vs m, one curve per rho."""
    set_paper_style()
    n_reps = int(df["n_reps"].iloc[0])
    rho_values = sorted(df["rho"].unique())
    clip_min = 1.0 / n_reps

    fig, ax = plt.subplots(figsize=(7, 5))

    m_dense = np.logspace(np.log10(1), np.log10(100), 200)

    for i, rho in enumerate(rho_values):
        color = PALETTE[i % len(PALETTE)]
        sub = df[df["rho"] == rho]
        r = 5  # Exp 2 uses fixed r=5
        y = _map_zeros(sub["prob_high_error"].values, n_reps)
        ax.plot(sub["m"], y, marker="o", markersize=4, color=color,
                label=f"$\\rho = {rho:.1f}$", linewidth=1.5)
        bound = _theory_bound(m_dense, r, rho, clip_min=clip_min)
        ax.plot(m_dense, bound, color=color, linestyle="--", linewidth=1.0,
                alpha=0.6)

    theory_handle = mlines.Line2D([], [], color="gray", linestyle="--",
                                  linewidth=1.8, alpha=0.8, label="$r\\rho^m$")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(theory_handle)
    labels.append("$r\\rho^m$")
    ax.legend(handles, labels, title="$\\rho$", fontsize=9, title_fontsize=10)

    ax.set_xscale("log")
    _setup_broken_log_y(ax, n_reps)
    ax.set_xlabel("Number of queries $m$")
    ax.set_ylabel("$P[\\mathrm{error} \\geq 0.5]$")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/exp2_error_vs_m_rho.pdf")
    plt.close(fig)


def plot_figure1(df_exp1: pd.DataFrame, df_exp2: pd.DataFrame,
                 df_exp3: pd.DataFrame, df_exp4: pd.DataFrame = None,
                 output_dir: str = "results/figures"):
    """Combined Figure 1: four panels for Exp 1-4 in a single row.

    Sized for ICLR/NeurIPS full-textwidth (≈5.5 in).
    """
    set_paper_style()

    fig, axes = plt.subplots(1, 4, figsize=(5.5, 1.6))
    ax1, ax2, ax4, ax3 = axes

    # --- Panel A: vary r ---
    n_reps1 = int(df_exp1["n_reps"].iloc[0])
    r_values = sorted(df_exp1["r"].unique())
    m_dense = np.logspace(np.log10(1), np.log10(100), 200)

    for i, r in enumerate(r_values):
        color = PALETTE[i % len(PALETTE)]
        sub = df_exp1[df_exp1["r"] == r]
        rho = sub["rho"].iloc[0]
        y = _map_zeros(sub["prob_high_error"].values, n_reps1)
        ax1.plot(sub["m"], y, marker="o", color=color,
                 label=f"$r = {r}$")
        bound = _theory_bound(m_dense, r, rho)
        ax1.plot(m_dense, _smooth_floor(bound, n_reps1), color=color,
                 linestyle="--", linewidth=1.0, alpha=0.7)

    theory_handle = mlines.Line2D([], [], color="gray", linestyle="--",
                                  linewidth=1.0, alpha=0.7, label="$r\\rho^m$")
    h1, l1 = ax1.get_legend_handles_labels()
    h1.append(theory_handle)
    l1.append("$r\\rho^m$")

    # Use consistent n_reps for shared y-axis
    n_reps = n_reps1
    ax1.set_xscale("log")
    _setup_broken_log_y(ax1, n_reps)
    ax1.set_xlabel("Queries $m$")
    ax1.set_ylabel("$P[\\mathrm{error} \\geq 0.5]$")
    ax1.set_title("(a) Varying rank $r$\n$n\\!=\\!100,\\; \\rho\\!\\approx\\!0.7,\\; M\\!=\\!100$",
                  fontsize=7)
    ax1.legend(h1, l1, loc="lower left", bbox_to_anchor=(0.02, 0), ncol=2)

    # --- Panel B: vary rho ---
    n_reps2 = int(df_exp2["n_reps"].iloc[0])
    rho_values = sorted(df_exp2["rho"].unique())

    for i, rho in enumerate(rho_values):
        color = PALETTE[i % len(PALETTE)]
        sub = df_exp2[df_exp2["rho"] == rho]
        r_exp2 = 5
        y = _map_zeros(sub["prob_high_error"].values, n_reps)
        ax2.plot(sub["m"], y, marker="o", color=color,
                 label=f"$\\rho = {rho:.1f}$")
        bound = _theory_bound(m_dense, r_exp2, rho)
        ax2.plot(m_dense, _smooth_floor(bound, n_reps), color=color,
                 linestyle="--", linewidth=1.0, alpha=0.7)

    theory_handle2 = mlines.Line2D([], [], color="gray", linestyle="--",
                                   linewidth=1.0, alpha=0.7, label="$r\\rho^m$")
    h2, l2 = ax2.get_legend_handles_labels()
    h2.append(theory_handle2)
    l2.append("$r\\rho^m$")

    ax2.set_xscale("log")
    _setup_broken_log_y(ax2, n_reps)
    ax2.set_yticklabels([])
    ax2.set_xlabel("Queries $m$")
    ax2.set_title("(b) Varying $\\rho$\n$n\\!=\\!100,\\; r\\!=\\!5,\\; M\\!=\\!100$",
                  fontsize=7)
    ax2.legend(h2, l2, loc="lower left", bbox_to_anchor=(0.02, 0))

    # --- Panel D: query distribution ---
    n_reps3 = int(df_exp3["n_reps"].iloc[0])
    dist_labels = {"uniform": "Uniform", "signal": "Signal",
                   "orthogonal": "Orthogonal"}
    dist_colors = {"uniform": PALETTE[0], "signal": PALETTE[1],
                   "orthogonal": PALETTE[2]}

    if "signal_prob" in df_exp3.columns:
        sp_values = sorted(df_exp3["signal_prob"].unique())
    else:
        sp_values = [None]

    linestyles = ["-", "--", ":", "-."]

    for j, sp in enumerate(sp_values):
        if sp is not None:
            sub_sp = df_exp3[df_exp3["signal_prob"] == sp]
        else:
            sub_sp = df_exp3
        ls = linestyles[j % len(linestyles)]

        for dist_name, label in dist_labels.items():
            sub = sub_sp[sub_sp["distribution"] == dist_name]
            if sub.empty:
                continue
            color = dist_colors[dist_name]
            full_label = label if j == 0 else None
            y3 = _map_zeros(sub["prob_high_error"].values, n_reps)
            ax3.plot(sub["m"], y3, marker="o", color=color,
                     label=full_label, linestyle=ls)

    if len(sp_values) > 1:
        dist_handles = [mlines.Line2D([], [], color=dist_colors[d],
                        label=dist_labels[d]) for d in dist_labels]
        rho_handles = [mlines.Line2D([], [], color="gray",
                       linestyle=linestyles[j],
                       label=f"$\\rho = {1-sp:.1f}$")
                       for j, sp in enumerate(sp_values)]
        ax3.legend(handles=dist_handles + rho_handles,
                   loc="lower left", bbox_to_anchor=(0.02, 0), ncol=2)
    else:
        ax3.legend(loc="lower left", bbox_to_anchor=(0.02, 0))

    ax3.set_xscale("log")
    _setup_broken_log_y(ax3, n_reps)
    ax3.set_yticklabels([])
    ax3.set_xlabel("Queries $m$")

    ax3.set_title("(d) Query distribution\n$n\\!=\\!100,\\; r\\!=\\!5,\\; M\\!=\\!100$",
                  fontsize=7)

    # --- Panel C: error vs n ---
    if df_exp4 is not None:
        n_reps4 = int(df_exp4["n_reps"].iloc[0])
        r_exp4 = int(df_exp4["r"].iloc[0])
        m_values_4 = sorted(df_exp4["m"].unique())
        rho4 = 1.0 - 0.3
        zp4 = _zero_pos(n_reps)

        for i, m in enumerate(m_values_4):
            color = PALETTE[i % len(PALETTE)]
            sub = df_exp4[df_exp4["m"] == m]
            y4 = _map_zeros(sub["prob_high_error"].values, n_reps)
            ax4.plot(sub["n_models"], y4, marker="o", color=color,
                     label=f"$m = {m}$")
            bound_val = r_exp4 * rho4 ** m
            if bound_val > zp4:
                ax4.axhline(y=bound_val, color=color, linestyle="--",
                            linewidth=1.0, alpha=0.7)

        theory_handle4 = mlines.Line2D([], [], color="gray", linestyle="--",
                                       linewidth=1.0, alpha=0.7,
                                       label="$r\\rho^m$")
        h4, l4 = ax4.get_legend_handles_labels()
        h4.append(theory_handle4)
        l4.append("$r\\rho^m$")

        ax4.set_xscale("log")
        _setup_broken_log_y(ax4, n_reps)
        ax4.set_yticklabels([])
        ax4.set_xlabel("Models $n$")
        ax4.set_title("(c) Varying $n$\n"
                      f"$r\\!=\\!{r_exp4},\\; \\rho\\!=\\!0.7,\\; M\\!=\\!100$",
                      fontsize=7)
        ax4.legend(h4, l4, loc="lower left", bbox_to_anchor=(0.02, 0))
    else:
        ax4.set_visible(False)

    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.22, top=0.82, wspace=0.08)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/figure1_error_vs_m.pdf")
    plt.close(fig)


def plot_exp3(df: pd.DataFrame, output_dir: str = "results/figures"):
    """Plot Exp 3: P[error >= 0.5] vs m for three query distributions."""
    set_paper_style()
    n_reps = int(df["n_reps"].iloc[0])
    fig, ax = plt.subplots(figsize=(6, 4))

    dist_labels = {"uniform": "Uniform", "signal": "Signal-concentrated",
                   "orthogonal": "Orthogonal-concentrated"}
    linestyles = ["-", "--", ":", "-."]

    if "signal_prob" in df.columns:
        sp_values = sorted(df["signal_prob"].unique())
    else:
        sp_values = [None]

    for j, sp in enumerate(sp_values):
        sub_sp = df[df["signal_prob"] == sp] if sp is not None else df
        ls = linestyles[j % len(linestyles)]
        suffix = f" ($\\rho={1-sp:.1f}$)" if sp is not None and len(sp_values) > 1 else ""

        for i, (dist_name, label) in enumerate(dist_labels.items()):
            sub = sub_sp[sub_sp["distribution"] == dist_name]
            if sub.empty:
                continue
            y = _map_zeros(sub["prob_high_error"].values, n_reps)
            ax.plot(sub["m"], y,
                    marker="o", markersize=4, color=PALETTE[i],
                    label=f"{label}{suffix}", linewidth=2, linestyle=ls)

    ax.set_xscale("log")
    _setup_broken_log_y(ax, n_reps)
    ax.set_xlabel("Number of queries $m$")
    ax.set_ylabel("$P[\\mathrm{error} \\geq 0.5]$")
    ax.legend(fontsize=8)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/exp3_query_distribution.pdf")
    plt.close(fig)


def plot_exp4(df: pd.DataFrame, output_dir: str = "results/figures"):
    """Plot Exp 4: P[error >= 0.5] vs n, one curve per m."""
    set_paper_style()
    n_reps = int(df["n_reps"].iloc[0])
    clip_min = 1.0 / n_reps
    r_exp4 = int(df["r"].iloc[0])
    rho4 = 1.0 - 0.3  # default signal_prob = 0.3
    fig, ax = plt.subplots(figsize=(6, 4))

    m_values = sorted(df["m"].unique())
    for i, m in enumerate(m_values):
        sub = df[df["m"] == m]
        y = _map_zeros(sub["prob_high_error"].values, n_reps)
        ax.plot(sub["n_models"], y,
                marker="o", markersize=4, color=PALETTE[i],
                label=f"$m = {m}$", linewidth=1.5)
        bound_val = r_exp4 * rho4 ** m
        zp = _zero_pos(n_reps)
        ax.axhline(y=max(bound_val, zp), color=PALETTE[i], linestyle="--",
                   linewidth=1.8, alpha=0.8)

    # Single legend entry for theory
    theory_handle = mlines.Line2D([], [], color="gray", linestyle="--",
                                  linewidth=1.8, alpha=0.8,
                                  label="$r\\rho^m$")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(theory_handle)
    labels.append("$r\\rho^m$")

    ax.set_xscale("log")
    _setup_broken_log_y(ax, n_reps)
    ax.set_xlabel("Number of models $n$")
    ax.set_ylabel("$P[\\mathrm{error} \\geq 0.5]$")
    ax.legend(handles, labels, fontsize=9)

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
