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
    ("uniform",        PALETTE[2], "-",  "s", "Uniform"),
    ("uniform_signal", PALETTE[0], "-",  "o", "Est. signal"),
    ("stepwise",       PALETTE[3], "-",  "^", "Forward"),
    ("backward",       PALETTE[1], "-",  "v", "Backward"),
    ("bidirectional",  PALETTE[4], "-",  "D", "Bidirectional"),
]

DATASET_TITLES = ["LoRA", "System Prompt", "RAG"]


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
    axes[0].legend(handles=leg, loc="upper right", fontsize=6, ncol=1)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved pilot selection figure to {output_path}")


def plot_pilot_oracle(
    motivating_csv: str,
    system_prompt_csv: str,
    rag_csv: str,
    output_path: str = "figures/figure_pilot_selection.pdf",
):
    """1×3 figure: estimated vs oracle signal/orthogonal/uniform."""
    set_paper_style()

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.6), sharey=True)

    datasets = list(zip(DATASET_TITLES,
                        [motivating_csv, system_prompt_csv, rag_csv]))

    sel_config = [
        ("uniform_signal",     PALETTE[0], "-",  "o"),
        ("uniform_orthogonal", PALETTE[1], "-",  "v"),
        ("uniform",            PALETTE[2], "-",  "s"),
        ("oracle_signal",      PALETTE[0], "--", "o"),
        ("oracle_orthogonal",  PALETTE[1], "--", "v"),
    ]

    n_plot = 80

    for ax, (title, csv_path) in zip(axes, datasets):
        df = pd.read_csv(csv_path)

        for sel_name, color, ls, marker in sel_config:
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

    # Legend: colors for query type, linestyle for estimated vs oracle
    leg = [
        Line2D([0], [0], color=PALETTE[0], lw=1.0, label="Signal"),
        Line2D([0], [0], color=PALETTE[1], lw=1.0, label="Orthogonal"),
        Line2D([0], [0], color=PALETTE[2], lw=1.0, label="Uniform"),
        Line2D([0], [0], color="0.4", linestyle="-", lw=0.8, label="Estimated"),
        Line2D([0], [0], color="0.4", linestyle="--", lw=0.8, label="Oracle"),
    ]
    axes[0].legend(handles=leg, loc="upper right", fontsize=3.5, ncol=2)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved pilot oracle figure to {output_path}")


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


# d-value styling for PCA stepwise figures
_D_CONFIG = [
    (2,   PALETTE[0], "-",  "o"),
    (8,   PALETTE[1], "-",  "s"),
    (32,  PALETTE[3], "-",  "^"),
    (128, PALETTE[4], "-",  "D"),
    (768, PALETTE[5], "-",  "v"),
]


def plot_pca_stepwise(
    motivating_csv: str,
    system_prompt_csv: str,
    rag_csv: str,
    output_path: str = "figures/figure_pca_stepwise.pdf",
):
    """1×3 figure: PCA bidirectional stepwise at varying d across datasets."""
    set_paper_style()

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.6), sharey=True)

    datasets = list(zip(DATASET_TITLES,
                        [motivating_csv, system_prompt_csv, rag_csv]))

    for ax, (title, csv_path) in zip(axes, datasets):
        if not Path(csv_path).exists():
            ax.set_title(title)
            continue
        df = pd.read_csv(csv_path)

        for d, color, ls, marker in _D_CONFIG:
            sel = f"pca_d{d}"
            sub = df[df["selector"] == sel].sort_values("m")
            if sub.empty:
                continue
            ax.plot(sub["m"], sub["mean_error"],
                    marker=marker, markersize=2, color=color,
                    linestyle=ls, linewidth=0.8)

        # Uniform baseline
        sub_u = df[df["selector"] == "uniform"].sort_values("m")
        if not sub_u.empty:
            ax.plot(sub_u["m"], sub_u["mean_error"],
                    marker="x", markersize=2.5, color=PALETTE[2],
                    linestyle="--", linewidth=0.8)

        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3, linewidth=0.5)
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 0.55)
        ax.set_xlabel("Queries $m$")
        ax.set_title(title)

    axes[0].set_ylabel("Mean error")

    # Shared legend
    leg = []
    for d, color, ls, marker in _D_CONFIG:
        leg.append(Line2D([0], [0], color=color, linestyle=ls, lw=0.8,
                          marker=marker, markersize=2, label=f"$d={d}$"))
    leg.append(Line2D([0], [0], color=PALETTE[2], linestyle="--", lw=0.8,
                      marker="x", markersize=2.5, label="Uniform"))
    axes[0].legend(handles=leg, loc="upper right", fontsize=4, ncol=2)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved PCA stepwise figure to {output_path}")


_POOL_SEL_CONFIG = [
    ("stepwise",    PALETTE[0], "-",  "o", "Bi-dir. stepwise (BIC)"),
    ("random_m",    PALETTE[2], "--", "s", "Uniform ($m$ chosen by BIC)"),
    ("uniform_all", PALETTE[1], ":",  "^", "All queries"),
]


def _plot_pool_panel(ax, df, x_col, fixed_col, fixed_val):
    """Plot selector curves on a single axis for pool experiments."""
    sub_all = df[df[fixed_col] == fixed_val]
    if sub_all.empty:
        return
    for sel, color, ls, marker, label in _POOL_SEL_CONFIG:
        sub = sub_all[sub_all["selector"] == sel].sort_values(x_col)
        if sub.empty:
            continue
        ax.errorbar(sub[x_col], sub["mean_error"], yerr=sub["std_error"],
                     marker=marker, markersize=2.5, color=color,
                     linestyle=ls, linewidth=0.8,
                     capsize=2, capthick=0.5)
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 0.55)
    x_vals = sorted(sub_all[x_col].unique())
    ax.set_xticks(x_vals)
    ax.set_xticklabels([str(int(v)) for v in x_vals])
    ax.xaxis.set_minor_locator(plt.NullLocator())


def plot_pool_vary_pool(
    motivating_csv: str,
    system_prompt_csv: str,
    rag_csv: str,
    fixed_n: int = 50,
    output_path: str = "figures/figure_pool_vary_pool.pdf",
):
    """1×3 figure: error vs query pool size at fixed n_train."""
    set_paper_style()

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.6), sharey=True)
    datasets = list(zip(DATASET_TITLES,
                        [motivating_csv, system_prompt_csv, rag_csv]))

    for ax, (title, csv_path) in zip(axes, datasets):
        ax.set_title(title)
        if not Path(csv_path).exists():
            continue
        df = pd.read_csv(csv_path)
        _plot_pool_panel(ax, df, "pool_size", "n_train", fixed_n)
        ax.set_xlabel("Query pool size $M$")

    axes[0].set_ylabel("Mean error")

    leg = [Line2D([0], [0], color=c, linestyle=ls, lw=0.8,
                  marker=mk, markersize=2.5, label=lbl)
           for _, c, ls, mk, lbl in _POOL_SEL_CONFIG]
    axes[0].legend(handles=leg, loc="upper right", fontsize=4)

    fig.suptitle(f"$n = {fixed_n}$ models", fontsize=8, y=1.02)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved pool-vary-pool figure to {output_path}")


def plot_pool_vary_n(
    motivating_csv: str,
    system_prompt_csv: str,
    rag_csv: str,
    fixed_pool: int = 50,
    output_path: str = "figures/figure_pool_vary_n.pdf",
):
    """1×3 figure: error vs n_train at fixed pool size."""
    set_paper_style()

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.6), sharey=True)
    datasets = list(zip(DATASET_TITLES,
                        [motivating_csv, system_prompt_csv, rag_csv]))

    for ax, (title, csv_path) in zip(axes, datasets):
        ax.set_title(title)
        if not Path(csv_path).exists():
            continue
        df = pd.read_csv(csv_path)
        _plot_pool_panel(ax, df, "n_train", "pool_size", fixed_pool)
        ax.set_xlabel("Models $n$")

    axes[0].set_ylabel("Mean error")

    leg = [Line2D([0], [0], color=c, linestyle=ls, lw=0.8,
                  marker=mk, markersize=2.5, label=lbl)
           for _, c, ls, mk, lbl in _POOL_SEL_CONFIG]
    axes[0].legend(handles=leg, loc="upper right", fontsize=4)

    fig.suptitle(f"$M = {fixed_pool}$ query pool", fontsize=8, y=1.02)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved pool-vary-n figure to {output_path}")


def plot_bic_m_boxplot(
    motivating_csv: str,
    system_prompt_csv: str,
    rag_csv: str,
    fixed_n: int = 50,
    output_path: str = "figures/figure_bic_m_boxplot.pdf",
):
    """1×3 box plot: distribution of m selected by BIC vs pool size M."""
    set_paper_style()

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.6), sharey=True)
    datasets = list(zip(DATASET_TITLES,
                        [motivating_csv, system_prompt_csv, rag_csv]))

    for ax, (title, csv_path) in zip(axes, datasets):
        ax.set_title(title)
        if not Path(csv_path).exists():
            continue
        df = pd.read_csv(csv_path)
        pool_sizes = sorted(df["pool_size"].unique())
        data = [df[df["pool_size"] == ps]["m_selected"].values for ps in pool_sizes]
        bp = ax.boxplot(data, positions=range(len(pool_sizes)),
                        widths=0.5, patch_artist=True,
                        medianprops=dict(color="k", lw=0.8),
                        flierprops=dict(marker=".", markersize=2, alpha=0.5),
                        boxprops=dict(lw=0.6),
                        whiskerprops=dict(lw=0.6),
                        capprops=dict(lw=0.6))
        for patch in bp["boxes"]:
            patch.set_facecolor(PALETTE[0])
            patch.set_alpha(0.6)
        ax.set_xticks(range(len(pool_sizes)))
        ax.set_xticklabels([str(ps) for ps in pool_sizes])
        ax.set_xlabel("Query pool size $M$")

    axes[0].set_ylabel("Queries selected ($m$)")

    fig.suptitle(f"$n = {fixed_n}$ models", fontsize=8, y=1.02)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved BIC m boxplot to {output_path}")


def plot_paired_diff_hist(
    motivating_csv: str,
    system_prompt_csv: str,
    rag_csv: str,
    fixed_n: int = 50,
    output_path: str = "figures/figure_paired_diff_hist.pdf",
):
    """3×3 histograms of paired error difference (Uniform - BIC stepwise).

    Rows = pool sizes M, columns = datasets.
    """
    from scipy.stats import wilcoxon
    set_paper_style()

    csv_paths = [motivating_csv, system_prompt_csv, rag_csv]
    dfs = {}
    pool_sizes = None
    for title, csv_path in zip(DATASET_TITLES, csv_paths):
        if Path(csv_path).exists():
            df = pd.read_csv(csv_path)
            dfs[title] = df
            if pool_sizes is None:
                pool_sizes = sorted(df["pool_size"].unique())

    if pool_sizes is None:
        return

    n_rows = len(pool_sizes)
    fig, axes = plt.subplots(n_rows, 3, figsize=(5.5, 1.4 * n_rows),
                             sharey=False)

    for col, title in enumerate(DATASET_TITLES):
        axes[0, col].set_title(title)
        if title not in dfs:
            continue
        df = dfs[title]

        for row, ps in enumerate(pool_sizes):
            ax = axes[row, col]
            sub = df[df["pool_size"] == ps]
            diff = sub["error_random_m"].values - sub["error_stepwise"].values
            med = np.median(diff)

            n_nonzero = np.sum(diff != 0)
            if n_nonzero >= 10:
                _, pval = wilcoxon(diff)
                p_str = f"$p={pval:.3f}$" if pval >= 0.001 else f"$p<0.001$"
            else:
                p_str = "$p=$n/a"

            ax.hist(diff, color=PALETTE[0], alpha=0.7)
            ax.axvline(0, color="k", ls="-", lw=1.5)
            ax.axvline(med, color=PALETTE[3], ls="--", lw=1.0)

            lbl = f"med={med:+.3f}, {p_str}"
            ax.text(0.97, 0.92, lbl, transform=ax.transAxes,
                    fontsize=4.5, ha="right", va="top")

            if col == 0:
                ax.set_ylabel(f"$M={ps}$", fontsize=7)
            if row == n_rows - 1:
                ax.set_xlabel("$\\Delta$ error (Uniform $-$ BIC)")

    fig.suptitle(f"$n = {fixed_n}$ models", fontsize=8, y=1.01)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved paired diff histogram to {output_path}")
