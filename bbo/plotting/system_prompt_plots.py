"""Plotting functions for the system prompt auditing experiment.

Figure 3 panels (a,b): Qualitative structure (MDS scatter + SVD spectrum)
Figure 4 row 1 (a,b,c): Quantitative results (error vs m)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from pathlib import Path

from bbo.plotting.style import set_paper_style, PALETTE
from bbo.distances.energy import pairwise_energy_distances_t0
from bbo.embedding.mds import ClassicalMDS


def plot_figure3_system_prompt(
    responses: np.ndarray,
    labels: np.ndarray,
    signal_indices: np.ndarray,
    orthogonal_indices: np.ndarray,
    ax_mds_top,
    ax_mds_bot,
    ax_svd,
    m_mds: int = 10,
    seed: int = 0,
):
    """Plot Figure 3 panels (a) and (b) for the system prompt experiment.

    (a) MDS scatter: signal vs orthogonal queries (stacked sub-panels)
    (b) Singular value spectrum of D
    """
    rng = np.random.RandomState(seed)

    # Sample queries for MDS
    sens_sub = rng.choice(signal_indices, size=m_mds, replace=False)
    orth_sub = rng.choice(orthogonal_indices, size=m_mds, replace=False)

    D_sens = pairwise_energy_distances_t0(responses, sens_sub)
    X_sens = ClassicalMDS(n_components=2).fit_transform(D_sens)

    D_orth = pairwise_energy_distances_t0(responses, orth_sub)
    X_orth = ClassicalMDS(n_components=2).fit_transform(D_orth)

    class0_mask = labels == 0
    class1_mask = labels == 1

    for ax, X, is_top, title in [
        (ax_mds_top, X_sens, True, f'(a) Signal queries, $m={m_mds}$'),
        (ax_mds_bot, X_orth, False, f'"Orthogonal" queries, $m={m_mds}$'),
    ]:
        ax.scatter(X[class0_mask, 0], X[class0_mask, 1],
                   c=[PALETTE[0]], marker="o", s=8, alpha=0.7, zorder=2)
        ax.scatter(X[class1_mask, 0], X[class1_mask, 1],
                   c=[PALETTE[1]], marker="s", s=8, alpha=0.7, zorder=2)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(title)
        ax.set_ylabel("MDS 2")

        if is_top:
            legend_elements = [
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=PALETTE[0], markersize=4, label="Neutral"),
                Line2D([0], [0], marker="s", color="w",
                       markerfacecolor=PALETTE[1], markersize=4, label="Biased"),
            ]
            ax.legend(handles=legend_elements, loc="best", fontsize=4)
        else:
            ax.set_xlabel("MDS 1")

    # SVD spectrum
    query_sets = [
        (signal_indices, "Signal", PALETTE[1], "-"),
        (orthogonal_indices, '"Orthogonal"', PALETTE[2], "--"),
    ]

    n_show = 50
    for q_idx, label, color, ls in query_sets:
        D = pairwise_energy_distances_t0(responses, q_idx)
        _, sv, _ = np.linalg.svd(D, full_matrices=False)
        sv = sv / sv[0]
        k = min(n_show, len(sv))
        ax_svd.plot(np.arange(1, k + 1), sv[:k],
                    color=color, linestyle=ls, linewidth=1.2,
                    marker="o", markersize=2, label=label)

    ax_svd.set_xlabel("Component $r$")
    ax_svd.set_ylabel("$\\sigma_r / \\sigma_1$")
    ax_svd.set_title("(b) Singular values of $D$")
    ax_svd.legend(loc="upper right", fontsize=4)


def plot_figure4_row1(
    config,
    ax_a,
    ax_b,
    ax_c,
    default_base_model: str = "ministral-8b",
    default_embedding_model: str = "nomic-embed-text-v1.5",
):
    """Plot Figure 4 row 1: quantitative results for system prompt experiment.

    (a) Error vs m by query type (+ baselines)
    (b) Error vs m across base models
    (c) Error vs m across embedding models
    """
    # --- Panel (a): Error vs m by query type ---
    csv_path = config.classification_csv(default_base_model, default_embedding_model)
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df_mds = df[df["method"] == "mds"]

        dist_styles = {"relevant": ("-", "Signal"), "orthogonal": ("--", '"Orthogonal"'),
                       "uniform": (":", "Uniform")}
        dist_colors = {"relevant": PALETTE[1], "orthogonal": PALETTE[2],
                       "uniform": PALETTE[0]}

        # Use largest available n for panel (a)
        available_n = sorted(df_mds["n"].unique())
        n_plot = available_n[-1] if available_n else max(config.n_values)
        sub = df_mds[df_mds["n"] == n_plot]

        for dist_name, (ls, label) in dist_styles.items():
            data = sub[sub["distribution"] == dist_name].sort_values("m")
            if data.empty:
                continue
            ax_a.plot(data["m"], 1 - data["mean_accuracy"], marker="o", markersize=2,
                      color=dist_colors[dist_name], linestyle=ls, linewidth=0.8,
                      label=label)

        # Baselines
        df_concat = df[(df["method"] == "concat") & (df["n"] == n_plot)]
        if not df_concat.empty:
            ax_a.plot(df_concat.sort_values("m")["m"],
                      1 - df_concat.sort_values("m")["mean_accuracy"],
                      marker="^", markersize=2, color="gray", linestyle="-.",
                      linewidth=0.8, label="Concat")

        df_pca = df[(df["method"] == "pca") & (df["n"] == n_plot)]
        if not df_pca.empty:
            ax_a.plot(df_pca.sort_values("m")["m"],
                      1 - df_pca.sort_values("m")["mean_accuracy"],
                      marker="v", markersize=2, color="gray", linestyle=":",
                      linewidth=0.8, label="PCA")

        df_sbq = df[(df["method"] == "single_best_query") & (df["n"] == n_plot)]
        if not df_sbq.empty:
            ax_a.axhline(y=1 - df_sbq["mean_accuracy"].values[0],
                         color="gray", linestyle="--", linewidth=0.6,
                         label="Best single query")

    ax_a.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3, linewidth=0.5)
    ax_a.set_xscale("log")
    ax_a.set_xlabel("Number of queries $m$")
    ax_a.set_ylabel("Mean error")
    ax_a.set_title("(a) Error vs $m$ by query type")
    ax_a.legend(loc="upper right", fontsize=4, ncol=2)

    # --- Panel (b): Error vs m across base models ---
    # Use available n values from data, not config
    all_n = sorted(df["n"].unique()) if csv_path.exists() else sorted(config.n_values)
    n_values_style = {all_n[-1]: "-", all_n[0]: "--"} if len(all_n) >= 2 else {all_n[0]: "-"}
    for i, bm in enumerate(config.base_models):
        csv_path = config.classification_csv(bm, default_embedding_model)
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df_mds = df[(df["method"] == "mds") & (df["distribution"] == "relevant")]
        color = PALETTE[i % len(PALETTE)]

        for n_val, ls in n_values_style.items():
            sub = df_mds[df_mds["n"] == n_val].sort_values("m")
            if sub.empty:
                continue
            label = f"{bm}" if ls == "-" else None
            ax_b.plot(sub["m"], 1 - sub["mean_accuracy"], marker="o", markersize=2,
                      color=color, linestyle=ls, linewidth=0.8, label=label)

    ax_b.set_xscale("log")
    ax_b.set_xlabel("Number of queries $m$")
    ax_b.set_ylabel("Mean error")
    ax_b.set_title("(b) Error vs $m$ across base models")

    # Legend: model colors + n line styles
    handles = [Line2D([0], [0], color=PALETTE[i], lw=1, label=bm)
               for i, bm in enumerate(config.base_models)]
    handles += [Line2D([0], [0], color="0.4", linestyle=ls, lw=1,
                        label=f"$n={n}$")
                for n, ls in n_values_style.items()]
    ax_b.legend(handles=handles, loc="upper right", fontsize=3.5, ncol=2)

    # --- Panel (c): Error vs m across embedding models ---
    for i, em in enumerate(config.embedding_models):
        csv_path = config.classification_csv(default_base_model, em)
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df_mds = df[(df["method"] == "mds") & (df["distribution"] == "relevant")]
        color = PALETTE[i % len(PALETTE)]

        for n_val, ls in n_values_style.items():
            sub = df_mds[df_mds["n"] == n_val].sort_values("m")
            if sub.empty:
                continue
            label = em.split("/")[-1] if ls == "-" else None
            ax_c.plot(sub["m"], 1 - sub["mean_accuracy"], marker="o", markersize=2,
                      color=color, linestyle=ls, linewidth=0.8, label=label)

    ax_c.set_xscale("log")
    ax_c.set_xlabel("Number of queries $m$")
    ax_c.set_ylabel("Mean error")
    ax_c.set_title("(c) Error vs $m$ across embeddings")

    handles = [Line2D([0], [0], color=PALETTE[i], lw=1,
                       label=em.split("/")[-1])
               for i, em in enumerate(config.embedding_models)]
    handles += [Line2D([0], [0], color="0.4", linestyle=ls, lw=1,
                        label=f"$n={n}$")
                for n, ls in n_values_style.items()]
    ax_c.legend(handles=handles, loc="upper right", fontsize=3.5, ncol=2)


def plot_system_prompt_figures(config, output_dir: str = "figures"):
    """Generate Figure 3 (a,b) and Figure 4 row 1 (a,b,c) for system prompt experiment."""
    set_paper_style()
    plt.rcParams.update({
        "font.size": 6,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
    })

    # Load default data (ministral-8b + nomic)
    default_bm = "ministral-8b"
    default_em = "nomic-embed-text-v1.5"
    npz_path = config.npz_path(default_bm, default_em)

    if not npz_path.exists():
        print(f"Embeddings not found at {npz_path}. Run embed step first.")
        return

    data = np.load(str(npz_path), allow_pickle=True)
    responses = data["responses"]
    labels = data["labels"]
    signal_indices = data["signal_indices"]
    orthogonal_indices = data["orthogonal_indices"]

    # --- Figure 3: Qualitative (panels a, b only — c, d are for RAG) ---
    fig3 = plt.figure(figsize=(5.5, 1.65))
    gs3 = GridSpec(2, 2, figure=fig3, wspace=0.55, hspace=0.55,
                   width_ratios=[1, 1])

    ax3_a_top = fig3.add_subplot(gs3[0, 0])
    ax3_a_bot = fig3.add_subplot(gs3[1, 0])
    ax3_b = fig3.add_subplot(gs3[:, 1])

    plot_figure3_system_prompt(
        responses, labels, signal_indices, orthogonal_indices,
        ax3_a_top, ax3_a_bot, ax3_b, m_mds=10,
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig3.savefig(f"{output_dir}/figure3_system_prompt.pdf")
    plt.close(fig3)
    print(f"Saved Figure 3 (system prompt) to {output_dir}/figure3_system_prompt.pdf")

    # --- Figure 4 row 1: Quantitative ---
    fig4 = plt.figure(figsize=(5.5, 1.8))
    gs4 = GridSpec(1, 3, figure=fig4, wspace=0.45)

    ax4_a = fig4.add_subplot(gs4[0, 0])
    ax4_b = fig4.add_subplot(gs4[0, 1])
    ax4_c = fig4.add_subplot(gs4[0, 2])

    plot_figure4_row1(config, ax4_a, ax4_b, ax4_c,
                      default_base_model=default_bm,
                      default_embedding_model=default_em)

    fig4.savefig(f"{output_dir}/figure4_row1_system_prompt.pdf")
    plt.close(fig4)
    print(f"Saved Figure 4 row 1 to {output_dir}/figure4_row1_system_prompt.pdf")
