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
    model_names: np.ndarray = None,
    output_dir: str = "figures",
    big_font: bool = False,
):
    """Figure 1: Motivating example (3 columns).

    Layout (GridSpec 2x3):
        gs[0, 0] = (a) MDS scatter — signal queries, m=5
        gs[1, 0] =     MDS scatter — orthogonal queries, m=5
        gs[:, 1] = (b) Error vs m for n=80 (signal, orthogonal, uniform)
        gs[:, 2] = (c) Error vs n for m=10 (signal, orthogonal, uniform)
    """
    set_paper_style()
    _bump = 2 if big_font else 0
    plt.rcParams.update({
        "font.size": 6 + _bump,
        "axes.labelsize": 7 + _bump,
        "axes.titlesize": 7 + _bump,
        "xtick.labelsize": 5 + _bump,
        "ytick.labelsize": 5 + _bump,
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

    # Load classification CSV
    df = pd.read_csv(classification_csv)

    # --- Layout ---
    _h = 2.4 if big_font else 1.6
    fig = plt.figure(figsize=(5.5, _h))
    gs = GridSpec(2, 3, figure=fig, wspace=0.55 if not big_font else 0.75,
                  hspace=0.65 if not big_font else 0.85)

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

    # --- Panel (a): MDS scatter at m=5 — signal vs orthogonal ---
    rng = np.random.RandomState(0)
    m_mds = 5
    sig_sub = rng.choice(sensitive_indices, size=m_mds, replace=False)
    orth_sub = rng.choice(orthogonal_indices, size=m_mds, replace=False)

    for ax, qi, title in [
        (ax_a_top, sig_sub, "(a) Signal queries"),
        (ax_a_bot, orth_sub, "Orthogonal queries"),
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

    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=PALETTE[0], markersize=4, label="Class 0"),
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=PALETTE[1], markersize=4, label="Class 1"),
    ]
    _leg_fs = 4 + _bump
    ax_a_top.legend(handles=legend_elements, loc="best", fontsize=_leg_fs)
    ax_a_bot.set_xlabel("MDS 1")

    # --- Panels (b) and (c): Signal vs Orthogonal vs Uniform ---
    df_mds = df[df["method"] == "mds"]

    dist_config = [
        ("relevant",   "-",  PALETTE[0], "Signal"),
        ("orthogonal", "--", PALETTE[1], "Orthogonal"),
        ("uniform",    ":",  PALETTE[2], "Uniform"),
    ]

    # Panel (b): Error vs m for n=50
    n_plot = 50
    for dist_name, ls, color, label in dist_config:
        sub = df_mds[(df_mds["distribution"] == dist_name) &
                     (df_mds["n"] == n_plot) &
                     (df_mds["m"] >= 2)].sort_values("m")
        if not sub.empty:
            ax_b.plot(sub["m"], 1 - sub["mean_accuracy"],
                      marker="o", markersize=2, color=color,
                      linestyle=ls, linewidth=0.8, label=label)

    ax_b.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3, linewidth=0.5)
    ax_b.set_xscale("log")
    ax_b.set_ylim(-0.02, 0.55)
    ax_b.set_xlabel("Queries $m$")
    ax_b.set_ylabel("Mean error")
    ax_b.set_title(f"(b) Error vs $m$ ($n\\!=\\!{n_plot}$)")
    ax_b.legend(loc="upper right", fontsize=_leg_fs)

    # Panel (c): Error vs n for m=5
    m_plot = 5
    all_n = sorted(df_mds["n"].unique())

    for dist_name, ls, color, label in dist_config:
        sub = df_mds[(df_mds["distribution"] == dist_name) &
                     (df_mds["m"] == m_plot)].sort_values("n")
        if not sub.empty:
            ax_c.plot(sub["n"], 1 - sub["mean_accuracy"],
                      marker="o", markersize=2, color=color,
                      linestyle=ls, linewidth=0.8, label=label)

    ax_c.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3, linewidth=0.5)
    ax_c.set_xscale("log")
    ax_c.set_ylim(-0.02, 0.55)
    ax_c.set_xticks(all_n)
    ax_c.set_xticklabels([str(n) for n in all_n])
    ax_c.xaxis.set_minor_locator(plt.NullLocator())
    ax_c.set_xlabel("Models $n$")
    ax_c.set_ylabel("Mean error")
    ax_c.set_title(f"(c) Error vs $n$ ($m\\!=\\!{m_plot}$)")
    ax_c.legend(loc="upper right", fontsize=_leg_fs)

    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    suffix = "_BIG_FONT" if big_font else ""
    fig.savefig(f"{output_dir}/figure1_motivating{suffix}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 1 to {output_dir}/figure1_motivating{suffix}.pdf")


def plot_motivating_estimation_row(
    responses: np.ndarray,
    labels: np.ndarray,
    sensitive_indices: np.ndarray,
    orthogonal_indices: np.ndarray,
    fail_csv_path: str,
    output_dir: str = "figures",
    big_font: bool = False,
):
    """Standalone 1×3 figure: the motivating row from the 3×3.

    Scree plot, GMM on |α̂_{q,ℓ}|, P[err >= 0.5] with fitted curve.
    """
    from bbo.plotting.real_plots import (
        _plot_scree, _plot_gmm_single, _plot_failure_prob,
    )
    from matplotlib.legend import Legend

    set_paper_style()
    _bump = 2 if big_font else 0
    if big_font:
        plt.rcParams.update({
            "font.size": 7 + _bump,
            "axes.labelsize": 8 + _bump,
            "axes.titlesize": 8 + _bump,
            "xtick.labelsize": 6 + _bump,
            "ytick.labelsize": 6 + _bump,
            "legend.fontsize": 4.5 + _bump,
        })

    _h = 2.0 if big_font else 1.5
    fig, (ax_scree, ax_gmm, ax_fail) = plt.subplots(
        1, 3, figsize=(5.5, _h),
    )

    r_hat, U, s, rho_hats, gmm_info, n_signal = _plot_scree(
        ax_scree, responses, labels, sensitive_indices, orthogonal_indices,
    )
    ax_scree.set_title("Singular values of $E$")

    _plot_gmm_single(ax_gmm, gmm_info, r_hat, rho_hats, n_signal, direction=0)
    ax_gmm.set_title("GMM on $|\\hat{\\alpha}_{q,\\ell}|$")
    ax_gmm.set_ylabel("Density")

    _leg_fs = 3.5 + _bump
    leg_sc = [
        Line2D([0], [0], color=PALETTE[1], lw=4, alpha=0.6, label="Signal"),
        Line2D([0], [0], color=PALETTE[2], lw=4, alpha=0.6, label="Orthogonal"),
    ]
    # Remove the K/BIC legend placed by _plot_gmm_single to avoid overlap
    ax_gmm.get_legend().remove()
    leg2 = Legend(ax_gmm, leg_sc, ["Signal", "Orthogonal"],
                  loc="upper left", fontsize=_leg_fs)
    ax_gmm.add_artist(leg2)

    _plot_failure_prob(ax_fail, fail_csv_path, rho_hats=rho_hats,
                       query_set="uniform")
    ax_fail.set_title("Estimated & Fitted Errors")

    # Bump legend fontsizes in scree and failure_prob panels
    if big_font:
        for ax in [ax_scree, ax_fail]:
            leg = ax.get_legend()
            if leg:
                for t in leg.get_texts():
                    t.set_fontsize(_leg_fs)
        # Also bump annotation text sizes
        for txt in ax_scree.texts + ax_gmm.texts + ax_fail.texts:
            txt.set_fontsize(txt.get_fontsize() + _bump)

    fig.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    suffix = "_BIG_FONT" if big_font else ""
    out = f"{output_dir}/figure_motivating_estimation{suffix}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out}")


def plot_motivating_estimated_queries(
    responses: np.ndarray,
    labels: np.ndarray,
    sensitive_indices: np.ndarray,
    orthogonal_indices: np.ndarray,
    metadata_path: str,
    classification_csv: str,
    model_names: np.ndarray = None,
    output_dir: str = "figures",
    big_font: bool = False,
):
    """Motivating figure using ESTIMATED signal/orthogonal query sets.

    Same layout as plot_figure1_motivating but uses GMM-estimated partition
    from raw E instead of the true signal/orthogonal indices.

    Layout (GridSpec 2x3):
        gs[0, 0] = (a) MDS scatter — est. signal queries, m=5
        gs[1, 0] =     MDS scatter — est. orthogonal queries, m=5
        gs[:, 1] = (b) Error vs m for n=50 (est. signal, est. ortho, uniform)
        gs[:, 2] = (c) Error vs n for m=5 (est. signal, est. ortho, uniform)
    """
    from bbo.distances.energy import per_query_energy_tensor
    from bbo.estimation.rank_rho import estimate_discriminative_rank, estimate_rho

    set_paper_style()
    _bump = 2 if big_font else 0
    plt.rcParams.update({
        "font.size": 6 + _bump, "axes.labelsize": 7 + _bump, "axes.titlesize": 7 + _bump,
        "xtick.labelsize": 5 + _bump, "ytick.labelsize": 5 + _bump,
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

    # Estimate signal/orthogonal partition from raw E
    query_pool = np.concatenate([sensitive_indices, orthogonal_indices])
    E, pairs = per_query_energy_tensor(responses[:, query_pool, :])
    r_hat, U, s = estimate_discriminative_rank(E)
    _, gmm_info = estimate_rho(U, r_hat)

    # GMM partition: label=1 is active, label=0 is zero-set
    gmm_labels = gmm_info["per_direction"][0]["labels"]
    est_signal_pool = np.where(gmm_labels == 1)[0]   # indices into query_pool
    est_ortho_pool = np.where(gmm_labels == 0)[0]
    est_signal = query_pool[est_signal_pool]           # original query indices
    est_ortho = query_pool[est_ortho_pool]

    # Load classification CSV
    df = pd.read_csv(classification_csv)

    # --- Layout ---
    _h = 2.4 if big_font else 1.6
    fig = plt.figure(figsize=(5.5, _h))
    gs = GridSpec(2, 3, figure=fig, wspace=0.55 if not big_font else 0.75,
                  hspace=0.65 if not big_font else 0.85)

    ax_a_top = fig.add_subplot(gs[0, 0])
    ax_a_bot = fig.add_subplot(gs[1, 0])
    ax_b = fig.add_subplot(gs[:, 1])
    ax_c = fig.add_subplot(gs[:, 2])

    # Orange gradient colormap
    light_orange = (1.0, 0.85, 0.6)
    orange_cmap = LinearSegmentedColormap.from_list(
        "orange_grad", [light_orange, PALETTE[1]]
    )

    class0_mask = labels == 0
    class1_mask = labels == 1
    fracs_1 = sensitive_fracs[class1_mask]
    frac_norm = (fracs_1 - fracs_1.min()) / (fracs_1.max() - fracs_1.min() + 1e-12)
    colors_1 = orange_cmap(frac_norm)

    # --- Panel (a): MDS scatter at m=5 — est. signal vs est. orthogonal ---
    rng = np.random.RandomState(0)
    m_mds = 5
    sig_sub = rng.choice(est_signal, size=m_mds, replace=False)
    orth_sub = rng.choice(est_ortho, size=m_mds, replace=False)

    for ax, qi, title in [
        (ax_a_top, sig_sub, "(a) Est. signal queries"),
        (ax_a_bot, orth_sub, "Est. orthogonal queries"),
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

    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=PALETTE[0], markersize=4, label="Class 0"),
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=PALETTE[1], markersize=4, label="Class 1"),
    ]
    _leg_fs = 4 + _bump
    ax_a_top.legend(handles=legend_elements, loc="best", fontsize=_leg_fs)
    ax_a_bot.set_xlabel("MDS 1")

    # --- Panels (b) and (c) ---
    df_mds = df[df["method"] == "mds"]

    dist_config = [
        ("relevant",   "-",  PALETTE[0], "Est. signal"),
        ("orthogonal", "--", PALETTE[1], "Est. orthogonal"),
        ("uniform",    ":",  PALETTE[2], "Uniform"),
    ]

    # Panel (b): Error vs m for n=50
    n_plot = 50
    for dist_name, ls, color, label in dist_config:
        sub = df_mds[(df_mds["distribution"] == dist_name) &
                     (df_mds["n"] == n_plot) &
                     (df_mds["m"] >= 2)].sort_values("m")
        if not sub.empty:
            ax_b.plot(sub["m"], 1 - sub["mean_accuracy"],
                      marker="o", markersize=2, color=color,
                      linestyle=ls, linewidth=0.8, label=label)

    ax_b.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3, linewidth=0.5)
    ax_b.set_xscale("log")
    ax_b.set_ylim(-0.02, 0.55)
    ax_b.set_xlabel("Queries $m$")
    ax_b.set_ylabel("Mean error")
    ax_b.set_title(f"(b) Error vs $m$ ($n\\!=\\!{n_plot}$)")
    ax_b.legend(loc="upper right", fontsize=_leg_fs)

    # Panel (c): Error vs n for m=5
    m_plot = 5
    all_n = sorted(df_mds["n"].unique())

    for dist_name, ls, color, label in dist_config:
        sub = df_mds[(df_mds["distribution"] == dist_name) &
                     (df_mds["m"] == m_plot)].sort_values("n")
        if not sub.empty:
            ax_c.plot(sub["n"], 1 - sub["mean_accuracy"],
                      marker="o", markersize=2, color=color,
                      linestyle=ls, linewidth=0.8, label=label)

    ax_c.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3, linewidth=0.5)
    ax_c.set_xscale("log")
    ax_c.set_ylim(-0.02, 0.55)
    ax_c.set_xticks(all_n)
    ax_c.set_xticklabels([str(n) for n in all_n])
    ax_c.xaxis.set_minor_locator(plt.NullLocator())
    ax_c.set_xlabel("Models $n$")
    ax_c.set_ylabel("Mean error")
    ax_c.set_title(f"(c) Error vs $n$ ($m\\!=\\!{m_plot}$)")
    ax_c.legend(loc="upper right", fontsize=_leg_fs)

    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    suffix = "_BIG_FONT" if big_font else ""
    out = f"{output_dir}/figure1_motivating_estimated{suffix}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out}")
