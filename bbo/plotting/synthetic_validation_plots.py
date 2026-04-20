"""Figure 2: Synthetic validation (2 rows × 4+3 panels).

Top row (Theorem 1): P[err >= 0.5] vs m
  (a) Varying r, (b) Varying ρ, (c) Varying n, (d) Query distribution

Bottom row (Estimation):
  (e) P[r̂ = r] vs m, (f) ρ̂ vs m, (g) Empirical vs predicted bound
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
from pathlib import Path

from bbo.plotting.style import set_paper_style, PALETTE
from bbo.plotting.synthetic_plots import (
    _zero_pos, _map_zeros, _smooth_floor, _setup_broken_log_y, _theory_bound,
)


def plot_synthetic_validation(
    df_exp1, df_exp2, df_panel_c, df_exp3,
    df_panel_e, df_panel_f, df_panel_g,
    output_path="figures/figure_synthetic_validation.pdf",
):
    """2-row synthetic validation figure."""
    set_paper_style()

    fig = plt.figure(figsize=(5.5, 3.2))
    gs = GridSpec(2, 12, figure=fig,
                  left=0.07, right=0.99, bottom=0.09, top=0.92,
                  wspace=0.4, hspace=0.72)

    # Row 1: 4 panels × 3 cols each
    ax_a = fig.add_subplot(gs[0, 0:3])
    ax_b = fig.add_subplot(gs[0, 3:6])
    ax_c = fig.add_subplot(gs[0, 6:9])
    ax_d = fig.add_subplot(gs[0, 9:12])

    # Row 2: 3 panels × 4 cols each
    ax_e = fig.add_subplot(gs[1, 0:4])
    ax_f = fig.add_subplot(gs[1, 4:8])
    ax_g = fig.add_subplot(gs[1, 8:12])

    n_reps = int(df_exp1["n_reps"].iloc[0])
    m_dense = np.logspace(np.log10(1), np.log10(100), 200)

    theory_handle = mlines.Line2D([], [], color="gray", linestyle="--",
                                  linewidth=1.0, alpha=0.7, label="$r\\rho^m$")

    # ===== Panel (a): Varying r =====
    r_values = sorted(df_exp1["r"].unique())
    for i, r in enumerate(r_values):
        color = PALETTE[i % len(PALETTE)]
        sub = df_exp1[df_exp1["r"] == r]
        rho = sub["rho"].iloc[0]
        y = _map_zeros(sub["prob_high_error"].values, n_reps)
        ax_a.plot(sub["m"], y, marker="o", color=color, label=f"$r = {r}$")
        bound = _theory_bound(m_dense, r, rho)
        ax_a.plot(m_dense, _smooth_floor(bound, n_reps), color=color,
                  linestyle="--", linewidth=1.0, alpha=0.7)

    h, l = ax_a.get_legend_handles_labels()
    h.append(theory_handle); l.append("$r\\rho^m$")
    ax_a.set_xscale("log")
    _setup_broken_log_y(ax_a, n_reps)
    ax_a.set_xlabel("Queries $m$")
    ax_a.set_ylabel("$P[\\mathrm{error} \\geq 0.5]$")
    ax_a.set_title("(a) Varying rank $r$\n$n\\!=\\!100,\\; \\rho\\!\\approx\\!0.7$",
                   fontsize=7)
    ax_a.legend(h, l, loc="lower left", bbox_to_anchor=(0.02, 0), ncol=2)

    # ===== Panel (b): Varying ρ =====
    rho_values = sorted(df_exp2["rho"].unique())
    for i, rho in enumerate(rho_values):
        color = PALETTE[i % len(PALETTE)]
        sub = df_exp2[df_exp2["rho"] == rho]
        y = _map_zeros(sub["prob_high_error"].values, n_reps)
        ax_b.plot(sub["m"], y, marker="o", color=color,
                  label=f"$\\rho = {rho:.1f}$")
        bound = _theory_bound(m_dense, 5, rho)
        ax_b.plot(m_dense, _smooth_floor(bound, n_reps), color=color,
                  linestyle="--", linewidth=1.0, alpha=0.7)

    h2, l2 = ax_b.get_legend_handles_labels()
    h2.append(theory_handle); l2.append("$r\\rho^m$")
    ax_b.set_xscale("log")
    _setup_broken_log_y(ax_b, n_reps)
    ax_b.set_yticklabels([])
    ax_b.set_xlabel("Queries $m$")
    ax_b.set_title("(b) Varying $\\rho$\n$n\\!=\\!100,\\; r\\!=\\!5$", fontsize=7)
    ax_b.legend(h2, l2, loc="lower left", bbox_to_anchor=(0.02, 0))

    # ===== Panel (c): Varying n =====
    n_values_c = sorted(df_panel_c["n_models"].unique())
    rho_c = df_panel_c["rho"].iloc[0]
    r_c = int(df_panel_c["r"].iloc[0])

    for i, n_val in enumerate(n_values_c):
        color = PALETTE[i % len(PALETTE)]
        sub = df_panel_c[df_panel_c["n_models"] == n_val].sort_values("m")
        y = _map_zeros(sub["prob_high_error"].values, n_reps)
        ax_c.plot(sub["m"], y, marker="o", color=color,
                  label=f"$n = {n_val}$")

    # Theory bound (same for all n)
    bound_c = _theory_bound(m_dense, r_c, rho_c)
    ax_c.plot(m_dense, _smooth_floor(bound_c, n_reps), color="gray",
              linestyle="--", linewidth=1.0, alpha=0.7)

    hc, lc = ax_c.get_legend_handles_labels()
    hc.append(theory_handle); lc.append("$r\\rho^m$")
    ax_c.set_xscale("log")
    _setup_broken_log_y(ax_c, n_reps)
    ax_c.set_yticklabels([])
    ax_c.set_xlabel("Queries $m$")
    ax_c.set_title(f"(c) Varying $n$\n$r\\!=\\!{r_c},\\; \\rho\\!=\\!{rho_c}$",
                   fontsize=7)
    ax_c.legend(hc, lc, loc="lower left", bbox_to_anchor=(0.02, 0))

    # ===== Panel (d): Query distribution =====
    df_d = df_exp3[df_exp3["rho"] == 0.7] if "rho" in df_exp3.columns else df_exp3
    dist_labels = {"signal": "Signal", "uniform": "Uniform",
                   "orthogonal": "Orthogonal"}
    dist_colors = {"signal": PALETTE[0], "uniform": PALETTE[1],
                   "orthogonal": PALETTE[2]}

    for dist_name, label in dist_labels.items():
        sub = df_d[df_d["distribution"] == dist_name].sort_values("m")
        if sub.empty:
            continue
        y = _map_zeros(sub["prob_high_error"].values, n_reps)
        ax_d.plot(sub["m"], y, marker="o", color=dist_colors[dist_name],
                  label=label)

    ax_d.set_xscale("log")
    _setup_broken_log_y(ax_d, n_reps)
    ax_d.set_yticklabels([])
    ax_d.set_xlabel("Queries $m$")
    ax_d.set_title("(d) Query distribution\n$n\\!=\\!100,\\; r\\!=\\!5,\\; \\rho\\!=\\!0.7$",
                   fontsize=7)
    ax_d.legend(loc="lower left", bbox_to_anchor=(0.02, 0))

    # ===== Panel (e): Rank recovery =====
    r_values_e = sorted(df_panel_e["r"].unique())
    for i, r in enumerate(r_values_e):
        color = PALETTE[i % len(PALETTE)]
        sub = df_panel_e[df_panel_e["r"] == r].sort_values("m")
        ax_e.plot(sub["m"], sub["p_correct_rank"], marker="o", color=color,
                  label=f"$r = {r}$")

    ax_e.set_xscale("log")
    ax_e.set_ylim(-0.02, 1.05)
    ax_e.set_xlabel("Queries $m$")
    ax_e.set_ylabel("$P[\\hat{r} = r]$")
    ax_e.set_title("(e) Rank recovery\n$\\rho\\!=\\!0.7,\\; n\\!=\\!100$",
                   fontsize=7)
    ax_e.legend(loc="lower right", fontsize=5)

    # ===== Panel (f): ρ̂ recovery =====
    rho_values_f = sorted(df_panel_f["rho_true"].unique())
    for i, rho in enumerate(rho_values_f):
        color = PALETTE[i % len(PALETTE)]
        sub = df_panel_f[df_panel_f["rho_true"] == rho].sort_values("m")
        ax_f.plot(sub["m"], sub["rho_hat_mean"], marker="o", color=color,
                  label=f"$\\rho = {rho}$")
        ax_f.axhline(y=rho, color=color, linestyle="--", linewidth=0.8,
                     alpha=0.5)

    ax_f.set_xscale("log")
    ax_f.set_ylim(-0.02, 1.05)
    ax_f.set_yticklabels([])
    ax_f.set_xlabel("Queries $m$")
    ax_f.set_title("(f) $\\hat{\\rho}$ recovery\n$r\\!=\\!5,\\; n\\!=\\!100$",
                   fontsize=7)
    ax_f.legend(loc="lower right", fontsize=5)

    # ===== Panel (g): Predicted vs empirical =====
    rho_values_g = sorted(df_panel_g["rho_true"].unique())
    for i, rho in enumerate(rho_values_g):
        color = PALETTE[i % len(PALETTE)]
        sub = df_panel_g[df_panel_g["rho_true"] == rho].sort_values("m")

        # Empirical (solid)
        y_emp = _map_zeros(sub["prob_high_error"].values, n_reps)
        ax_g.plot(sub["m"], y_emp, marker="o", color=color,
                  linestyle="-", linewidth=0.8)

        # Predicted from estimated params (dotted)
        y_pred = sub["predicted_bound_mean"].values
        ax_g.plot(sub["m"], _smooth_floor(y_pred, n_reps), color=color,
                  linestyle=":", linewidth=1.0, alpha=0.8)

        # True bound (dashed)
        y_true = sub["true_bound"].values
        ax_g.plot(sub["m"], _smooth_floor(y_true, n_reps), color=color,
                  linestyle="--", linewidth=1.0, alpha=0.7)

    ax_g.set_xscale("log")
    _setup_broken_log_y(ax_g, n_reps)
    ax_g.set_yticklabels([])
    ax_g.set_xlabel("Queries $m$")
    ax_g.set_title("(g) Predicted vs empirical\n$r\\!=\\!5,\\; n\\!=\\!100$",
                   fontsize=7)

    leg_g = [
        mlines.Line2D([], [], color="0.4", linestyle="-", marker="o",
                      markersize=2, lw=0.8, label="Empirical"),
        mlines.Line2D([], [], color="0.4", linestyle=":", lw=1.0,
                      label="$\\hat{r}\\hat{\\rho}^m$"),
        mlines.Line2D([], [], color="0.4", linestyle="--", lw=1.0,
                      label="$r\\rho^m$"),
    ]
    ax_g.legend(handles=leg_g, loc="lower left", bbox_to_anchor=(0.02, 0),
                fontsize=5)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved to {output_path}")
