"""1×3 figure for the pilot query selection experiment.

One panel per dataset, each showing error vs m for
{uniform, uniform_signal, uniform_orthogonal, greedy} × {n_train=20, n_train=80}.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

from bbo.plotting.style import set_paper_style, PALETTE


def plot_pilot_selection(
    motivating_csv: str,
    system_prompt_csv: str,
    rag_csv: str,
    output_path: str = "figures/figure_pilot_selection.pdf",
):
    """1×3 figure: pilot query selection results."""
    set_paper_style()

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.6), sharey=True)

    datasets = [
        ("Motivating", motivating_csv),
        ("System Prompt", system_prompt_csv),
        ("RAG", rag_csv),
    ]

    sel_config = [
        ("uniform",            PALETTE[2], "-",  "s", "Uniform"),
        ("uniform_signal",     PALETTE[0], "-",  "o", "Est. signal"),
        ("stepwise",           PALETTE[3], "-",  "^", "Stepwise"),
    ]

    n_plot = 80

    for ax, (title, csv_path) in zip(axes, datasets):
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
        ax.set_xlabel("Queries $m$")
        ax.set_title(title)

    axes[0].set_ylabel("Mean error")

    # Shared legend
    leg = []
    for sel_name, color, ls, marker, label in sel_config:
        leg.append(Line2D([0], [0], color=color, linestyle=ls, lw=0.8,
                          marker=marker, markersize=2, label=label))
    axes[1].legend(handles=leg, loc="upper right", fontsize=4, ncol=2)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved pilot selection figure to {output_path}")
