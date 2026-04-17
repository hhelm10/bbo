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

    n_colors = {20: PALETTE[0], 80: PALETTE[1]}
    sel_config = [
        ("uniform",            "--", "o"),
        ("uniform_signal",     "-",  "s"),
        ("uniform_orthogonal", ":",  "v"),
        ("greedy",             "-",  "D"),
    ]
    sel_labels = {
        "uniform": "Uniform",
        "uniform_signal": "Est. signal",
        "uniform_orthogonal": "Est. orthogonal",
        "greedy": "Greedy",
    }

    for ax, (title, csv_path) in zip(axes, datasets):
        df = pd.read_csv(csv_path)

        for n_train in [20, 80]:
            color = n_colors[n_train]
            for sel_name, ls, marker in sel_config:
                sub = df[(df["selector"] == sel_name) &
                         (df["n_train"] == n_train)].sort_values("m")
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
    leg = [
        Line2D([0], [0], color=n_colors[20], lw=1.0, label="$n=20$"),
        Line2D([0], [0], color=n_colors[80], lw=1.0, label="$n=80$"),
    ]
    for sel_name, ls, marker in sel_config:
        leg.append(Line2D([0], [0], color="0.4", linestyle=ls, lw=0.8,
                          marker=marker, markersize=2,
                          label=sel_labels[sel_name]))
    axes[1].legend(handles=leg, loc="upper right", fontsize=3.5, ncol=2)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved pilot selection figure to {output_path}")
