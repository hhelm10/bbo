"""Figure for the embedding model (g) study.

Grid of loading histograms: rows = embedding models, cols = datasets.
Each panel shows |loadings| colored by true signal/orthogonal membership,
annotated with rho_hat and balanced accuracy of the estimated signal set.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from bbo.plotting.style import set_paper_style, PALETTE

EMBED_LABELS = {
    "nomic-embed-text-v1.5": "nomic-embed-\ntext-v1.5",
    "all-MiniLM-L6-v2": "all-MiniLM-\nL6-v2",
    "text-embedding-3-small": "text-embedding-\n3-small",
}

DATASET_TITLES = {"motivating": "LoRA", "system_prompt": "System Prompt",
                  "rag": "RAG"}


def plot_embedding_study(
    results_dir: str = "results/embedding_study",
    output_path: str = "figures/figure_embedding_study.pdf",
):
    set_paper_style()

    summary = pd.read_csv(Path(results_dir) / "summary.csv")
    embed_models = list(EMBED_LABELS.keys())
    datasets = list(DATASET_TITLES.keys())

    fig, axes = plt.subplots(len(embed_models), len(datasets),
                             figsize=(5.5, 1.25 * len(embed_models)))

    for col, ds in enumerate(datasets):
        npz = np.load(Path(results_dir) / f"{ds}_loadings.npz")

        for row, em in enumerate(embed_models):
            ax = axes[row, col]
            loadings = npz[f"{em}_loadings"]
            true_signal = npz[f"{em}_true_signal"]

            L_sig = loadings[true_signal == 1]
            L_orth = loadings[true_signal == 0]
            bins = np.linspace(loadings.min(), loadings.max(), 28)
            ax.hist(L_sig, bins=bins, alpha=0.6, color=PALETTE[0],
                    density=True, edgecolor="none", label="Signal")
            ax.hist(L_orth, bins=bins, alpha=0.6, color=PALETTE[1],
                    density=True, edgecolor="none", label="Orthogonal")

            sub = summary[(summary["dataset"] == ds)
                          & (summary["embed_model"] == em)].iloc[0]
            ax.text(0.97, 0.90,
                    f"$\\hat{{r}}={int(sub['r_hat'])}$, "
                    f"$\\hat{{\\rho}}_1={sub['rho_1']:.2f}$\n"
                    f"bal. acc. $={sub['balanced_accuracy']:.2f}$",
                    transform=ax.transAxes, fontsize=4.5,
                    ha="right", va="top")

            ax.set_yticks([])
            if row == 0:
                ax.set_title(DATASET_TITLES[ds])
            if row == len(embed_models) - 1:
                ax.set_xlabel("$|\\hat{\\alpha}_1(q)|$")
            if col == 0:
                ax.set_ylabel(EMBED_LABELS[em], fontsize=6)

    axes[0, 0].legend(loc="upper left", fontsize=4.5)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved embedding study figure to {output_path}")


if __name__ == "__main__":
    plot_embedding_study()
