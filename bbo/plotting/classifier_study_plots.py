"""Appendix figure for the classifier robustness study.

1x3 panels (datasets), accuracy vs m curves for four downstream classifiers
applied to the same distance -> MDS embeddings.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from bbo.plotting.style import set_paper_style, PALETTE

CLF_LABELS = {
    "rf": ("Random Forest", PALETTE[0], "-", "o"),
    "1nn": ("1-NN", PALETTE[1], "--", "s"),
    "linear_svm": ("Linear SVM", PALETTE[2], "-.", "^"),
    "rbf_svm": ("RBF SVM", PALETTE[3], ":", "D"),
}

DATASET_TITLES = {"motivating": "LoRA", "system_prompt": "System Prompt",
                  "rag": "RAG"}


def plot_classifier_study(
    results_dir: str = "results/classifier_study",
    output_path: str = "figures/figure_classifier_study.pdf",
):
    set_paper_style()

    df = pd.read_csv(Path(results_dir) / "summary.csv")

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.8), sharey=True)

    for col, (ds, title) in enumerate(DATASET_TITLES.items()):
        ax = axes[col]
        sub = df[df["dataset"] == ds]
        for clf, (label, color, ls, marker) in CLF_LABELS.items():
            g = sub[sub["classifier"] == clf].groupby("m")["accuracy"]
            mean = g.mean()
            se = g.std() / (g.count() ** 0.5)
            ax.plot(mean.index, mean.values, color=color, ls=ls,
                    marker=marker, ms=3, lw=1.0, label=label)
            ax.fill_between(mean.index, mean - 2 * se, mean + 2 * se,
                            color=color, alpha=0.15, lw=0)

        ax.set_xscale("log")
        ax.axhline(0.5, color="gray", ls=":", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("Number of queries $m$")
        if col == 0:
            ax.set_ylabel("Accuracy")

    axes[0].legend(loc="lower right", fontsize=5)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved classifier study figure to {output_path}")


if __name__ == "__main__":
    plot_classifier_study()
