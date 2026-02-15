"""Paper-quality matplotlib defaults."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def set_paper_style():
    """Set matplotlib/seaborn defaults for paper-quality figures."""
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("whitegrid")

    plt.rcParams.update({
        "figure.figsize": (6, 4),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "pdf.fonttype": 42,  # TrueType fonts in PDF
        "ps.fonttype": 42,
    })


# Color palette for consistent plotting
PALETTE = sns.color_palette("colorblind", 10)
