"""Paper-quality matplotlib defaults."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def set_paper_style():
    """Set matplotlib/seaborn defaults for camera-ready figures.

    Sized for ICLR/NeurIPS: textwidth ≈ 5.5 in.
    All font sizes are set so that they remain legible at 1:1 scale
    when included via \\includegraphics[width=\\textwidth].
    """
    sns.set_context("paper", font_scale=1.0)
    sns.set_style("whitegrid")

    plt.rcParams.update({
        "figure.figsize": (5.5, 2.0),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "text.usetex": False,
        "font.size": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 5,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "axes.linewidth": 0.5,
        "grid.linewidth": 0.3,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 0.8,
        "legend.borderpad": 0.3,
        "legend.labelspacing": 0.2,
        "legend.framealpha": 0.6,
        "pdf.fonttype": 42,  # TrueType fonts in PDF
        "ps.fonttype": 42,
    })


# Color palette for consistent plotting
PALETTE = sns.color_palette("colorblind", 10)
