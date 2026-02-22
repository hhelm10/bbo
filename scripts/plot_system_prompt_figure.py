"""Plot three-panel figure for system prompt experiment.

Left:   MDS (relevant) vs baselines (concat, oracle best, MDS orthogonal)
Center: MDS across base models (only ministral-8b for now)
Right:  MDS across embedding models (only nomic for now)

Line styles: solid = n=80, dashed = n=10
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from pathlib import Path

from bbo.plotting.style import set_paper_style, PALETTE


def load_or_compute(npz_path, n_values, m_values, n_reps=200, seed=42):
    """Compute MDS/concat/oracle results for one (base_model, embed_model) pair."""
    from bbo.queries.query_set import sample_queries
    from bbo.queries.distributions import SubsetDistribution
    from bbo.classification.evaluate import make_classifier
    from bbo.distances.energy import pairwise_energy_distances_t0
    from bbo.embedding.mds import ClassicalMDS

    data = np.load(str(npz_path), allow_pickle=True)
    responses = data["responses"]
    labels = data["labels"]
    signal_idx = data["signal_indices"]

    # Combined orthogonal = weak_signal + null
    weak_idx = data["weak_signal_indices"] if "weak_signal_indices" in data else np.array([], dtype=np.int64)
    null_idx = data["null_indices"] if "null_indices" in data else data["orthogonal_indices"]
    ortho_combined = np.concatenate([weak_idx, null_idx]) if len(weak_idx) > 0 else null_idx

    n_models, M, p = responses.shape
    dist_rel = SubsetDistribution(signal_idx, mass=1.0)
    dist_ort = SubsetDistribution(ortho_combined, mass=1.0)

    rows = []

    for method_name, dist, use_mds in [
        ("mds_relevant", dist_rel, True),
        ("mds_orthogonal", dist_ort, True),
        ("concat", dist_rel, False),
    ]:
        for n in n_values:
            for m in m_values:
                accs = []
                for rep in range(n_reps):
                    rng = np.random.default_rng(
                        seed + rep * 100003 + m * 1009 + n * 31
                        + hash(method_name) % 10000
                    )
                    query_idx = sample_queries(M, m, distribution=dist, rng=rng)

                    if use_mds:
                        D = pairwise_energy_distances_t0(responses, query_idx)
                        mds = ClassicalMDS(n_components=min(10, n_models - 1))
                        X = mds.fit_transform(D)
                    else:
                        X = responses[:, query_idx, :].reshape(n_models, -1)

                    class0 = np.where(labels == 0)[0]
                    class1 = np.where(labels == 1)[0]
                    n_per = n // 2
                    sel0 = rng.choice(class0, n_per, replace=False)
                    sel1 = rng.choice(class1, n_per, replace=False)
                    train_idx = np.concatenate([sel0, sel1])
                    test_idx = np.setdiff1d(np.arange(n_models), train_idx)

                    clf = make_classifier("rf")
                    clf.fit(X[train_idx], labels[train_idx])
                    preds = clf.predict(X[test_idx])
                    accs.append((preds == labels[test_idx]).mean())

                rows.append({
                    "method": method_name, "n": n, "m": m,
                    "mean_acc": np.mean(accs), "std_acc": np.std(accs),
                    "max_acc": np.max(accs),
                })
                print(f"  {method_name} n={n} m={m}: {np.mean(accs):.1%}")

    return pd.DataFrame(rows)


def plot_figure(csv_path, embed_csv_path=None,
                output_path="figures/figure4_system_prompt.pdf"):
    """Build the three-panel figure from precomputed CSV.

    Parameters
    ----------
    csv_path : str
        CSV with method/n/m/mean_acc columns (panels a and b).
    embed_csv_path : str, optional
        CSV with embed_model/n/m/mean_acc columns (panel c).
        If None, falls back to using csv_path for panel c.
    """
    set_paper_style()

    df = pd.read_csv(csv_path)

    fig = plt.figure(figsize=(5.5, 2.0))
    gs = GridSpec(1, 3, figure=fig, left=0.08, right=0.99, bottom=0.20, top=0.85,
                  wspace=0.35)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    n_styles = {80: "-", 10: "--"}

    # ── Left panel: MDS vs baselines ──
    method_cfg = {
        "mds_relevant":   {"color": PALETTE[0], "label": "MDS (signal)"},
        "mds_orthogonal": {"color": PALETTE[2], "label": "MDS (orthogonal)"},
        "concat":         {"color": PALETTE[1], "label": "Concat"},
    }

    for method, cfg in method_cfg.items():
        for n, ls in n_styles.items():
            sub = df[(df["method"] == method) & (df["n"] == n)].sort_values("m")
            if sub.empty:
                continue
            ax_a.plot(sub["m"], 1 - sub["mean_acc"], marker="o", markersize=2,
                      color=cfg["color"], linestyle=ls, linewidth=0.8)

    ax_a.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4, linewidth=0.5)
    ax_a.set_xscale("log")
    ax_a.set_ylim(-0.02, 0.55)
    ax_a.set_xlabel("Number of queries $m$")
    ax_a.set_ylabel("Classification error")
    ax_a.set_title("(a) MDS vs. baselines")

    leg_methods = [Line2D([0], [0], color=cfg["color"], lw=1, label=cfg["label"])
                   for cfg in method_cfg.values()]
    leg_n = [Line2D([0], [0], color="0.5", linestyle=ls, lw=0.8, label=f"$n={n}$")
             for n, ls in n_styles.items()]
    ax_a.legend(handles=leg_methods + leg_n, loc="upper right", ncol=2, fontsize=4)

    # ── Center panel: across base models (only ministral-8b for now) ──
    base_models = ["ministral-8b"]  # extend later
    for i, bm in enumerate(base_models):
        color = PALETTE[i]
        for n, ls in n_styles.items():
            sub = df[(df["method"] == "mds_relevant") & (df["n"] == n)].sort_values("m")
            if sub.empty:
                continue
            ax_b.plot(sub["m"], 1 - sub["mean_acc"], marker="o", markersize=2,
                      color=color, linestyle=ls, linewidth=0.8)

    ax_b.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4, linewidth=0.5)
    ax_b.set_xscale("log")
    ax_b.set_ylim(-0.02, 0.55)
    ax_b.set_xlabel("Number of queries $m$")
    ax_b.set_yticklabels([])
    ax_b.set_title("(b) Across base LLMs")

    leg_bm = [Line2D([0], [0], color=PALETTE[i], lw=1, label=bm)
              for i, bm in enumerate(base_models)]
    leg_n2 = [Line2D([0], [0], color="0.5", linestyle=ls, lw=0.8, label=f"$n={n}$")
              for n, ls in n_styles.items()]
    ax_b.legend(handles=leg_bm + leg_n2, loc="upper right", fontsize=4)

    # ── Right panel: across embedding models ──
    embed_models = [
        ("nomic-embed-text-v1.5", "nomic"),
        ("text-embedding-3-small", "OAI-small"),
        ("text-embedding-3-large", "OAI-large"),
        ("gemini-embedding", "gemini"),
        ("all-MiniLM-L6-v2", "MiniLM"),
    ]

    if embed_csv_path is not None:
        df_embed = pd.read_csv(embed_csv_path)
    else:
        # Fallback: use mds_relevant from main CSV as nomic-only
        df_embed = df[df["method"] == "mds_relevant"].copy()
        df_embed["embed_model"] = "nomic-embed-text-v1.5"

    for i, (em_key, em_label) in enumerate(embed_models):
        color = PALETTE[i]
        sub_em = df_embed[df_embed["embed_model"] == em_key]
        if sub_em.empty:
            continue
        for n, ls in n_styles.items():
            sub = sub_em[sub_em["n"] == n].sort_values("m")
            if sub.empty:
                continue
            ax_c.plot(sub["m"], 1 - sub["mean_acc"], marker="o", markersize=2,
                      color=color, linestyle=ls, linewidth=0.8)

    ax_c.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4, linewidth=0.5)
    ax_c.set_xscale("log")
    ax_c.set_ylim(-0.02, 0.55)
    ax_c.set_xlabel("Number of queries $m$")
    ax_c.set_yticklabels([])
    ax_c.set_title("(c) Across embeddings")

    leg_em = [Line2D([0], [0], color=PALETTE[i], lw=1, label=em_label)
              for i, (_, em_label) in enumerate(embed_models)
              if not df_embed[df_embed["embed_model"] == embed_models[i][0]].empty]
    leg_n3 = [Line2D([0], [0], color="0.5", linestyle=ls, lw=0.8, label=f"$n={n}$")
              for n, ls in n_styles.items()]
    ax_c.legend(handles=leg_em + leg_n3, loc="upper right", fontsize=3.5,
                ncol=2, handlelength=1.2, columnspacing=0.5)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    fig.savefig(output_path.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    import sys

    npz = "results/system_prompt/embeddings/ministral-8b__nomic-embed-text-v1.5.npz"
    csv = "results/system_prompt/figure_data_v3.csv"
    embed_csv = "results/system_prompt/embed_panel_all.csv"
    out = "figures/figure4_system_prompt.pdf"

    if "--compute" in sys.argv or not Path(csv).exists():
        print("Computing results...")
        df = load_or_compute(npz, n_values=[10, 80], m_values=[1, 2, 5, 10, 20, 50])
        df.to_csv(csv, index=False)
        print(f"Saved to {csv}")

    print("Plotting...")
    embed_csv_path = embed_csv if Path(embed_csv).exists() else None
    plot_figure(csv, embed_csv_path=embed_csv_path, output_path=out)
