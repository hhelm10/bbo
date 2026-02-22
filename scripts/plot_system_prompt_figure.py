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


def plot_figure(csv_path, embed_csv_path=None, oracle_csv_path=None,
                base_csv_path=None,
                output_path="figures/figure4_system_prompt.pdf"):
    """Build the three-panel figure from precomputed CSV.

    Parameters
    ----------
    csv_path : str
        CSV with method/n/m/mean_acc columns (panel a).
    embed_csv_path : str, optional
        CSV with embed_model/n/m/mean_acc columns (panel c).
    oracle_csv_path : str, optional
        CSV with n/m/oracle_acc columns (oracle line in panel a).
    base_csv_path : str, optional
        CSV with base_model/n/m/mean_acc columns (panel b).
    """
    set_paper_style()

    df = pd.read_csv(csv_path)

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(5.5, 1.6))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.22, top=0.82, wspace=0.08)

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

    # Oracle line
    if oracle_csv_path and Path(oracle_csv_path).exists():
        df_oracle = pd.read_csv(oracle_csv_path)
        for n, ls in n_styles.items():
            sub = df_oracle[df_oracle["n"] == n].sort_values("m")
            if not sub.empty:
                ax_a.plot(sub["m"], 1 - sub["oracle_acc"], marker="s", markersize=2,
                          color=PALETTE[4], linestyle=ls, linewidth=0.8)

    ax_a.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4, linewidth=0.5)
    ax_a.set_xscale("log")
    ax_a.set_ylim(-0.02, 0.55)
    ax_a.set_xlabel("Number of queries $m$")
    ax_a.set_ylabel("Mean error")
    ax_a.set_title("(a) MDS vs. baselines")

    leg_methods = [Line2D([0], [0], color=cfg["color"], lw=0, marker="o",
                          markersize=3, label=cfg["label"])
                   for cfg in method_cfg.values()]
    if oracle_csv_path and Path(oracle_csv_path).exists():
        leg_methods.append(Line2D([0], [0], color=PALETTE[4], lw=0, marker="s",
                                  markersize=3, label="Best"))
    leg_n = [Line2D([0], [0], color="0.5", linestyle="--", lw=0.8, label="$n=10$"),
             Line2D([0], [0], color="0.5", linestyle="-", lw=0.8, label="$n=80$")]
    ax_a.legend(handles=leg_methods + leg_n, loc="upper right", ncol=2)

    # ── Center panel: across base models ──
    base_models = [
        ("ministral-8b", "ministral-8b"),
        ("ministral-3b", "ministral-3b"),
        ("mistral-small", "mistral-small"),
        ("mistral-large", "mistral-large"),
        ("gpt-4o-mini", "GPT-4o-mini"),
    ]

    if base_csv_path is not None:
        df_base = pd.read_csv(base_csv_path)
        # Cap at m=50 to match panels (a) and (c)
        df_base = df_base[df_base["m"] <= 50]
    else:
        # Fallback: use mds_relevant from main CSV as ministral-8b only
        df_base = df[df["method"] == "mds_relevant"].copy()
        df_base["base_model"] = "ministral-8b"

    for i, (bm_key, bm_label) in enumerate(base_models):
        color = PALETTE[i]
        sub_bm = df_base[df_base["base_model"] == bm_key]
        if sub_bm.empty:
            continue
        for n, ls in n_styles.items():
            sub = sub_bm[sub_bm["n"] == n].sort_values("m")
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

    leg_bm = [Line2D([0], [0], color=PALETTE[i], lw=0, marker="o",
                     markersize=3, label=bm_label)
              for i, (bm_key, bm_label) in enumerate(base_models)
              if not df_base[df_base["base_model"] == bm_key].empty]
    leg_n2 = [Line2D([0], [0], color="0.5", linestyle="--", lw=0.8, label="$n=10$"),
              Line2D([0], [0], color="0.5", linestyle="-", lw=0.8, label="$n=80$")]
    ax_b.legend(handles=leg_bm + leg_n2, loc="upper right", ncol=2)

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

    leg_em = [Line2D([0], [0], color=PALETTE[i], lw=0, marker="o",
                     markersize=3, label=em_label)
              for i, (_, em_label) in enumerate(embed_models)
              if not df_embed[df_embed["embed_model"] == embed_models[i][0]].empty]
    leg_n3 = [Line2D([0], [0], color="0.5", linestyle="--", lw=0.8, label="$n=10$"),
              Line2D([0], [0], color="0.5", linestyle="-", lw=0.8, label="$n=80$")]
    ax_c.legend(handles=leg_em + leg_n3, loc="upper right", ncol=2)

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
    base_csv = "results/system_prompt/base_panel_all.csv"
    out = "figures/figure4_system_prompt.pdf"

    if "--compute" in sys.argv or not Path(csv).exists():
        print("Computing results...")
        df = load_or_compute(npz, n_values=[10, 80], m_values=[1, 2, 5, 10, 20, 50])
        df.to_csv(csv, index=False)
        print(f"Saved to {csv}")

    oracle_csv = "results/system_prompt/oracle_data.csv"

    print("Plotting...")
    plot_figure(csv,
                embed_csv_path=embed_csv if Path(embed_csv).exists() else None,
                oracle_csv_path=oracle_csv if Path(oracle_csv).exists() else None,
                base_csv_path=base_csv if Path(base_csv).exists() else None,
                output_path=out)
