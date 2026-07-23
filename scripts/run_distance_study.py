#!/usr/bin/env python
"""Distance measure (delta) study: effect on estimated signal/orthogonal sets.

For each dataset and each per-query dissimilarity metric, runs the full
estimation pipeline (between-class centering -> SVD -> GMM on loadings)
and measures how well the estimated signal/orthogonal split recovers the
ground-truth query sets.

Outputs:
    results/distance_study/summary.csv
    results/distance_study/{dataset}_loadings.npz  (per-metric loadings + GMM labels)
"""

import numpy as np
import pandas as pd
from pathlib import Path

from bbo.distances.energy import per_query_dissimilarity_tensor
from bbo.estimation.rank_rho import (
    compute_E_disc, estimate_discriminative_rank, estimate_rho, predict_mstar,
)

DATASETS = {
    "motivating": ("results/motivating/motivating_responses.npz",
                   "sensitive_indices", "orthogonal_indices"),
    "system_prompt": ("results/system_prompt/embeddings/mistral-small__nomic-embed-text-v1.5.npz",
                      "signal_indices", "orthogonal_indices"),
    "rag": ("results/rag/embeddings/ministral-8b__nomic-embed-text-v1.5.npz",
            "signal_indices", "control_indices"),
}

METRICS = ["sq_euclidean", "euclidean", "cosine", "l1", "rbf"]


def main():
    out_dir = Path("results/distance_study")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, (npz_path, sig_key, orth_key) in DATASETS.items():
        data = np.load(npz_path, allow_pickle=True)
        responses = data["responses"]
        labels = data["labels"]
        sig_idx = data[sig_key]
        orth_idx = data[orth_key]
        all_idx = np.concatenate([sig_idx, orth_idx])
        n_signal = len(sig_idx)
        true_signal = np.zeros(len(all_idx), dtype=int)
        true_signal[:n_signal] = 1

        print(f"\n{'='*60}")
        print(f"Dataset: {name} (M={len(all_idx)}, signal={n_signal})")
        print(f"{'='*60}")

        loadings_store = {}
        for metric in METRICS:
            E, pairs = per_query_dissimilarity_tensor(
                responses[:, all_idx, :], metric=metric)
            E_disc, _, B_q = compute_E_disc(E, pairs, labels)
            r_hat, U, s = estimate_discriminative_rank(E_disc)
            rho_hats, info = estimate_rho(U, r_hat)

            # Estimated signal set: active GMM component on direction 1
            est_signal = info["per_direction"][0]["labels"]

            tp = int(np.sum((est_signal == 1) & (true_signal == 1)))
            fp = int(np.sum((est_signal == 1) & (true_signal == 0)))
            fn = int(np.sum((est_signal == 0) & (true_signal == 1)))
            tn = int(np.sum((est_signal == 0) & (true_signal == 0)))
            prec = tp / (tp + fp) if tp + fp > 0 else 0.0
            rec = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
            bal_acc = 0.5 * (rec + (tn / (tn + fp) if tn + fp > 0 else 0.0))

            sv_ratio = float(s[0] / s[1]) if len(s) > 1 else np.inf
            rho_str = ", ".join(f"{r:.3f}" for r in rho_hats)

            rows.append({
                "dataset": name,
                "metric": metric,
                "r_hat": r_hat,
                "rho_1": rho_hats[0],
                "sv_ratio": sv_ratio,
                "mstar_95": predict_mstar(rho_hats, epsilon=0.05),
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "balanced_accuracy": bal_acc,
            })
            print(f"  {metric:14s}: r_hat={r_hat}, rho=[{rho_str}], "
                  f"sv_ratio={sv_ratio:6.2f}, "
                  f"prec={prec:.3f}, rec={rec:.3f}, bal_acc={bal_acc:.3f}")

            loadings_store[f"{metric}_loadings"] = info["per_direction"][0]["loadings"]
            loadings_store[f"{metric}_est_signal"] = est_signal

        loadings_store["true_signal"] = true_signal
        np.savez(out_dir / f"{name}_loadings.npz", **loadings_store)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary.csv", index=False)
    print(f"\nSaved to {out_dir}/summary.csv")


if __name__ == "__main__":
    main()
