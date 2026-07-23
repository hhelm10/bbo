#!/usr/bin/env python
"""Embedding model study: effect on estimated signal/orthogonal sets.

Runs the estimation pipeline (between-class centering -> SVD -> GMM on
loadings) for each dataset x embedding model, measuring recovery of the
ground-truth signal/orthogonal query sets.

Outputs:
    results/embedding_study/summary.csv
    results/embedding_study/{dataset}_loadings.npz
"""

import numpy as np
import pandas as pd
from pathlib import Path

from bbo.distances.energy import per_query_dissimilarity_tensor
from bbo.estimation.rank_rho import (
    compute_E_disc, estimate_discriminative_rank, estimate_rho, predict_mstar,
)

EMBED_MODELS = ["nomic-embed-text-v1.5", "all-MiniLM-L6-v2",
                "text-embedding-3-small"]

NPZ_PATHS = {
    ("motivating", "nomic-embed-text-v1.5"): "results/motivating/motivating_responses.npz",
    ("motivating", "all-MiniLM-L6-v2"): "results/motivating/embeddings/all-MiniLM-L6-v2.npz",
    ("motivating", "text-embedding-3-small"): "results/motivating/embeddings/text-embedding-3-small.npz",
    ("system_prompt", "nomic-embed-text-v1.5"): "results/system_prompt/embeddings/ministral-8b__nomic-embed-text-v1.5.npz",
    ("system_prompt", "all-MiniLM-L6-v2"): "results/system_prompt/embeddings/ministral-8b__all-MiniLM-L6-v2.npz",
    ("system_prompt", "text-embedding-3-small"): "results/system_prompt/embeddings/ministral-8b__text-embedding-3-small.npz",
    ("rag", "nomic-embed-text-v1.5"): "results/rag/embeddings/ministral-8b__nomic-embed-text-v1.5.npz",
    ("rag", "all-MiniLM-L6-v2"): "results/rag/embeddings/ministral-8b__all-MiniLM-L6-v2.npz",
    ("rag", "text-embedding-3-small"): "results/rag/embeddings/ministral-8b__text-embedding-3-small.npz",
}

POOL_KEYS = {
    "motivating": ("sensitive_indices", "orthogonal_indices"),
    "system_prompt": ("signal_indices", "orthogonal_indices"),
    "rag": ("signal_indices", "control_indices"),
}


def main():
    out_dir = Path("results/embedding_study")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ds, (sig_key, orth_key) in POOL_KEYS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds}")
        print(f"{'='*60}")

        loadings_store = {}
        for em in EMBED_MODELS:
            npz_path = NPZ_PATHS[(ds, em)]
            data = np.load(npz_path, allow_pickle=True)
            responses = data["responses"]
            labels = data["labels"]
            sig_idx = data[sig_key]
            orth_idx = data[orth_key]
            all_idx = np.concatenate([sig_idx, orth_idx])
            n_signal = len(sig_idx)
            true_signal = np.zeros(len(all_idx), dtype=int)
            true_signal[:n_signal] = 1

            E, pairs = per_query_dissimilarity_tensor(
                responses[:, all_idx, :], metric="sq_euclidean")
            E_disc, _, B_q = compute_E_disc(E, pairs, labels)
            r_hat, U, s = estimate_discriminative_rank(E_disc)
            rho_hats, info = estimate_rho(U, r_hat)

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
                "dataset": ds,
                "embed_model": em,
                "r_hat": r_hat,
                "rho_1": rho_hats[0],
                "sv_ratio": sv_ratio,
                "mstar_95": predict_mstar(rho_hats, epsilon=0.05),
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "balanced_accuracy": bal_acc,
            })
            print(f"  {em:24s}: r_hat={r_hat}, rho=[{rho_str}], "
                  f"sv_ratio={sv_ratio:6.2f}, "
                  f"prec={prec:.3f}, rec={rec:.3f}, bal_acc={bal_acc:.3f}")

            loadings_store[f"{em}_loadings"] = info["per_direction"][0]["loadings"]
            loadings_store[f"{em}_est_signal"] = est_signal
            loadings_store[f"{em}_true_signal"] = true_signal

        np.savez(out_dir / f"{ds}_loadings.npz", **loadings_store)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary.csv", index=False)
    print(f"\nSaved to {out_dir}/summary.csv")


if __name__ == "__main__":
    main()
