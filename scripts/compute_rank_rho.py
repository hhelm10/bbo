#!/usr/bin/env python
"""Compute discriminative rank r̂, zero-set probability ρ̂, and predicted m*
for all system prompt and motivating experiment NPZ files.

Uses the per-query between-class excess B_q with a 2-component GMM to
estimate ρ̂, and the singular value spectrum of E to estimate r̂.

Outputs a CSV with columns:
    experiment, base_model, embed_model, query_set, n_queries,
    r_hat, rho_hat, pi0, sv_ratio, mstar_80, mstar_90, mstar_95

Usage:
    python scripts/compute_rank_rho.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bbo.distances.energy import per_query_energy_tensor
from bbo.estimation.rank_rho import (
    compute_E_disc,
    estimate_discriminative_rank,
    estimate_rho,
    predict_mstar,
)


def process_npz(responses, labels, all_indices):
    """Compute r̂, ρ̂ from B_q (per-query between-class excess) via GMM.

    Parameters
    ----------
    responses : ndarray of shape (n_models, M, p)
    labels : ndarray of shape (n_models,)
    all_indices : ndarray of shape (M_total,)

    Returns
    -------
    dict with estimation results.
    """
    E, pairs = per_query_energy_tensor(responses[:, all_indices, :])

    # r̂ from scree of full E
    r_hat, U, s = estimate_discriminative_rank(E, n_elbows=1)

    # B_q from between-class centering
    _, _, B_q = compute_E_disc(E, pairs, labels)

    # ρ̂ from 2-component GMM on B_q
    rho_hat, info = estimate_rho(B_q)

    return {
        "n_queries": len(all_indices),
        "r_hat": r_hat,
        "rho_hat": rho_hat,
        "pi0": info["pi0"],
        "bic1": info["bic1"],
        "bic2": info["bic2"],
        "sv_ratio": float(s[0] / s[1]) if len(s) > 1 else np.inf,
        "mstar_80": predict_mstar(r_hat, rho_hat, epsilon=0.20),
        "mstar_90": predict_mstar(r_hat, rho_hat, epsilon=0.10),
        "mstar_95": predict_mstar(r_hat, rho_hat, epsilon=0.05),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute r̂, ρ̂, m* for all experiments"
    )
    args = parser.parse_args()

    rows = []

    # --- System prompt experiments ---
    sp_dir = Path("results/system_prompt/embeddings")

    if sp_dir.exists():
        for npz_path in sorted(sp_dir.glob("*.npz")):
            stem = npz_path.stem
            parts = stem.split("__")
            if len(parts) != 2:
                print(f"Skipping {npz_path.name} (unexpected naming)")
                continue
            base_model, embed_model = parts

            print(f"Processing system_prompt: {base_model} + {embed_model}")

            data = np.load(str(npz_path), allow_pickle=True)
            responses = data["responses"]
            labels = data["labels"]

            # Collect all query indices
            signal_idx = data["signal_indices"] if "signal_indices" in data else np.array([])
            null_idx = data["null_indices"] if "null_indices" in data else np.array([])
            if len(null_idx) == 0 and "orthogonal_indices" in data:
                null_idx = data["orthogonal_indices"]

            all_idx = np.concatenate([signal_idx, null_idx])
            if len(all_idx) == 0:
                print("  No queries found, skipping")
                continue

            res = process_npz(responses, labels, all_idx)

            rows.append({
                "experiment": "system_prompt",
                "base_model": base_model,
                "embed_model": embed_model,
                "query_set": "all",
                "n_queries": res["n_queries"],
                "r_hat": res["r_hat"],
                "rho_hat": round(res["rho_hat"], 4),
                "pi0": round(res["pi0"], 4),
                "sv_ratio": round(res["sv_ratio"], 2),
                "mstar_80": res["mstar_80"],
                "mstar_90": res["mstar_90"],
                "mstar_95": res["mstar_95"],
            })
            print(f"  all (M={res['n_queries']:3d}): "
                  f"r̂={res['r_hat']}, ρ̂={res['rho_hat']:.4f}, "
                  f"π₀={res['pi0']:.4f}, σ₁/σ₂={res['sv_ratio']:.2f}, "
                  f"m*={res['mstar_95']}")

    # --- Motivating experiment ---
    mot_npz = Path("results/motivating/motivating_responses.npz")

    if mot_npz.exists():
        print(f"\nProcessing motivating experiment")
        data = np.load(str(mot_npz), allow_pickle=True)
        responses = data["responses"]
        labels = data["labels"]

        signal_idx = data["sensitive_indices"] if "sensitive_indices" in data else np.array([])
        orth_idx = data["orthogonal_indices"] if "orthogonal_indices" in data else np.array([])

        all_idx = np.concatenate([signal_idx, orth_idx])
        if len(all_idx) > 0:
            res = process_npz(responses, labels, all_idx)

            rows.append({
                "experiment": "motivating",
                "base_model": "Qwen2.5-1.5B",
                "embed_model": "nomic-embed-text-v1.5",
                "query_set": "all",
                "n_queries": res["n_queries"],
                "r_hat": res["r_hat"],
                "rho_hat": round(res["rho_hat"], 4),
                "pi0": round(res["pi0"], 4),
                "sv_ratio": round(res["sv_ratio"], 2),
                "mstar_80": res["mstar_80"],
                "mstar_90": res["mstar_90"],
                "mstar_95": res["mstar_95"],
            })
            print(f"  all (M={res['n_queries']:3d}): "
                  f"r̂={res['r_hat']}, ρ̂={res['rho_hat']:.4f}, "
                  f"π₀={res['pi0']:.4f}, σ₁/σ₂={res['sv_ratio']:.2f}, "
                  f"m*={res['mstar_95']}")

    # --- Save results ---
    if rows:
        df = pd.DataFrame(rows)
        out_path = Path("results/system_prompt/rank_rho_estimates.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\nSaved to {out_path}")
        print(df.to_string(index=False))
    else:
        print("No NPZ files found.")


if __name__ == "__main__":
    main()
