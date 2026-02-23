#!/usr/bin/env python
"""Compute discriminative rank r̂, zero-set probability ρ̂, and predicted m*
for all system prompt and motivating experiment NPZ files.

Computes estimates for each query subset (signal, orthogonal, weak signal,
null/factual) separately.

Outputs a CSV with columns:
    experiment, base_model, embed_model, query_set, n_queries,
    r_hat, rho_hat, sv_ratio, mstar_80, mstar_90, mstar_95

Usage:
    python scripts/compute_rank_rho.py
    python scripts/compute_rank_rho.py --tau 0.01
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bbo.distances.energy import per_query_energy_tensor
from bbo.estimation.rank_rho import (
    estimate_discriminative_rank,
    estimate_rho,
    predict_mstar,
)


def process_query_set(responses, query_indices, tau=0.01):
    """Compute r̂, ρ̂, m* for one query subset.

    Parameters
    ----------
    responses : ndarray of shape (n_models, M, p)
    query_indices : ndarray of shape (m,)
    tau : float
        Threshold for zero-set probability estimation.

    Returns
    -------
    dict with estimation results.
    """
    E, _ = per_query_energy_tensor(responses[:, query_indices, :])
    r_hat, U, s = estimate_discriminative_rank(E, n_elbows=1)
    rho_hat, rhos = estimate_rho(U, r_hat, tau=tau)

    return {
        "n_queries": len(query_indices),
        "r_hat": r_hat,
        "rho_hat": rho_hat,
        "rhos": rhos,
        "sv_ratio": float(s[0] / s[1]) if len(s) > 1 else np.inf,
        "mstar_80": predict_mstar(r_hat, rho_hat, epsilon=0.20),
        "mstar_90": predict_mstar(r_hat, rho_hat, epsilon=0.10),
        "mstar_95": predict_mstar(r_hat, rho_hat, epsilon=0.05),
    }


def process_npz(npz_path, query_set_keys, tau=0.01):
    """Compute estimates for all query subsets in one NPZ file.

    Parameters
    ----------
    npz_path : str or Path
    query_set_keys : dict mapping query_set_name → NPZ key
    tau : float

    Returns
    -------
    list of dicts, one per query set.
    """
    data = np.load(str(npz_path), allow_pickle=True)
    responses = data["responses"]
    results = []

    for qs_name, qs_key in query_set_keys.items():
        if qs_key not in data:
            continue
        idx = data[qs_key]
        if len(idx) == 0:
            continue
        res = process_query_set(responses, idx, tau=tau)
        res["query_set"] = qs_name
        results.append(res)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute r̂, ρ̂, m* for all experiments"
    )
    parser.add_argument("--tau", type=float, default=0.05,
                        help="Threshold for zero-set probability (default: 0.05)")
    args = parser.parse_args()

    rows = []

    # --- System prompt experiments ---
    sp_dir = Path("results/system_prompt/embeddings")
    sp_query_sets = {
        "signal": "signal_indices",
        "weak_signal": "weak_signal_indices",
        "null": "null_indices",
    }

    if sp_dir.exists():
        for npz_path in sorted(sp_dir.glob("*.npz")):
            stem = npz_path.stem
            parts = stem.split("__")
            if len(parts) != 2:
                print(f"Skipping {npz_path.name} (unexpected naming)")
                continue
            base_model, embed_model = parts

            print(f"Processing system_prompt: {base_model} + {embed_model}")
            results = process_npz(npz_path, sp_query_sets, tau=args.tau)

            for res in results:
                rows.append({
                    "experiment": "system_prompt",
                    "base_model": base_model,
                    "embed_model": embed_model,
                    "query_set": res["query_set"],
                    "n_queries": res["n_queries"],
                    "r_hat": res["r_hat"],
                    "rho_hat": round(res["rho_hat"], 4),
                    "sv_ratio": round(res["sv_ratio"], 2),
                    "mstar_80": res["mstar_80"],
                    "mstar_90": res["mstar_90"],
                    "mstar_95": res["mstar_95"],
                })
                print(f"  {res['query_set']:15s} (M={res['n_queries']:3d}): "
                      f"r̂={res['r_hat']}, ρ̂={res['rho_hat']:.4f}, "
                      f"σ₁/σ₂={res['sv_ratio']:.2f}, m*={res['mstar_95']}")

    # --- Motivating experiment ---
    mot_npz = Path("results/motivating/motivating_responses.npz")
    mot_query_sets = {
        "signal": "sensitive_indices",
        "orthogonal": "orthogonal_indices",
    }

    if mot_npz.exists():
        print(f"\nProcessing motivating experiment")
        results = process_npz(mot_npz, mot_query_sets, tau=args.tau)

        for res in results:
            rows.append({
                "experiment": "motivating",
                "base_model": "Qwen2.5-1.5B",
                "embed_model": "nomic-embed-text-v1.5",
                "query_set": res["query_set"],
                "n_queries": res["n_queries"],
                "r_hat": res["r_hat"],
                "rho_hat": round(res["rho_hat"], 4),
                "sv_ratio": round(res["sv_ratio"], 2),
                "mstar_80": res["mstar_80"],
                "mstar_90": res["mstar_90"],
                "mstar_95": res["mstar_95"],
            })
            print(f"  {res['query_set']:15s} (M={res['n_queries']:3d}): "
                  f"r̂={res['r_hat']}, ρ̂={res['rho_hat']:.4f}, "
                  f"σ₁/σ₂={res['sv_ratio']:.2f}, m*={res['mstar_95']}")

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
