#!/usr/bin/env python
"""Varimax check: do the r=2 RAG discriminative directions align with the
designed finance/HR domains after a sparsity-maximizing rotation?

For each embedding model: SVD of E_disc -> varimax on U[:, :2] -> report
per-domain mean |loading| and fraction of queries dominant on each rotated
direction.

Outputs:
    results/varimax_check/summary.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

from bbo.distances.energy import per_query_dissimilarity_tensor
from bbo.estimation.rank_rho import estimate_discriminative_rank

NPZ_PATHS = {
    "nomic-embed-text-v1.5": "results/rag/embeddings/ministral-8b__nomic-embed-text-v1.5.npz",
    "all-MiniLM-L6-v2": "results/rag/embeddings/ministral-8b__all-MiniLM-L6-v2.npz",
    "bge-large-en-v1.5": "results/rag/embeddings/ministral-8b__bge-large-en-v1.5.npz",
    "text-embedding-3-small": "results/rag/embeddings/ministral-8b__text-embedding-3-small.npz",
}


def varimax(Phi, gamma=1.0, q_iter=100, tol=1e-8):
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for _ in range(q_iter):
        L = Phi @ R
        u, s, vt = np.linalg.svd(
            Phi.T @ (L**3 - (gamma / p) * L @ np.diag(np.sum(L**2, axis=0))))
        R = u @ vt
        d_new = np.sum(s)
        if d_new < d * (1 + tol):
            break
        d = d_new
    return Phi @ R, R


def main():
    out_dir = Path("results/varimax_check")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for em, npz_path in NPZ_PATHS.items():
        d = np.load(npz_path, allow_pickle=True)
        responses, labels = d["responses"], d["labels"]
        fin, hr, ctrl = (d["finance_signal_indices"], d["hr_signal_indices"],
                         d["control_indices"])
        all_idx = np.concatenate([fin, hr, ctrl])
        groups = np.array(["fin"] * len(fin) + ["hr"] * len(hr)
                          + ["ctrl"] * len(ctrl))

        E, pairs = per_query_dissimilarity_tensor(
            responses[:, all_idx, :], metric="sq_euclidean")
        r_hat, U, s = estimate_discriminative_rank(E)

        V, R = varimax(U[:, :2])
        # Order rotated directions so direction 1 = finance-dominant
        fin_mass = [np.abs(V[groups == "fin", l]).mean() for l in range(2)]
        if fin_mass[1] > fin_mass[0]:
            V = V[:, ::-1]

        print(f"\n{em}: r_hat={r_hat}")
        for l in range(2):
            dom = np.abs(V[:, l]) > np.abs(V[:, 1 - l])
            row = {"embed_model": em, "r_hat": r_hat, "direction": l + 1}
            for g in ["fin", "hr", "ctrl"]:
                m = groups == g
                row[f"{g}_mean_abs_loading"] = float(np.abs(V[m, l]).mean())
                row[f"{g}_frac_dominant"] = float(dom[m].mean())
            rows.append(row)
            print(f"  rotated dir {l+1}: " + ", ".join(
                f"{g}: |v|={row[f'{g}_mean_abs_loading']:.4f} "
                f"dom={row[f'{g}_frac_dominant']:.2f}"
                for g in ["fin", "hr", "ctrl"]))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary.csv", index=False)
    print(f"\nSaved to {out_dir}/summary.csv")


if __name__ == "__main__":
    main()
