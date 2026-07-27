"""Temperature-1.0 analysis (KojF Q6).

For each task, loads the K=5 temperature-1.0 draws, stacks them into
(n_models, M, K, p), and runs:

1. Estimation: r̂, ρ̂, sv-ratio, signal-set ARI using
   (a) the K-draw U-statistic squared energy distance tensor, and
   (b) each single draw with sq. Euclidean (K=1 view),
   compared against the temperature-0 run under the identical pipeline.

2. Classification: accuracy vs m (uniform query sampling, full panel,
   MDS -> RF) for temp-0, each temp-1 single draw, and the temp-1 U-stat
   distance, all under the same protocol.

Outputs: results/{task}_temp1/analysis/{estimation.csv, classification.csv}

Usage:
    python scripts/run_temp1_analysis.py --task rag
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bbo.distances.energy import (
    per_query_energy_tensor, per_query_energy_ustat_tensor,
)
from bbo.estimation.rank_rho import estimate_discriminative_rank, estimate_rho
from bbo.embedding.mds import ClassicalMDS
from bbo.classification.evaluate import classify_and_evaluate

REPO = Path(__file__).resolve().parent.parent
N_DRAWS = 5
M_VALUES = [1, 2, 5, 10, 20, 50, 100]
N_REPS = 200

# npz path templates and (signal_key, orthogonal_key) per task
TASKS = {
    "motivating": {
        "temp0": "results/motivating/motivating_responses.npz",
        "temp1": "results/motivating_temp1/draw{d}/motivating_responses.npz",
        "sig_key": "sensitive_indices",
        "orth_key": "orthogonal_indices",
        "n_components": 8,
    },
    "system_prompt": {
        "temp0": "results/system_prompt/embeddings/ministral-8b__nomic-embed-text-v1.5.npz",
        "temp1": "results/system_prompt_temp1/draw{d}/embeddings/ministral-8b__nomic-embed-text-v1.5.npz",
        "sig_key": "signal_indices",
        "orth_key": "orthogonal_indices",
        "n_components": None,
    },
    "rag": {
        "temp0": "results/rag/embeddings/ministral-8b__nomic-embed-text-v1.5.npz",
        "temp1": "results/rag_temp1/draw{d}/embeddings/ministral-8b__nomic-embed-text-v1.5.npz",
        "sig_key": "signal_indices",
        "orth_key": "control_indices",
        "n_components": None,
    },
}


def ari_permutation_test(true, est, n_perm=10000, seed=0):
    rng = np.random.default_rng(seed)
    ari = adjusted_rand_score(true, est)
    null = np.array([adjusted_rand_score(true, rng.permutation(est))
                     for _ in range(n_perm)])
    p = (1 + np.sum(null >= ari)) / (1 + n_perm)
    return ari, p


def load_task(task):
    spec = TASKS[task]
    d0 = np.load(REPO / spec["temp0"], allow_pickle=True)
    resp0 = d0["responses"]
    labels = d0["labels"]
    sig_idx = d0[spec["sig_key"]]
    orth_idx = d0[spec["orth_key"]]

    draws = []
    for d in range(N_DRAWS):
        dd = np.load(REPO / spec["temp1"].format(d=d), allow_pickle=True)
        assert np.array_equal(dd["labels"], labels), f"label mismatch draw {d}"
        draws.append(dd["responses"])
    resp1 = np.stack(draws, axis=2)  # (n, M, K, p)
    return resp0, resp1, labels, sig_idx, orth_idx


def estimation_row(E, true_signal, label):
    r_hat, U, s = estimate_discriminative_rank(E)
    rho_hats, info = estimate_rho(U, r_hat)
    est_signal = info["per_direction"][0]["labels"]
    sv_ratio = float(s[0] / s[1]) if len(s) > 1 else np.inf
    ari, ari_p = ari_permutation_test(true_signal, est_signal)
    print(f"  {label:16s}: r_hat={r_hat}, rho_1={rho_hats[0]:.3f}, "
          f"sv_ratio={sv_ratio:6.2f}, ari={ari:.3f} (p={ari_p:.4f})")
    return {
        "condition": label,
        "r_hat": r_hat,
        "rho_1": float(rho_hats[0]),
        "sv_ratio": sv_ratio,
        "ari": ari,
        "ari_pvalue": ari_p,
    }


def run_estimation(task, resp0, resp1, sig_idx, orth_idx):
    all_idx = np.concatenate([sig_idx, orth_idx])
    true_signal = np.zeros(len(all_idx), dtype=int)
    true_signal[:len(sig_idx)] = 1

    rows = []
    E0, _ = per_query_energy_tensor(resp0[:, all_idx, :])
    rows.append(estimation_row(E0, true_signal, "temp0"))

    for d in range(N_DRAWS):
        E1d, _ = per_query_energy_tensor(resp1[:, all_idx, d, :])
        rows.append(estimation_row(E1d, true_signal, f"temp1_draw{d}"))

    Eu, _ = per_query_energy_ustat_tensor(resp1[:, all_idx, :, :])
    rows.append(estimation_row(Eu, true_signal, "temp1_ustat_K5"))

    df = pd.DataFrame(rows)
    df.insert(0, "task", task)
    return df


METRICS = ["sq_euclidean", "euclidean", "cosine", "l1", "rbf"]


def run_metric_study(task, resp1, sig_idx, orth_idx):
    """Estimation under the K=5 U-stat tensor for each base metric delta."""
    all_idx = np.concatenate([sig_idx, orth_idx])
    true_signal = np.zeros(len(all_idx), dtype=int)
    true_signal[:len(sig_idx)] = 1

    rows = []
    for metric in METRICS:
        Eu, _ = per_query_energy_ustat_tensor(resp1[:, all_idx, :, :],
                                              metric=metric)
        rows.append(estimation_row(Eu, true_signal, f"ustat_{metric}"))
    df = pd.DataFrame(rows)
    df.insert(0, "task", task)
    return df


def _classify_trial(T, labels, M, m, seed, n_components):
    rng = np.random.default_rng(seed)
    q = rng.choice(M, size=m, replace=False)
    D_sq = np.maximum(T[q].sum(axis=0), 0.0)
    n = len(labels)
    D = np.zeros((n, n))
    iu = np.triu_indices(n, k=1)
    D[iu] = np.sqrt(D_sq)
    D = D + D.T
    nc = min(n_components, n - 1) if n_components is not None else None
    mds = ClassicalMDS(n_components=nc) if nc else ClassicalMDS()
    X = mds.fit_transform(D)
    return classify_and_evaluate(X, labels, "rf", random_state=seed)


def run_classification(task, resp0, resp1, labels, n_components, n_jobs=-1):
    M = resp0.shape[1]
    tensors = {"temp0": per_query_energy_tensor(resp0)[0]}
    for d in range(N_DRAWS):
        tensors[f"temp1_draw{d}"] = per_query_energy_tensor(resp1[:, :, d, :])[0]
    tensors["temp1_ustat_K5"] = per_query_energy_ustat_tensor(resp1)[0]

    rows = []
    for cond, T in tensors.items():
        for m in M_VALUES:
            seeds = [1234 + rep * 9973 + m * 101 for rep in range(N_REPS)]
            errs = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_classify_trial)(T, labels, M, m, s, n_components)
                for s in seeds
            )
            acc = 1.0 - float(np.mean(errs))
            rows.append({"task": task, "condition": cond, "m": m,
                         "accuracy": acc, "n_reps": N_REPS})
            print(f"  {cond:16s} m={m:3d}: acc={acc:.3f}")

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=list(TASKS))
    parser.add_argument("--step", default="all",
                        choices=["all", "estimation", "classification", "metrics"])
    args = parser.parse_args()

    out_dir = REPO / "results" / f"{args.task}_temp1" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.task} data...")
    resp0, resp1, labels, sig_idx, orth_idx = load_task(args.task)
    print(f"  temp0: {resp0.shape}, temp1: {resp1.shape}")

    if args.step in ("all", "estimation"):
        print("\n=== Estimation ===")
        df = run_estimation(args.task, resp0, resp1, sig_idx, orth_idx)
        df.to_csv(out_dir / "estimation.csv", index=False)
        print(f"Saved {out_dir / 'estimation.csv'}")

    if args.step == "metrics":
        print("\n=== Metric study (U-stat K=5) ===")
        df = run_metric_study(args.task, resp1, sig_idx, orth_idx)
        df.to_csv(out_dir / "estimation_metrics.csv", index=False)
        print(f"Saved {out_dir / 'estimation_metrics.csv'}")

    if args.step in ("all", "classification"):
        print("\n=== Classification ===")
        nc = TASKS[args.task]["n_components"]
        df = run_classification(args.task, resp0, resp1, labels, nc)
        df.to_csv(out_dir / "classification.csv", index=False)
        print(f"Saved {out_dir / 'classification.csv'}")


if __name__ == "__main__":
    main()
