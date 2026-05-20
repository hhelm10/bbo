"""Create a compressed data package for sharing with a colleague."""

import shutil
import tarfile
from pathlib import Path

import numpy as np

from bbo.distances.energy import pairwise_energy_distances_t0, per_query_energy_tensor
from bbo.embedding.mds import ClassicalMDS
from bbo.estimation.rank_rho import estimate_discriminative_rank, estimate_rho

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "data_package"
M_MDS = 5
SEED = 0


def build_motivating(out_dir: Path):
    npz = np.load(RESULTS / "motivating" / "motivating_responses.npz",
                  allow_pickle=True)
    responses = npz["responses"]
    labels = npz["labels"]
    sig_idx = npz["sensitive_indices"]
    orth_idx = npz["orthogonal_indices"]

    rng = np.random.RandomState(SEED)

    # --- True partition ---
    true_dir = out_dir / "true_partition"
    true_dir.mkdir(parents=True)

    np.save(true_dir / "signal_indices.npy", sig_idx)
    np.save(true_dir / "orthogonal_indices.npy", orth_idx)

    sig_sub = rng.choice(sig_idx, size=M_MDS, replace=False)
    D_sig = pairwise_energy_distances_t0(responses, sig_sub)
    X_sig = ClassicalMDS(n_components=2).fit_transform(D_sig)
    np.save(true_dir / "mds_signal_m5.npy", X_sig)

    orth_sub = rng.choice(orth_idx, size=M_MDS, replace=False)
    D_orth = pairwise_energy_distances_t0(responses, orth_sub)
    X_orth = ClassicalMDS(n_components=2).fit_transform(D_orth)
    np.save(true_dir / "mds_orthogonal_m5.npy", X_orth)

    # --- Estimated (GMM) partition ---
    est_dir = out_dir / "estimated_partition"
    est_dir.mkdir(parents=True)

    query_pool = np.concatenate([sig_idx, orth_idx])
    E, _pairs = per_query_energy_tensor(responses[:, query_pool, :])
    r_hat, U, s = estimate_discriminative_rank(E)
    rho_hats, gmm_info = estimate_rho(U, r_hat)

    gmm_labels = gmm_info["per_direction"][0]["labels"]
    est_signal = query_pool[np.where(gmm_labels == 1)[0]]
    est_ortho = query_pool[np.where(gmm_labels == 0)[0]]

    np.save(est_dir / "est_signal_indices.npy", est_signal)
    np.save(est_dir / "est_orthogonal_indices.npy", est_ortho)
    np.save(est_dir / "svd_loadings.npy", np.abs(U[:, :r_hat]))
    np.save(est_dir / "singular_values.npy", s)
    np.save(est_dir / "rho_hats.npy", rho_hats)
    (est_dir / "r_hat.txt").write_text(str(r_hat))

    rng2 = np.random.RandomState(SEED)
    sig_sub_est = rng2.choice(est_signal, size=M_MDS, replace=False)
    D_sig_est = pairwise_energy_distances_t0(responses, sig_sub_est)
    X_sig_est = ClassicalMDS(n_components=2).fit_transform(D_sig_est)
    np.save(est_dir / "mds_est_signal_m5.npy", X_sig_est)

    orth_sub_est = rng2.choice(est_ortho, size=M_MDS, replace=False)
    D_orth_est = pairwise_energy_distances_t0(responses, orth_sub_est)
    X_orth_est = ClassicalMDS(n_components=2).fit_transform(D_orth_est)
    np.save(est_dir / "mds_est_orthogonal_m5.npy", X_orth_est)

    # --- Copy static files ---
    np.save(out_dir / "labels.npy", labels)
    shutil.copy2(RESULTS / "motivating" / "config.json", out_dir / "config.json")
    shutil.copy2(
        RESULTS / "motivating" / "classification_results.csv",
        out_dir / "classification_results.csv",
    )

    print(f"  True signal: {len(sig_idx)} queries, orthogonal: {len(orth_idx)}")
    print(f"  Est. signal: {len(est_signal)} queries, orthogonal: {len(est_ortho)}")
    print(f"  r̂ = {r_hat}, ρ̂ = {rho_hats}")


def build_synthetic(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    src = RESULTS / "synthetic"
    count = 0
    for f in sorted(src.iterdir()):
        if f.suffix in (".csv", ".json"):
            shutil.copy2(f, out_dir / f.name)
            count += 1
    print(f"  Copied {count} files from synthetic/")


def write_readme(out_dir: Path):
    readme = """\
# BBO Data Package

Reproducibility data for "Black-Box Classification via MDS Embeddings".

## Directory Structure

```
data_package/
├── motivating/           # LoRA fine-tuning experiment (100 models, 200 queries)
│   ├── config.json       # Experiment parameters
│   ├── labels.npy        # (100,) binary class labels
│   ├── classification_results.csv  # Accuracy by method/n/m/distribution
│   ├── true_partition/   # Ground-truth signal/orthogonal split
│   │   ├── signal_indices.npy       # (100,) query indices
│   │   ├── orthogonal_indices.npy   # (100,) query indices
│   │   ├── mds_signal_m5.npy       # (100, 2) 2D MDS coordinates
│   │   └── mds_orthogonal_m5.npy   # (100, 2) 2D MDS coordinates
│   └── estimated_partition/  # GMM-estimated signal/orthogonal split
│       ├── est_signal_indices.npy
│       ├── est_orthogonal_indices.npy
│       ├── mds_est_signal_m5.npy
│       ├── mds_est_orthogonal_m5.npy
│       ├── svd_loadings.npy    # |U[:, :r̂]| from SVD of energy tensor
│       ├── singular_values.npy # Full singular value spectrum
│       ├── rho_hats.npy        # Estimated ρ per SVD direction
│       └── r_hat.txt           # Estimated discriminative rank
└── synthetic/            # Synthetic validation experiments
    ├── exp[1-5]_results.csv + exp[1-5]_config.json
    ├── exp_[e-g]_results.csv + exp_[e-g]_config.json
    └── panel_[c,e,f,g]_results.csv
```

## Classification Results CSV Columns

| Column | Description |
|--------|-------------|
| method | Embedding method: "mds" or "concat" |
| n | Number of models sampled per trial |
| distribution | Query subset: "relevant" (signal), "orthogonal", or "uniform" |
| m | Number of queries used |
| mean_accuracy | Mean classification accuracy over trials |
| std_accuracy | Standard deviation of accuracy |
| p10_accuracy | 10th percentile |
| p90_accuracy | 90th percentile |

Mean error = 1 - mean_accuracy.

## How MDS Coordinates Were Computed

1. Select m=5 queries from the signal (or orthogonal) set
2. Compute pairwise energy distances: D(f, f') = sqrt(sum_k ||emb_f(q_k) - emb_f'(q_k)||^2)
3. Apply classical MDS to the 100x100 distance matrix → 2D coordinates
4. Random seed = 0 for query subset selection

## How GMM Partition Was Computed

1. Compute per-query energy tensor E ∈ R^{M x C(n,2)} for all queries in the pool
2. SVD: E = U S V^T
3. Estimate discriminative rank r̂ from largest spectral gap
4. For each direction ℓ = 1..r̂, fit 2-component GMM on |U[:, ℓ]|
5. Component with lower mean = orthogonal (zero-set); higher mean = signal (active)
6. ρ̂_ℓ = weight of zero-set component

## Dependencies

numpy, scipy, scikit-learn (for GMM)
"""
    (out_dir / "README.md").write_text(readme)
    print("  Wrote README.md")


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()

    print("Building motivating data...")
    build_motivating(OUT / "motivating")

    print("Building synthetic data...")
    build_synthetic(OUT / "synthetic")

    print("Writing README...")
    write_readme(OUT)

    archive = ROOT / "data_package.tar.gz"
    print(f"Creating {archive.name}...")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(OUT, arcname="data_package")
    print(f"Done. Archive: {archive} ({archive.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
