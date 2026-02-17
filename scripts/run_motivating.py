#!/usr/bin/env python
"""CLI entry point for the motivating example experiment.

Usage:
    python scripts/run_motivating.py --step all           # full pipeline
    python scripts/run_motivating.py --step prepare       # data preparation only
    python scripts/run_motivating.py --step train         # fine-tuning only
    python scripts/run_motivating.py --step generate      # response generation only
    python scripts/run_motivating.py --step embed         # embedding only
    python scripts/run_motivating.py --step classify      # classification only
    python scripts/run_motivating.py --step plot          # plotting only

    # Smoke test with small settings:
    python scripts/run_motivating.py --step all --n-per-class 3 \\
        --n-sensitive-queries 10 --n-orthogonal-queries 10
"""

import argparse
from pathlib import Path

from bbo.experiments.motivating.config import MotivatingConfig


STEPS = ["prepare", "train", "generate", "embed", "classify", "plot"]


def run_step(step: str, config: MotivatingConfig):
    """Run a single pipeline step."""
    if step == "prepare":
        from bbo.experiments.motivating.prepare_data import run_prepare
        run_prepare(config)

    elif step == "train":
        from bbo.experiments.motivating.train_adapters import run_train
        run_train(config)

    elif step == "generate":
        from bbo.experiments.motivating.generate_responses import run_generate
        run_generate(config)

    elif step == "embed":
        from bbo.experiments.motivating.embed_responses import run_embed
        run_embed(config)

    elif step == "classify":
        from bbo.experiments.motivating.run_classification import run_classification
        import pandas as pd
        df = run_classification(config)
        out_path = Path(config.output_dir) / "classification_results.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\nResults saved to {out_path}")
        print(df.to_string(index=False))

    elif step == "plot":
        import numpy as np
        import pandas as pd
        from bbo.plotting.real_plots import plot_exp8
        from bbo.plotting.motivating_plots import plot_motivating_figure

        csv_path = Path(config.output_dir) / "classification_results.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Classification results not found at {csv_path}. "
                "Run 'classify' step first."
            )

        npz_path = config.npz_path
        if not npz_path.exists():
            raise FileNotFoundError(
                f"Embedded responses not found at {npz_path}. "
                "Run 'embed' step first."
            )

        fig_dir = Path(config.output_dir) / "figures"

        # Existing per-experiment plot
        df = pd.read_csv(csv_path)
        plot_exp8(df, str(fig_dir))

        # Motivating combined figure
        data = np.load(str(npz_path), allow_pickle=True)
        metadata_path = str(config.data_dir / "adapter_metadata.json")
        plot_motivating_figure(
            responses=data["responses"],
            labels=data["labels"],
            sensitive_indices=data["sensitive_indices"],
            orthogonal_indices=data["orthogonal_indices"],
            metadata_path=metadata_path,
            classification_csv=str(csv_path),
            output_dir="figures",
        )
        print(f"Figures saved to {fig_dir}")

    else:
        raise ValueError(f"Unknown step: {step}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the motivating example experiment"
    )
    parser.add_argument(
        "--step", default="all",
        help=f"Pipeline step to run. One of: {', '.join(STEPS)}, all"
    )
    parser.add_argument("--base-model", default=None,
                        help="Base model (default: Qwen/Qwen2.5-1.5B)")
    parser.add_argument("--n-per-class", type=int, default=None,
                        help="Number of adapters per class (default: 50)")
    parser.add_argument("--n-sensitive-queries", type=int, default=None,
                        help="Number of sensitive queries (default: 100)")
    parser.add_argument("--n-orthogonal-queries", type=int, default=None,
                        help="Number of orthogonal queries (default: 100)")
    parser.add_argument("--n-reps", type=int, default=None,
                        help="Number of classification reps (default: 200)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: results/motivating)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Build config, overriding defaults with CLI args
    config_kwargs = {}
    if args.base_model is not None:
        config_kwargs["base_model"] = args.base_model
    if args.n_per_class is not None:
        config_kwargs["n_per_class"] = args.n_per_class
    if args.n_sensitive_queries is not None:
        config_kwargs["n_sensitive_queries"] = args.n_sensitive_queries
    if args.n_orthogonal_queries is not None:
        config_kwargs["n_orthogonal_queries"] = args.n_orthogonal_queries
    if args.n_reps is not None:
        config_kwargs["n_reps"] = args.n_reps
    if args.output_dir is not None:
        config_kwargs["output_dir"] = args.output_dir
    if args.seed is not None:
        config_kwargs["seed"] = args.seed

    config = MotivatingConfig(**config_kwargs)

    # Save config
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    config.save(str(Path(config.output_dir) / "config.json"))

    # Determine steps to run
    if args.step == "all":
        steps = STEPS
    else:
        steps = [args.step]

    for step in steps:
        print(f"\n{'='*60}")
        print(f"Step: {step}")
        print(f"{'='*60}")
        run_step(step, config)

    print("\nDone!")


if __name__ == "__main__":
    main()
