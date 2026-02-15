#!/usr/bin/env python
"""CLI entry point for real LLM experiments (Exp 6-10)."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from bbo.experiments.real.data_loader import load_responses_npz
from bbo.experiments.real.exp6_effective_rank import run_exp6
from bbo.experiments.real.exp7_accuracy_vs_m import run_exp7
from bbo.experiments.real.exp8_relevant_vs_orthogonal import run_exp8
from bbo.experiments.real.exp9_baselines import run_exp9
from bbo.experiments.real.exp10_predict_mstar import fit_decay_curve
from bbo.experiments.config import Exp7Config
from bbo.plotting.real_plots import (
    plot_exp6_scree, plot_exp7, plot_exp8, plot_exp9,
)


def main():
    parser = argparse.ArgumentParser(description="Run real LLM BBO experiments")
    parser.add_argument("data_path", help="Path to .npz file with precomputed responses")
    parser.add_argument("experiments", nargs="*", default=["exp6", "exp7", "exp8", "exp9", "exp10"],
                        help="Which experiments to run")
    parser.add_argument("--output-dir", default="results/real",
                        help="Output directory")
    parser.add_argument("--figure-dir", default="results/figures",
                        help="Figure output directory")
    parser.add_argument("--n-reps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.figure_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {args.data_path}")
    responses, labels, model_names = load_responses_npz(args.data_path)
    print(f"  {len(labels)} models, {responses.shape[1]} queries, {responses.shape[2]}d embeddings")
    print(f"  Labels: {np.unique(labels, return_counts=True)}")

    for exp_name in args.experiments:
        print(f"\n{'='*60}")
        print(f"Running {exp_name}")
        print(f"{'='*60}")

        if exp_name == "exp6":
            result = run_exp6(responses)
            print(f"  Effective rank: r90={result['r90']}, r95={result['r95']}, r99={result['r99']}")
            np.savez(f"{args.output_dir}/exp6_results.npz", **result)
            if not args.no_plot:
                plot_exp6_scree(result["singular_values"], args.figure_dir)

        elif exp_name == "exp7":
            config = Exp7Config(seed=args.seed, n_reps=args.n_reps)
            df = run_exp7(responses, labels, config)
            df.to_csv(f"{args.output_dir}/exp7_results.csv", index=False)
            if not args.no_plot:
                plot_exp7(df, args.figure_dir)

        elif exp_name == "exp8":
            df = run_exp8(responses, labels, n_reps=args.n_reps, seed=args.seed)
            df.to_csv(f"{args.output_dir}/exp8_results.csv", index=False)
            if not args.no_plot:
                plot_exp8(df, args.figure_dir)

        elif exp_name == "exp9":
            df = run_exp9(responses, labels, n_reps=args.n_reps, seed=args.seed)
            df.to_csv(f"{args.output_dir}/exp9_results.csv", index=False)
            if not args.no_plot:
                plot_exp9(df, args.figure_dir)

        elif exp_name == "exp10":
            # Depends on exp7 results
            exp7_path = f"{args.output_dir}/exp7_results.csv"
            if not Path(exp7_path).exists():
                print("  Running exp7 first...")
                config = Exp7Config(seed=args.seed, n_reps=args.n_reps)
                df7 = run_exp7(responses, labels, config)
                df7.to_csv(exp7_path, index=False)
            else:
                df7 = pd.read_csv(exp7_path)

            m_vals = df7["m"].values
            error_probs = 1.0 - df7["mean_accuracy"].values
            result = fit_decay_curve(m_vals, error_probs)
            print(f"  Estimated: r_hat={result.get('r_hat', 'N/A'):.2f}, "
                  f"rho_hat={result.get('rho_hat', 'N/A'):.3f}")
            for k, v in result.items():
                if k.startswith("m_star"):
                    print(f"  {k}: {v}")

            import json
            with open(f"{args.output_dir}/exp10_results.json", "w") as f:
                json.dump({k: float(v) if isinstance(v, (int, float, np.floating)) else v
                           for k, v in result.items()
                           if not isinstance(v, np.ndarray)}, f, indent=2)


if __name__ == "__main__":
    main()
