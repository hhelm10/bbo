#!/usr/bin/env python
"""CLI entry point for synthetic experiments (Exp 1-5)."""

import argparse
import json
from pathlib import Path
import pandas as pd

from bbo.experiments.config import (
    Exp1Config, Exp2Config, Exp3Config, Exp4Config, Exp5Config,
)
from bbo.experiments.synthetic.exp1_error_vs_m_rank import run_exp1
from bbo.experiments.synthetic.exp2_error_vs_m_rho import run_exp2
from bbo.experiments.synthetic.exp3_query_distribution import run_exp3
from bbo.experiments.synthetic.exp4_error_vs_n import run_exp4
from bbo.experiments.synthetic.exp5_bayes_convergence import run_exp5
from bbo.plotting.synthetic_plots import (
    plot_exp1, plot_exp2, plot_exp3, plot_exp4, plot_exp5,
    plot_figure1,
)


EXPERIMENTS = {
    "exp1": (Exp1Config, run_exp1, plot_exp1),
    "exp2": (Exp2Config, run_exp2, plot_exp2),
    "exp3": (Exp3Config, run_exp3, plot_exp3),
    "exp4": (Exp4Config, run_exp4, plot_exp4),
    "exp5": (Exp5Config, run_exp5, plot_exp5),
}


def main():
    parser = argparse.ArgumentParser(description="Run synthetic BBO experiments")
    parser.add_argument("experiments", nargs="*", default=list(EXPERIMENTS.keys()),
                        help="Which experiments to run (e.g., exp1 exp2)")
    parser.add_argument("--output-dir", default="results/synthetic",
                        help="Output directory for results")
    parser.add_argument("--figure-dir", default="results/figures",
                        help="Output directory for figures")
    parser.add_argument("--n-reps", type=int, default=None,
                        help="Override number of repetitions")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plotting")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.figure_dir).mkdir(parents=True, exist_ok=True)

    results = {}

    for exp_name in args.experiments:
        if exp_name not in EXPERIMENTS:
            print(f"Unknown experiment: {exp_name}. Available: {list(EXPERIMENTS.keys())}")
            continue

        print(f"\n{'='*60}")
        print(f"Running {exp_name}")
        print(f"{'='*60}")

        ConfigClass, run_fn, plot_fn = EXPERIMENTS[exp_name]
        config = ConfigClass(seed=args.seed)
        if args.n_reps is not None:
            config.n_reps = args.n_reps

        # Save config
        config.save(f"{args.output_dir}/{exp_name}_config.json")

        # Run experiment
        df = run_fn(config)
        results[exp_name] = df

        # Save results
        csv_path = f"{args.output_dir}/{exp_name}_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

        # Plot individual
        if not args.no_plot:
            plot_fn(df, output_dir=args.figure_dir)
            print(f"Figure saved to {args.figure_dir}/")

    # Generate combined Figure 1 if both exp1 and exp2 were run
    if not args.no_plot and "exp1" in results and "exp2" in results:
        plot_figure1(results["exp1"], results["exp2"], output_dir=args.figure_dir)
        print(f"Combined figure1 saved to {args.figure_dir}/figure1_error_vs_m.pdf")


if __name__ == "__main__":
    main()
