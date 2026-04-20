#!/usr/bin/env python
"""Run synthetic estimation validation experiments (panels c, e, f, g)."""

import argparse
from pathlib import Path

from bbo.experiments.synthetic.estimation_validation import (
    run_panel_c, run_panel_e, run_panel_f, run_panel_g,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("panels", nargs="*", default=["c", "e", "f", "g"])
    parser.add_argument("--n-reps", type=int, default=1000)
    parser.add_argument("--output-dir", default="results/synthetic")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if "c" in args.panels:
        print("\n=== Panel (c): P[fail] vs m for varying n ===")
        df = run_panel_c(n_reps=args.n_reps)
        df.to_csv(f"{args.output_dir}/panel_c_results.csv", index=False)
        print(f"Saved to {args.output_dir}/panel_c_results.csv")

    if "e" in args.panels:
        print("\n=== Panel (e): Rank recovery ===")
        df = run_panel_e(n_reps=args.n_reps)
        df.to_csv(f"{args.output_dir}/panel_e_results.csv", index=False)
        print(f"Saved to {args.output_dir}/panel_e_results.csv")

    if "f" in args.panels:
        print("\n=== Panel (f): Zero-set probability recovery ===")
        df = run_panel_f(n_reps=args.n_reps)
        df.to_csv(f"{args.output_dir}/panel_f_results.csv", index=False)
        print(f"Saved to {args.output_dir}/panel_f_results.csv")

    if "g" in args.panels:
        print("\n=== Panel (g): Predicted vs empirical failure ===")
        df = run_panel_g(n_reps=args.n_reps)
        df.to_csv(f"{args.output_dir}/panel_g_results.csv", index=False)
        print(f"Saved to {args.output_dir}/panel_g_results.csv")


if __name__ == "__main__":
    main()
