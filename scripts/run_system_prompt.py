#!/usr/bin/env python
"""CLI entry point for the system prompt auditing experiment.

Usage:
    python scripts/run_system_prompt.py --step all
    python scripts/run_system_prompt.py --step prepare
    python scripts/run_system_prompt.py --step generate --base-model mistral-small
    python scripts/run_system_prompt.py --step embed --base-model mistral-small --embedding-model nomic-embed-text-v1.5
    python scripts/run_system_prompt.py --step classify --base-model mistral-small --embedding-model nomic-embed-text-v1.5
    python scripts/run_system_prompt.py --step plot
"""

import argparse
from pathlib import Path

from bbo.experiments.system_prompt.config import SystemPromptConfig


STEPS = ["prepare", "generate", "embed", "classify", "plot"]


def run_step(step: str, config: SystemPromptConfig, args):
    if step == "prepare":
        from bbo.experiments.system_prompt.prepare_data import run_prepare
        run_prepare(config)

    elif step == "generate":
        from bbo.experiments.system_prompt.generate_responses import run_generate
        model_ids = None
        if hasattr(args, 'model_ids') and args.model_ids:
            model_ids = []
            for part in args.model_ids.split(","):
                if "-" in part:
                    lo, hi = part.split("-", 1)
                    model_ids.extend(range(int(lo), int(hi) + 1))
                else:
                    model_ids.append(int(part))
            model_ids = sorted(set(model_ids))
        run_generate(config, base_model=args.base_model,
                     max_workers=args.max_workers, model_ids=model_ids)

    elif step == "embed":
        from bbo.experiments.system_prompt.embed_responses import run_embed
        truncate = getattr(args, 'truncate_chars', None)
        run_embed(config, base_model=args.base_model,
                  embedding_model=args.embedding_model,
                  truncate_chars=truncate)

    elif step == "classify":
        from bbo.experiments.system_prompt.run_classification import run_classification
        import pandas as pd
        bm = args.base_model or config.base_models[0]
        em = args.embedding_model or config.embedding_models[0]
        df = run_classification(config, base_model=bm, embedding_model=em)
        out_path = config.classification_csv(bm, em)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\nResults saved to {out_path}")
        print(df.to_string(index=False))

    elif step == "plot":
        from bbo.plotting.system_prompt_plots import plot_system_prompt_figures
        plot_system_prompt_figures(config, output_dir="figures")

    else:
        raise ValueError(f"Unknown step: {step}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the system prompt auditing experiment"
    )
    parser.add_argument("--step", default="all",
                        help=f"Pipeline step: {', '.join(STEPS)}, all")
    parser.add_argument("--base-model", default=None,
                        help="Base model (e.g. mistral-small). Default: all.")
    parser.add_argument("--embedding-model", default=None,
                        help="Embedding model (e.g. nomic-embed-text-v1.5). Default: all.")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Parallel threads for API calls (default: 4)")
    parser.add_argument("--model-ids", default=None,
                        help="Comma-separated model IDs to generate (e.g. 0,1,50,51)")
    parser.add_argument("--truncate-chars", type=int, default=None,
                        help="Truncate responses to N chars before embedding")
    parser.add_argument("--n-reps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config_kwargs = {}
    if args.n_reps is not None:
        config_kwargs["n_reps"] = args.n_reps
    if args.output_dir is not None:
        config_kwargs["output_dir"] = args.output_dir
    if args.seed is not None:
        config_kwargs["seed"] = args.seed

    config = SystemPromptConfig(**config_kwargs)

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    config.save(str(Path(config.output_dir) / "config.json"))

    if args.step == "all":
        steps = STEPS
    else:
        steps = [args.step]

    for step in steps:
        print(f"\n{'='*60}")
        print(f"Step: {step}")
        print(f"{'='*60}")
        run_step(step, config, args)

    print("\nDone!")


if __name__ == "__main__":
    main()
