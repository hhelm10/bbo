#!/usr/bin/env python
"""CLI entry point for the RAG compliance auditing experiment.

Usage:
    python scripts/run_rag.py --step stores
    python scripts/run_rag.py --step assign
    python scripts/run_rag.py --step queries
    python scripts/run_rag.py --step generate --max-workers 4
    python scripts/run_rag.py --step embed
    python scripts/run_rag.py --step classify
    python scripts/run_rag.py --step pilot           # minimal validation run
    python scripts/run_rag.py --step all
    python scripts/run_rag.py --step all --pilot      # full pipeline in pilot mode
"""

import argparse
from pathlib import Path

from bbo.experiments.rag.config import RAGConfig


STEPS = ["stores", "assign", "queries", "generate", "embed", "classify"]


def run_step(step: str, config: RAGConfig, args):
    if step == "stores":
        from bbo.experiments.rag.document_store import run_create_stores
        run_create_stores(config)

    elif step == "assign":
        from bbo.experiments.rag.document_store import run_create_stores
        from bbo.experiments.rag.system_assignment import create_system_assignments
        # Ensure stores exist
        stores = run_create_stores(config)
        create_system_assignments(config, stores)

    elif step == "queries":
        from bbo.experiments.rag.retrieval import generate_queries, compute_retrieval_mapping
        generate_queries(config)
        # Also compute retrieval mapping (requires stores + assignments)
        compute_retrieval_mapping(config)

    elif step == "generate":
        from bbo.experiments.rag.generate_responses import run_generate
        run_generate(config, max_workers=args.max_workers)

    elif step == "embed":
        from bbo.experiments.rag.embed_responses import run_embed
        run_embed(config, embedding_model=args.embedding_model)

    elif step == "classify":
        from bbo.experiments.rag.run_classification import run_classification
        df = run_classification(config, embedding_model=args.embedding_model)
        out_path = config.classification_csv(embedding_model=args.embedding_model)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\nResults saved to {out_path}")
        print(df.to_string(index=False))

    else:
        raise ValueError(f"Unknown step: {step}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the RAG compliance auditing experiment"
    )
    parser.add_argument("--step", default="all",
                        help=f"Pipeline step: {', '.join(STEPS)}, all, pilot")
    parser.add_argument("--pilot", action="store_true",
                        help="Run in pilot mode (4 systems, 30 queries)")
    parser.add_argument("--embedding-model", default=None,
                        help="Embedding model (default: nomic-embed-text-v1.5)")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Parallel threads for API calls (default: 4)")
    parser.add_argument("--n-reps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config_kwargs = {}
    if args.pilot:
        config_kwargs["pilot"] = True
    if args.n_reps is not None:
        config_kwargs["n_reps"] = args.n_reps
    if args.output_dir is not None:
        config_kwargs["output_dir"] = args.output_dir
    if args.seed is not None:
        config_kwargs["seed"] = args.seed

    config = RAGConfig(**config_kwargs)

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    config.save(str(Path(config.output_dir) / "config.json"))

    if args.step == "pilot":
        # Pilot is just --pilot with all steps
        if not config.pilot:
            config = RAGConfig(pilot=True, **{k: v for k, v in config_kwargs.items()
                                               if k != "pilot"})
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)
            config.save(str(Path(config.output_dir) / "config.json"))
        steps = STEPS
    elif args.step == "all":
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
