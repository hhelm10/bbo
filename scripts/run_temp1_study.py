"""Temperature-1.0 study driver (KojF Q6).

Runs K independent generation draws at temperature 1.0 for each task,
reusing the existing generate -> embed pipelines via per-draw output dirs:

    results/{task}_temp1/draw{d}/

Shared inputs (data, adapters, stores) are symlinked from the temperature-0
run so prepare/train steps are not repeated.

Usage:
    python scripts/run_temp1_study.py --task system_prompt --step setup
    python scripts/run_temp1_study.py --task system_prompt --step generate --draws 0 1
    python scripts/run_temp1_study.py --task system_prompt --step embed
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent
N_DRAWS = 5
EMBED_MODEL = "nomic-embed-text-v1.5"
BASE_MODEL = "ministral-8b"

SHARED_DIRS = {
    "motivating": ["data", "adapters"],
    "system_prompt": ["data"],
    "rag": ["data", "stores"],
}


def draw_dir(task: str, draw: int) -> Path:
    return REPO / "results" / f"{task}_temp1" / f"draw{draw}"


def make_config(task: str, draw: int):
    out = str(draw_dir(task, draw).relative_to(REPO))
    if task == "motivating":
        from bbo.experiments.motivating.config import MotivatingConfig
        return MotivatingConfig(output_dir=out, temperature=1.0)
    elif task == "system_prompt":
        from bbo.experiments.system_prompt.config import SystemPromptConfig
        return SystemPromptConfig(output_dir=out, temperature=1.0)
    elif task == "rag":
        from bbo.experiments.rag.config import RAGConfig
        return RAGConfig(output_dir=out, temperature=1.0)
    raise ValueError(f"Unknown task: {task}")


def run_setup(task: str, draws):
    src_root = REPO / "results" / task
    for d in draws:
        dd = draw_dir(task, d)
        dd.mkdir(parents=True, exist_ok=True)
        for name in SHARED_DIRS[task]:
            src = src_root / name
            if not src.exists():
                raise FileNotFoundError(f"Shared input missing: {src}")
            link = dd / name
            if link.is_symlink() or link.exists():
                continue
            link.symlink_to(src)
        print(f"[{task}] draw{d} ready at {dd}")


def run_generate(task: str, draws, max_workers: int, model_ids):
    for d in draws:
        config = make_config(task, d)
        print(f"\n=== [{task}] generate draw{d} (temperature=1.0) ===")
        if task == "motivating":
            from bbo.experiments.motivating.generate_responses import run_generate
            run_generate(config, seed=1000003 * (d + 1))
        elif task == "system_prompt":
            from bbo.experiments.system_prompt.generate_responses import run_generate
            run_generate(config, base_model=BASE_MODEL,
                         max_workers=max_workers, model_ids=model_ids)
        elif task == "rag":
            from bbo.experiments.rag.generate_responses import run_generate
            run_generate(config, max_workers=max_workers)


def run_embed(task: str, draws):
    for d in draws:
        config = make_config(task, d)
        print(f"\n=== [{task}] embed draw{d} ===")
        if task == "motivating":
            from bbo.experiments.motivating.embed_responses import run_embed
            run_embed(config, embedding_model=EMBED_MODEL)
        elif task == "system_prompt":
            from bbo.experiments.system_prompt.embed_responses import run_embed
            run_embed(config, base_model=BASE_MODEL, embedding_model=EMBED_MODEL)
        elif task == "rag":
            from bbo.experiments.rag.embed_responses import run_embed
            run_embed(config, embedding_model=EMBED_MODEL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True,
                        choices=["motivating", "system_prompt", "rag"])
    parser.add_argument("--step", required=True,
                        choices=["setup", "generate", "embed"])
    parser.add_argument("--draws", type=int, nargs="+",
                        default=list(range(N_DRAWS)))
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--model-ids", type=int, nargs="+", default=None,
                        help="system_prompt only: restrict to these model IDs")
    args = parser.parse_args()

    if args.step == "setup":
        run_setup(args.task, args.draws)
    elif args.step == "generate":
        run_setup(args.task, args.draws)
        run_generate(args.task, args.draws, args.max_workers, args.model_ids)
    elif args.step == "embed":
        run_embed(args.task, args.draws)


if __name__ == "__main__":
    main()
