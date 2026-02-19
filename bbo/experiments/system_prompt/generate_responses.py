"""Response generation for the system prompt experiment.

Calls base model APIs with each (system_prompt, query) pair.
Checkpoints after each system prompt to allow resumption.
Parallelized across queries with per-call retry/backoff.
"""

import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from bbo.api.clients import chat_completion
from bbo.experiments.system_prompt.config import SystemPromptConfig


def _call_one(base_model, system_text, query_text, temperature, max_tokens):
    """Single API call with error handling. Returns (success, response_or_error)."""
    try:
        resp = chat_completion(
            model=base_model,
            system_prompt=system_text,
            user_message=query_text,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return True, resp
    except Exception as e:
        return False, str(e)


def _generate_one_prompt(base_model, prompt, queries, temperature, max_tokens,
                         max_workers):
    """Generate responses for one system prompt across all queries, in parallel."""
    responses = [None] * len(queries)
    failures = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for i, q in enumerate(queries):
            fut = executor.submit(
                _call_one, base_model, prompt["text"], q["text"],
                temperature, max_tokens,
            )
            future_to_idx[fut] = i

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                ok, result = future.result()
                if ok:
                    responses[idx] = result
                else:
                    failures.append((idx, result))
                    responses[idx] = ""  # empty string placeholder
            except Exception as e:
                failures.append((idx, str(e)))
                responses[idx] = ""

    return responses, failures


def run_generate(config: SystemPromptConfig, base_model: str = None,
                 max_workers: int = 10, model_ids: list = None):
    """Generate responses for all (system_prompt, query) pairs.

    Parameters
    ----------
    config : SystemPromptConfig
    base_model : str, optional
        If provided, only generate for this base model. Otherwise all.
    max_workers : int
        Number of parallel threads for API calls per system prompt.
    model_ids : list of int, optional
        If provided, only generate for these model IDs.
    """
    with open(config.data_dir / "system_prompts.json") as f:
        prompts = json.load(f)

    if model_ids is not None:
        prompts = [p for p in prompts if p["id"] in model_ids]
    with open(config.data_dir / "queries.json") as f:
        queries = json.load(f)

    models = [base_model] if base_model else config.base_models

    for bm in models:
        out_dir = config.responses_dir(bm)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Check which prompts are already done (no empty responses)
        done = set()
        partial = {}  # model_id -> list of indices with empty responses
        for p in prompts:
            path = out_dir / f"model_{p['id']:03d}.json"
            if path.exists():
                with open(path) as f:
                    existing = json.load(f)
                if len(existing) == len(queries):
                    empty_idx = [i for i, r in enumerate(existing)
                                 if not r or (isinstance(r, str) and r.strip() == "")]
                    if empty_idx:
                        partial[p["id"]] = (existing, empty_idx)
                    else:
                        done.add(p["id"])

        # Retry partial models (fill in empty responses)
        if partial:
            print(f"[{bm}] Retrying {len(partial)} models with empty responses...")
            for pid, (existing, empty_idx) in partial.items():
                prompt = next(p for p in prompts if p["id"] == pid)
                retry_queries = [(i, queries[i]) for i in empty_idx]
                filled = 0
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_idx = {}
                    for i, q in retry_queries:
                        fut = executor.submit(
                            _call_one, bm, prompt["text"], q["text"],
                            config.temperature, config.max_tokens,
                        )
                        future_to_idx[fut] = i
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            ok, result = future.result()
                            if ok:
                                existing[idx] = result
                                filled += 1
                        except Exception:
                            pass
                out_path = out_dir / f"model_{pid:03d}.json"
                with open(out_path, "w") as f:
                    json.dump(existing, f)
                remaining_empty = sum(1 for r in existing
                                      if not r or (isinstance(r, str) and r.strip() == ""))
                if remaining_empty == 0:
                    done.add(pid)
                print(f"  model_{pid:03d}: filled {filled}/{len(empty_idx)}, "
                      f"{remaining_empty} still empty")

        remaining = [p for p in prompts if p["id"] not in done
                     and p["id"] not in partial]
        print(f"\n[{bm}] {len(done)}/{len(prompts)} done, "
              f"{len(remaining)} remaining ({len(queries)} queries each, "
              f"{max_workers} workers)")

        all_failures = []
        for prompt in tqdm(remaining, desc=f"[{bm}]"):
            responses, failures = _generate_one_prompt(
                bm, prompt, queries, config.temperature, config.max_tokens,
                max_workers,
            )

            if failures:
                all_failures.extend(
                    (prompt["id"], idx, err) for idx, err in failures
                )
                tqdm.write(f"  model_{prompt['id']:03d}: "
                           f"{len(failures)} failed queries")

            out_path = out_dir / f"model_{prompt['id']:03d}.json"
            with open(out_path, "w") as f:
                json.dump(responses, f)

        total = sum(1 for p in prompts
                    if (out_dir / f"model_{p['id']:03d}.json").exists())
        print(f"[{bm}] Complete: {total}/{len(prompts)} models")

        if all_failures:
            print(f"  WARNING: {len(all_failures)} total failed queries:")
            for pid, qidx, err in all_failures[:10]:
                print(f"    model_{pid:03d} query {qidx}: {err}")
            if len(all_failures) > 10:
                print(f"    ... and {len(all_failures) - 10} more")
