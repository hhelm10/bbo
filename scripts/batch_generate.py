"""Generate responses using Batch APIs (OpenAI + Mistral).

Usage:
    python scripts/batch_generate.py submit   [--models ...]
    python scripts/batch_generate.py poll     [--models ...]
    python scripts/batch_generate.py download [--models ...]
"""

import json
import os
import time
import argparse
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from bbo.api.clients import MODEL_REGISTRY

DATA_DIR = Path("results/system_prompt/data")
BATCH_DIR = Path("results/system_prompt/batch_jobs")
RESPONSES_DIR = Path("results/system_prompt/raw_responses")


def load_experiment():
    with open(DATA_DIR / "system_prompts.json") as f:
        prompts = json.load(f)
    with open(DATA_DIR / "queries.json") as f:
        queries = json.load(f)
    return prompts, queries


def already_done(base_model, prompts, queries):
    """Return set of model IDs that already have complete responses."""
    out_dir = RESPONSES_DIR / base_model
    done = set()
    for p in prompts:
        path = out_dir / f"model_{p['id']:03d}.json"
        if path.exists():
            with open(path) as f:
                existing = json.load(f)
            if len(existing) == len(queries):
                empty = [i for i, r in enumerate(existing)
                         if not r or (isinstance(r, str) and r.strip() == "")]
                if not empty:
                    done.add(p["id"])
    return done


# ── OpenAI Batch ─────────────────────────────────────────────────

def submit_openai(base_model, prompts, queries, done_ids):
    from openai import OpenAI
    client = OpenAI()
    _, model_id = MODEL_REGISTRY[base_model]

    # Build JSONL
    jsonl_path = BATCH_DIR / f"{base_model}_input.jsonl"
    count = 0
    with open(jsonl_path, "w") as f:
        for p in prompts:
            if p["id"] in done_ids:
                continue
            for q in queries:
                line = {
                    "custom_id": f"m{p['id']:03d}_q{q['id']:03d}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model_id,
                        "temperature": 0,
                        "max_tokens": 128,
                        "messages": [
                            {"role": "system", "content": p["text"]},
                            {"role": "user", "content": q["text"]},
                        ],
                    },
                }
                f.write(json.dumps(line) + "\n")
                count += 1

    print(f"[{base_model}] Created {jsonl_path} with {count} requests")

    # Upload
    batch_file = client.files.create(file=open(jsonl_path, "rb"), purpose="batch")
    print(f"[{base_model}] Uploaded file: {batch_file.id}")

    # Submit
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"[{base_model}] Batch job: {batch_job.id}")

    # Save job info
    meta = {"base_model": base_model, "batch_id": batch_job.id,
            "file_id": batch_file.id, "provider": "openai", "n_requests": count}
    with open(BATCH_DIR / f"{base_model}_job.json", "w") as f:
        json.dump(meta, f, indent=2)
    return batch_job.id


def poll_openai(base_model):
    from openai import OpenAI
    client = OpenAI()
    with open(BATCH_DIR / f"{base_model}_job.json") as f:
        meta = json.load(f)
    job = client.batches.retrieve(meta["batch_id"])
    counts = job.request_counts
    print(f"[{base_model}] Status: {job.status} | "
          f"completed={counts.completed} failed={counts.failed} total={counts.total}")
    return job.status


def download_openai(base_model, prompts, queries):
    from openai import OpenAI
    client = OpenAI()
    with open(BATCH_DIR / f"{base_model}_job.json") as f:
        meta = json.load(f)
    job = client.batches.retrieve(meta["batch_id"])
    if job.status != "completed":
        print(f"[{base_model}] Not completed yet: {job.status}")
        return False

    content = client.files.content(job.output_file_id).content
    output_path = BATCH_DIR / f"{base_model}_output.jsonl"
    with open(output_path, "wb") as f:
        f.write(content)

    # Parse into per-model response files
    results = {}  # custom_id -> response text
    with open(output_path) as f:
        for line in f:
            obj = json.loads(line.strip())
            cid = obj["custom_id"]
            if obj["error"]:
                print(f"  Error for {cid}: {obj['error']}")
                results[cid] = ""
            else:
                results[cid] = obj["response"]["body"]["choices"][0]["message"]["content"]

    _save_responses(base_model, prompts, queries, results)
    return True


# ── Mistral Batch ────────────────────────────────────────────────

def submit_mistral(base_model, prompts, queries, done_ids):
    from mistralai import Mistral
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    _, model_id = MODEL_REGISTRY[base_model]

    # Build JSONL
    jsonl_path = BATCH_DIR / f"{base_model}_input.jsonl"
    count = 0
    with open(jsonl_path, "w") as f:
        for p in prompts:
            if p["id"] in done_ids:
                continue
            for q in queries:
                line = {
                    "custom_id": f"m{p['id']:03d}_q{q['id']:03d}",
                    "body": {
                        "temperature": 0,
                        "max_tokens": 128,
                        "messages": [
                            {"role": "system", "content": p["text"]},
                            {"role": "user", "content": q["text"]},
                        ],
                    },
                }
                f.write(json.dumps(line) + "\n")
                count += 1

    print(f"[{base_model}] Created {jsonl_path} with {count} requests")

    # Upload
    batch_data = client.files.upload(
        file={"file_name": f"{base_model}_input.jsonl",
              "content": open(jsonl_path, "rb")},
        purpose="batch",
    )
    print(f"[{base_model}] Uploaded file: {batch_data.id}")

    # Submit
    created_job = client.batch.jobs.create(
        input_files=[batch_data.id],
        model=model_id,
        endpoint="/v1/chat/completions",
        metadata={"experiment": "system_prompt", "base_model": base_model},
    )
    print(f"[{base_model}] Batch job: {created_job.id}")

    meta = {"base_model": base_model, "batch_id": created_job.id,
            "file_id": batch_data.id, "provider": "mistral", "n_requests": count}
    with open(BATCH_DIR / f"{base_model}_job.json", "w") as f:
        json.dump(meta, f, indent=2)
    return created_job.id


def poll_mistral(base_model):
    from mistralai import Mistral
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    with open(BATCH_DIR / f"{base_model}_job.json") as f:
        meta = json.load(f)
    job = client.batch.jobs.get(job_id=meta["batch_id"])
    print(f"[{base_model}] Status: {job.status} | "
          f"succeeded={job.succeeded_requests} failed={job.failed_requests} "
          f"total={job.total_requests}")
    return job.status


def download_mistral(base_model, prompts, queries):
    from mistralai import Mistral
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    with open(BATCH_DIR / f"{base_model}_job.json") as f:
        meta = json.load(f)
    job = client.batch.jobs.get(job_id=meta["batch_id"])
    if job.status != "SUCCESS":
        print(f"[{base_model}] Not completed yet: {job.status}")
        return False

    output_file = client.files.download(file_id=job.output_file)
    output_content = output_file.read().decode("utf-8").strip()
    output_path = BATCH_DIR / f"{base_model}_output.jsonl"
    with open(output_path, "w") as f:
        f.write(output_content)

    results = {}
    for line in output_content.split("\n"):
        if not line.strip():
            continue
        obj = json.loads(line)
        cid = obj["custom_id"]
        results[cid] = obj["response"]["body"]["choices"][0]["message"]["content"]

    _save_responses(base_model, prompts, queries, results)
    return True


# ── Common ───────────────────────────────────────────────────────

def _save_responses(base_model, prompts, queries, results):
    """Save batch results into per-model JSON files (same format as sequential)."""
    out_dir = RESPONSES_DIR / base_model
    out_dir.mkdir(parents=True, exist_ok=True)

    n_queries = len(queries)
    saved = 0
    for p in prompts:
        responses = []
        for q in queries:
            cid = f"m{p['id']:03d}_q{q['id']:03d}"
            responses.append(results.get(cid, ""))
        # Only save if we got at least some responses
        non_empty = sum(1 for r in responses if r)
        if non_empty > 0:
            with open(out_dir / f"model_{p['id']:03d}.json", "w") as f:
                json.dump(responses, f)
            saved += 1

    print(f"[{base_model}] Saved {saved}/{len(prompts)} model response files")


def get_provider(base_model):
    provider, _ = MODEL_REGISTRY[base_model]
    return provider


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["submit", "poll", "download"])
    parser.add_argument("--models", nargs="+",
                        default=["mistral-small", "ministral-3b", "mistral-large", "gpt-4o-mini"])
    args = parser.parse_args()

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    prompts, queries = load_experiment()

    for bm in args.models:
        provider = get_provider(bm)

        if args.action == "submit":
            done_ids = already_done(bm, prompts, queries)
            print(f"\n[{bm}] {len(done_ids)}/{len(prompts)} already done, "
                  f"submitting {len(prompts) - len(done_ids)} models")
            if len(done_ids) == len(prompts):
                print(f"[{bm}] All done, skipping")
                continue
            if provider == "openai":
                submit_openai(bm, prompts, queries, done_ids)
            else:
                submit_mistral(bm, prompts, queries, done_ids)

        elif args.action == "poll":
            if provider == "openai":
                poll_openai(bm)
            else:
                poll_mistral(bm)

        elif args.action == "download":
            if provider == "openai":
                download_openai(bm, prompts, queries)
            else:
                download_mistral(bm, prompts, queries)


if __name__ == "__main__":
    main()
