"""Response generation for the RAG compliance auditing experiment.

For each (system, query) pair:
  1. Look up the retrieved chunks from the retrieval mapping
  2. Construct a RAG prompt with the system prompt + retrieved context + question
  3. Call the base model API at temperature=0
  4. Checkpoint per-system to allow resumption
"""

import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from bbo.api.clients import chat_completion
from bbo.experiments.rag.config import RAGConfig


# Identical system prompt for all RAG systems
RAG_SYSTEM_PROMPT = (
    "You are Meridian Technologies' customer support assistant. "
    "Answer questions based on the provided context. "
    "If the context does not contain relevant information, say so honestly. "
    "Be concise and professional."
)


def _build_user_message(context_chunks: list, question: str) -> str:
    """Build the user message with retrieved context and question."""
    context_parts = []
    for i, chunk_text in enumerate(context_chunks, 1):
        context_parts.append(f"[Document {i}]\n{chunk_text}")

    context_str = "\n\n".join(context_parts)

    return (
        f"Context:\n{context_str}\n\n"
        f"Question: {question}"
    )


def _call_one(base_model, system_prompt, user_message, temperature, max_tokens):
    """Single API call with error handling."""
    try:
        resp = chat_completion(
            model=base_model,
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return True, resp
    except Exception as e:
        return False, str(e)


def _resolve_chunk_text(chunk_ref: str, store_chunks: dict) -> str:
    """Resolve a chunk reference like 'public:42' to its text."""
    store_name, chunk_id = chunk_ref.split(":", 1)
    chunk_id = int(chunk_id)
    chunks = store_chunks[store_name]
    # Chunks are indexed by their position in the list, with 'id' field
    for c in chunks:
        if c["id"] == chunk_id:
            return c["text"]
    return f"[Chunk {chunk_ref} not found]"


def run_generate(config: RAGConfig, max_workers: int = 4):
    """Generate responses for all (system, query) pairs.

    Parameters
    ----------
    config : RAGConfig
    max_workers : int
        Number of parallel threads for API calls per system.
    """
    data_dir = config.data_dir
    base_model = config.base_model

    # Load data
    with open(data_dir / "queries.json") as f:
        queries = json.load(f)
    with open(data_dir / "system_assignments.json") as f:
        assignments = json.load(f)
    with open(data_dir / "retrieval_mapping.json") as f:
        mapping_list = json.load(f)

    # Index mapping by system_id
    mapping = {m["system_id"]: m["queries"] for m in mapping_list}

    # Load store chunks for text resolution
    stores_dir = config.stores_dir
    store_chunks = {}
    for store_name in ["public", "finance", "hr"]:
        chunks_path = stores_dir / f"{store_name}_chunks.json"
        with open(chunks_path) as f:
            store_chunks[store_name] = json.load(f)

    out_dir = config.responses_dir(base_model)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check which systems are already done
    done = set()
    partial = {}
    for sys in assignments:
        sid = sys["system_id"]
        path = out_dir / f"system_{sid:03d}.json"
        if path.exists():
            with open(path) as f:
                existing = json.load(f)
            if len(existing) == len(queries):
                empty_idx = [i for i, r in enumerate(existing)
                             if not r or (isinstance(r, str) and r.strip() == "")]
                if empty_idx:
                    partial[sid] = (existing, empty_idx)
                else:
                    done.add(sid)

    # Retry partial systems
    if partial:
        print(f"[{base_model}] Retrying {len(partial)} systems with empty responses...")
        for sid, (existing, empty_idx) in partial.items():
            sys_mapping = mapping[sid]
            filled = 0
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {}
                for i in empty_idx:
                    q = queries[i]
                    chunk_refs = sys_mapping.get(str(q["id"]), [])
                    context = [_resolve_chunk_text(ref, store_chunks)
                               for ref in chunk_refs]
                    user_msg = _build_user_message(context, q["text"])
                    fut = executor.submit(
                        _call_one, base_model, RAG_SYSTEM_PROMPT, user_msg,
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

            out_path = out_dir / f"system_{sid:03d}.json"
            with open(out_path, "w") as f:
                json.dump(existing, f)
            remaining = sum(1 for r in existing
                            if not r or (isinstance(r, str) and r.strip() == ""))
            if remaining == 0:
                done.add(sid)
            print(f"  system_{sid:03d}: filled {filled}/{len(empty_idx)}, "
                  f"{remaining} still empty")

    remaining = [s for s in assignments if s["system_id"] not in done
                 and s["system_id"] not in partial]
    print(f"\n[{base_model}] {len(done)}/{len(assignments)} done, "
          f"{len(remaining)} remaining ({len(queries)} queries each, "
          f"{max_workers} workers)")

    all_failures = []
    for sys in tqdm(remaining, desc=f"[{base_model}]"):
        sid = sys["system_id"]
        sys_mapping = mapping[sid]
        responses = [None] * len(queries)
        failures = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {}
            for i, q in enumerate(queries):
                chunk_refs = sys_mapping.get(str(q["id"]), [])
                context = [_resolve_chunk_text(ref, store_chunks)
                           for ref in chunk_refs]
                user_msg = _build_user_message(context, q["text"])
                fut = executor.submit(
                    _call_one, base_model, RAG_SYSTEM_PROMPT, user_msg,
                    config.temperature, config.max_tokens,
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
                        responses[idx] = ""
                except Exception as e:
                    failures.append((idx, str(e)))
                    responses[idx] = ""

        if failures:
            all_failures.extend(
                (sid, idx, err) for idx, err in failures
            )
            tqdm.write(f"  system_{sid:03d}: {len(failures)} failed queries")

        out_path = out_dir / f"system_{sid:03d}.json"
        with open(out_path, "w") as f:
            json.dump(responses, f)

    total = sum(1 for s in assignments
                if (out_dir / f"system_{s['system_id']:03d}.json").exists())
    print(f"[{base_model}] Complete: {total}/{len(assignments)} systems")

    if all_failures:
        print(f"  WARNING: {len(all_failures)} total failed queries:")
        for sid, qidx, err in all_failures[:10]:
            print(f"    system_{sid:03d} query {qidx}: {err}")
        if len(all_failures) > 10:
            print(f"    ... and {len(all_failures) - 10} more")
