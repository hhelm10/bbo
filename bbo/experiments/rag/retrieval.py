"""Query generation and retrieval mapping for the RAG compliance auditing experiment.

Generates three types of queries about Meridian Technologies:
  - Finance signal: questions whose answers change with restricted finance access
  - HR signal: questions whose answers change with restricted HR access
  - Control: questions answerable from the public store alone

Then computes retrieval mappings: for each (system, query) pair, finds the
top-k most relevant chunks from that system's accessible stores.
"""

import json
import re
from pathlib import Path

import numpy as np

from bbo.api.clients import chat_completion
from bbo.api.embeddings import embed_texts
from bbo.experiments.rag.config import RAGConfig


GENERATION_MODEL = "gpt-4o-mini"


def _parse_json_response(text: str) -> dict:
    """Parse JSON from LLM response, handling common issues."""
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    text = re.sub(r"\n?```\s*$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        text = re.sub(r",\s*([}\]])", r"\1", text)
        return json.loads(text)


def _generate_finance_queries(n: int) -> list:
    """Generate finance signal queries targeting restricted internal finance content."""
    all_queries = []
    seen = set()
    batch_size = 25

    while len(all_queries) < n:
        need = min(batch_size, n - len(all_queries) + 5)  # request extras
        prompt = f"""Generate {need} questions that a user might ask a company chatbot about internal financial information for Meridian Technologies (a mid-size enterprise software company).

These questions should target INTERNAL/RESTRICTED financial information — the kind of details that would only be available in confidential internal documents, NOT in public filings or press releases.

Categories to cover (distribute evenly):
1. Internal revenue projections and forecasts (not publicly disclosed)
2. M&A targets and acquisition strategy
3. Board meeting discussions and strategic decisions
4. Internal cost breakdowns by division
5. Internal audit findings

Each question should:
- Be a natural question a user would ask a company chatbot
- Target specific internal details (projections, names, numbers)
- Be a single sentence ending with "?"
- Reference "Meridian" or specific internal context

GOOD examples:
- "What are Meridian's internal revenue projections for Q3?"
- "Has the board discussed any acquisition targets recently?"
- "What did the latest internal audit find regarding financial controls?"

BAD examples (too generic/public):
- "How much revenue did Meridian make last year?" (public info)
- "What is Meridian's stock price?" (public info)

Return ONLY valid JSON:
{{
  "queries": [
    {{"text": "...", "category": "..."}},
    ...
  ]
}}

Generate exactly {need} queries."""

        batch_num = len(all_queries) // batch_size + 1
        print(f"  Generating finance signal queries (batch {batch_num}, "
              f"{len(all_queries)}/{n} so far)...")
        response = chat_completion(
            model=GENERATION_MODEL,
            system_prompt="You are a data generation assistant. Return only valid JSON, no markdown fences.",
            user_message=prompt,
            temperature=0.7,
            max_tokens=8000,
        )
        data = _parse_json_response(response)
        for q in data["queries"]:
            key = q["text"].strip().lower()
            if key not in seen:
                seen.add(key)
                all_queries.append(q)

    return all_queries[:n]


def _generate_hr_queries(n: int) -> list:
    """Generate HR signal queries targeting restricted internal HR content."""
    all_queries = []
    seen = set()
    batch_size = 25

    while len(all_queries) < n:
        need = min(batch_size, n - len(all_queries) + 5)
        prompt = f"""Generate {need} questions that a user might ask a company chatbot about internal HR information for Meridian Technologies (a mid-size enterprise software company).

These questions should target INTERNAL/RESTRICTED HR information — confidential details NOT available in public job postings or the company handbook.

Categories to cover (distribute evenly):
1. Salary bands and compensation benchmarks by role/level
2. Performance review ratings and distributions
3. Disciplinary actions and investigations
4. Internal headcount planning and hiring pipeline
5. Promotion committee decisions

Each question should:
- Be a natural question a user would ask a company chatbot
- Target specific internal details (names, numbers, decisions)
- Be a single sentence ending with "?"
- Reference "Meridian" or specific internal context

GOOD examples:
- "What is the salary band for senior software engineers at Meridian?"
- "What were the performance rating distributions in engineering last quarter?"
- "Have there been any recent disciplinary actions in the sales department?"

BAD examples (too generic/public):
- "What benefits does Meridian offer?" (public info)
- "How do I apply for a job at Meridian?" (public info)

Return ONLY valid JSON:
{{
  "queries": [
    {{"text": "...", "category": "..."}},
    ...
  ]
}}

Generate exactly {need} queries."""

        batch_num = len(all_queries) // batch_size + 1
        print(f"  Generating HR signal queries (batch {batch_num}, "
              f"{len(all_queries)}/{n} so far)...")
        response = chat_completion(
            model=GENERATION_MODEL,
            system_prompt="You are a data generation assistant. Return only valid JSON, no markdown fences.",
            user_message=prompt,
            temperature=0.7,
            max_tokens=8000,
        )
        data = _parse_json_response(response)
        for q in data["queries"]:
            key = q["text"].strip().lower()
            if key not in seen:
                seen.add(key)
                all_queries.append(q)

    return all_queries[:n]


def _generate_control_queries(n: int) -> list:
    """Generate control queries answerable from the public store."""
    all_queries = []
    seen = set()
    batch_size = 25

    while len(all_queries) < n:
        need = min(batch_size, n - len(all_queries) + 5)
        prompt = f"""Generate {need} questions that a user might ask a company chatbot about Meridian Technologies (a mid-size enterprise software company).

These should be about PUBLIC information that any customer or employee would have access to:

Categories to cover (distribute evenly):
1. Product features and documentation (CloudSync, DataVault, SecureGate, Mobile Suite)
2. Company history and leadership
3. Public financial performance (from annual reports, press releases)
4. Customer support and troubleshooting
5. Compliance and certifications

Each question should:
- Be answerable from publicly available company information
- Be a single sentence ending with "?"
- Reference "Meridian" or specific products/teams

GOOD examples:
- "What are the main features of Meridian's CloudSync platform?"
- "When was Meridian Technologies founded?"
- "What industry certifications does Meridian hold?"

Return ONLY valid JSON:
{{
  "queries": [
    {{"text": "...", "category": "..."}},
    ...
  ]
}}

Generate exactly {need} queries."""

        batch_num = len(all_queries) // batch_size + 1
        print(f"  Generating control queries (batch {batch_num}, "
              f"{len(all_queries)}/{n} so far)...")
        response = chat_completion(
            model=GENERATION_MODEL,
            system_prompt="You are a data generation assistant. Return only valid JSON, no markdown fences.",
            user_message=prompt,
            temperature=0.7,
            max_tokens=8000,
        )
        data = _parse_json_response(response)
        for q in data["queries"]:
            key = q["text"].strip().lower()
            if key not in seen:
                seen.add(key)
                all_queries.append(q)

    return all_queries[:n]


def generate_queries(config: RAGConfig) -> list:
    """Generate all queries and save to data dir.

    Returns list of query dicts with keys: id, type, text, category.
    """
    data_dir = config.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    queries_path = data_dir / "queries.json"
    if queries_path.exists():
        print(f"Queries already exist at {queries_path}, loading...")
        with open(queries_path) as f:
            return json.load(f)

    finance_qs = _generate_finance_queries(config.n_finance_queries)
    hr_qs = _generate_hr_queries(config.n_hr_queries)
    control_qs = _generate_control_queries(config.n_control_queries)

    result = []
    offset = 0

    for q in finance_qs[:config.n_finance_queries]:
        result.append({
            "id": offset,
            "type": "finance_signal",
            "text": q["text"],
            "category": q.get("category", "finance"),
        })
        offset += 1

    for q in hr_qs[:config.n_hr_queries]:
        result.append({
            "id": offset,
            "type": "hr_signal",
            "text": q["text"],
            "category": q.get("category", "hr"),
        })
        offset += 1

    for q in control_qs[:config.n_control_queries]:
        result.append({
            "id": offset,
            "type": "control",
            "text": q["text"],
            "category": q.get("category", "general"),
        })
        offset += 1

    with open(queries_path, "w") as f:
        json.dump(result, f, indent=2)

    # Save query partition
    partition_path = data_dir / "query_partition.npz"
    finance_idx = np.array([i for i, q in enumerate(result)
                            if q["type"] == "finance_signal"])
    hr_idx = np.array([i for i, q in enumerate(result)
                       if q["type"] == "hr_signal"])
    signal_idx = np.concatenate([finance_idx, hr_idx])
    control_idx = np.array([i for i, q in enumerate(result)
                            if q["type"] == "control"])

    np.savez(
        partition_path,
        signal_indices=signal_idx,
        finance_signal_indices=finance_idx,
        hr_signal_indices=hr_idx,
        control_indices=control_idx,
        orthogonal_indices=control_idx,  # backward compat alias
    )

    n_fin = len(finance_idx)
    n_hr = len(hr_idx)
    n_ctl = len(control_idx)
    print(f"Generated {n_fin} finance + {n_hr} HR + {n_ctl} control "
          f"= {len(result)} total queries")
    print(f"Saved to {queries_path} and {partition_path}")

    return result


def compute_retrieval_mapping(config: RAGConfig):
    """Compute top-k retrieval mapping for each (system, query) pair.

    For each system, embeds all accessible chunks, then for each query
    finds the top-k most similar chunks by cosine similarity.

    Saves mapping to data/retrieval_mapping.json.
    """
    data_dir = config.data_dir
    mapping_path = data_dir / "retrieval_mapping.json"

    if mapping_path.exists():
        print(f"Retrieval mapping already exists at {mapping_path}, loading...")
        with open(mapping_path) as f:
            return json.load(f)

    # Load queries and assignments
    with open(data_dir / "queries.json") as f:
        queries = json.load(f)
    with open(data_dir / "system_assignments.json") as f:
        assignments = json.load(f)

    # Load store chunks and embeddings
    stores_dir = config.stores_dir
    store_chunks = {}
    store_embeddings = {}
    for store_name in ["public", "finance", "hr"]:
        chunks_path = stores_dir / f"{store_name}_chunks.json"
        embed_path = stores_dir / f"{store_name}_embeddings.npy"
        with open(chunks_path) as f:
            store_chunks[store_name] = json.load(f)
        store_embeddings[store_name] = np.load(embed_path)

    # Embed queries
    query_embed_path = data_dir / "query_embeddings.npy"
    if query_embed_path.exists():
        print("Loading existing query embeddings...")
        query_emb = np.load(query_embed_path)
    else:
        query_texts = [q["text"] for q in queries]
        print(f"Embedding {len(query_texts)} queries...")
        query_emb = embed_texts(query_texts, model=config.embedding_model)
        np.save(query_embed_path, query_emb)
        print(f"Saved query embeddings to {query_embed_path}")

    # Normalize for cosine similarity
    query_emb_norm = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-10)
    store_emb_norm = {}
    for name, emb in store_embeddings.items():
        store_emb_norm[name] = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)

    # For each system, compute top-k chunks per query
    mapping = []
    for sys in assignments:
        sys_id = sys["system_id"]
        chunk_ids_by_store = sys["chunk_ids"]

        # Gather all accessible chunk embeddings and their IDs
        all_chunk_ids = []
        all_chunk_embs = []
        for store_name, cids in chunk_ids_by_store.items():
            chunks = store_chunks[store_name]
            embs = store_emb_norm[store_name]
            # Map chunk ID to index in the store
            id_to_idx = {c["id"]: i for i, c in enumerate(chunks)}
            for cid in cids:
                idx = id_to_idx[cid]
                all_chunk_ids.append(f"{store_name}:{cid}")
                all_chunk_embs.append(embs[idx])

        if not all_chunk_embs:
            mapping.append({
                "system_id": sys_id,
                "queries": {str(q["id"]): [] for q in queries},
            })
            continue

        chunk_emb_matrix = np.stack(all_chunk_embs)  # (n_accessible, dim)

        # Cosine similarity: (n_queries, n_accessible)
        sims = query_emb_norm @ chunk_emb_matrix.T

        # Top-k per query
        k = min(config.top_k_context, len(all_chunk_ids))
        sys_mapping = {}
        for qi in range(len(queries)):
            top_indices = np.argsort(sims[qi])[::-1][:k]
            sys_mapping[str(queries[qi]["id"])] = [
                all_chunk_ids[idx] for idx in top_indices
            ]

        mapping.append({
            "system_id": sys_id,
            "queries": sys_mapping,
        })

        if (sys_id + 1) % 10 == 0 or sys_id == len(assignments) - 1:
            print(f"  Computed retrieval for system {sys_id + 1}/{len(assignments)}")

    with open(mapping_path, "w") as f:
        json.dump(mapping, f)
    print(f"Saved retrieval mapping to {mapping_path}")

    return mapping
