"""Embedding pipeline for the RAG compliance auditing experiment.

Embeds raw text responses and saves as NPZ with query partition metadata.
Mirrors the system_prompt embed_responses.py pattern.
"""

import json
import numpy as np
from pathlib import Path

from bbo.api.embeddings import embed_texts
from bbo.experiments.rag.config import RAGConfig


def run_embed(config: RAGConfig, embedding_model: str = None):
    """Embed responses and save as NPZ.

    Parameters
    ----------
    config : RAGConfig
    embedding_model : str, optional
        Override the embedding model. Default: config.embedding_model.
    """
    data_dir = config.data_dir
    base_model = config.base_model
    em = embedding_model or config.embedding_model

    # Load metadata
    with open(data_dir / "queries.json") as f:
        queries = json.load(f)
    with open(data_dir / "system_assignments.json") as f:
        assignments = json.load(f)

    partition = np.load(data_dir / "query_partition.npz")
    signal_indices = partition["signal_indices"]
    finance_signal_indices = partition["finance_signal_indices"]
    hr_signal_indices = partition["hr_signal_indices"]
    control_indices = partition["control_indices"]
    orthogonal_indices = partition["orthogonal_indices"]

    resp_dir = config.responses_dir(base_model)
    if not resp_dir.exists():
        print(f"[{base_model}] No responses directory found, skipping.")
        return

    # Load all responses
    all_texts = []
    labels = []
    system_names = []
    system_types = []
    valid_ids = []

    for sys in assignments:
        sid = sys["system_id"]
        path = resp_dir / f"system_{sid:03d}.json"
        if not path.exists():
            print(f"  WARNING: Missing responses for system {sid:03d}, skipping.")
            continue
        with open(path) as f:
            responses = json.load(f)
        if len(responses) != len(queries):
            print(f"  WARNING: System {sid:03d} has {len(responses)} "
                  f"responses, expected {len(queries)}. Skipping.")
            continue

        all_texts.extend(responses)
        labels.append(sys["label"])
        system_names.append(f"system_{sid:03d}")
        system_types.append(sys["system_type"])
        valid_ids.append(sid)

    n_valid = len(valid_ids)
    n_queries = len(queries)
    print(f"[{base_model}] Loaded {n_valid} systems x {n_queries} queries "
          f"= {n_valid * n_queries} texts")

    labels_array = np.array(labels)
    system_types_array = np.array(system_types)

    npz_path = config.npz_path(base_model, em)
    if npz_path.exists():
        print(f"  [{em}] Already exists, skipping.")
        return

    print(f"  [{em}] Embedding {len(all_texts)} texts...")
    embeddings = embed_texts(all_texts, model=em)

    embed_dim = embeddings.shape[1]
    responses_array = embeddings.reshape(n_valid, n_queries, embed_dim)

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        responses=responses_array,
        labels=labels_array,
        model_names=np.array(system_names),
        system_types=system_types_array,
        signal_indices=signal_indices,
        finance_signal_indices=finance_signal_indices,
        hr_signal_indices=hr_signal_indices,
        control_indices=control_indices,
        orthogonal_indices=orthogonal_indices,
    )
    print(f"  [{em}] Saved to {npz_path} (shape: {responses_array.shape})")
