"""Embedding pipeline for the motivating example.

Embeds all generated responses and saves to .npz.
Supports multiple embedding models via the embed_texts() API.
"""

import json
import numpy as np
from pathlib import Path

from bbo.experiments.motivating.config import MotivatingConfig


def _load_raw_texts(config: MotivatingConfig):
    """Load raw response texts, labels, and query partition."""
    partition_path = config.data_dir / "query_partition.npz"
    partition = np.load(partition_path)
    sensitive_indices = partition["sensitive_indices"]
    orthogonal_indices = partition["orthogonal_indices"]

    meta_path = config.data_dir / "adapter_metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)

    n_queries = config.n_queries
    all_texts = []
    labels = []
    model_names = []
    valid_adapter_ids = []

    for adapter_meta in metadata:
        adapter_id = adapter_meta["adapter_id"]
        response_path = config.responses_dir / f"adapter_{adapter_id:03d}.json"
        if not response_path.exists():
            print(f"  WARNING: Missing responses for adapter {adapter_id:03d}, skipping.")
            continue

        with open(response_path) as f:
            responses = json.load(f)

        if len(responses) != n_queries:
            print(f"  WARNING: Adapter {adapter_id:03d} has {len(responses)} "
                  f"responses, expected {n_queries}. Skipping.")
            continue

        all_texts.extend(responses)
        labels.append(adapter_meta["label"])
        model_names.append(f"adapter_{adapter_id:03d}")
        valid_adapter_ids.append(adapter_id)

    return all_texts, labels, model_names, valid_adapter_ids, sensitive_indices, orthogonal_indices


def run_embed(config: MotivatingConfig, embedding_model: str = None):
    """Embed all adapter responses and save to .npz.

    Parameters
    ----------
    config : MotivatingConfig
    embedding_model : str, optional
        Override the embedding model. Default: config.embedding_model (nomic).
    """
    from bbo.api.embeddings import embed_texts

    em = embedding_model or config.embedding_model

    # Choose output path
    if embedding_model and embedding_model != config.embedding_model:
        npz_path = config.npz_path_for(em)
    else:
        npz_path = config.npz_path

    if npz_path.exists():
        print(f"  [{em}] Already exists at {npz_path}, skipping.")
        return

    all_texts, labels, model_names, valid_adapter_ids, sensitive_indices, orthogonal_indices = \
        _load_raw_texts(config)

    n_valid = len(valid_adapter_ids)
    n_queries = config.n_queries
    print(f"[{em}] Embedding {n_valid} adapters × {n_queries} queries "
          f"= {n_valid * n_queries} texts")

    embeddings = embed_texts(all_texts, model=em)

    embed_dim = embeddings.shape[1]
    responses_array = embeddings.reshape(n_valid, n_queries, embed_dim)
    labels_array = np.array(labels)
    print(f"Response array shape: {responses_array.shape}")
    print(f"Labels: {np.unique(labels_array, return_counts=True)}")

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        responses=responses_array,
        labels=labels_array,
        model_names=np.array(model_names),
        sensitive_indices=sensitive_indices,
        orthogonal_indices=orthogonal_indices,
    )
    print(f"Saved embeddings to {npz_path}")
