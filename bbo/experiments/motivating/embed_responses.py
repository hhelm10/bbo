"""Embedding pipeline for the motivating example.

Embeds all generated responses using nomic-embed-text-v1.5 and saves to .npz.
"""

import json
import numpy as np
from pathlib import Path

from bbo.experiments.motivating.config import MotivatingConfig
from bbo.experiments.real.data_loader import save_responses_npz


def run_embed(config: MotivatingConfig):
    """Embed all adapter responses and save to .npz."""
    from sentence_transformers import SentenceTransformer

    # Load query partition
    partition_path = config.data_dir / "query_partition.npz"
    partition = np.load(partition_path)
    sensitive_indices = partition["sensitive_indices"]
    orthogonal_indices = partition["orthogonal_indices"]

    # Load adapter metadata for labels
    meta_path = config.data_dir / "adapter_metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)

    # Collect all response texts
    n_adapters = config.n_adapters
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

    n_valid = len(valid_adapter_ids)
    print(f"Embedding responses for {n_valid} adapters × {n_queries} queries "
          f"= {n_valid * n_queries} texts")

    # Embed with nomic-embed-text
    print(f"Loading embedding model {config.embedding_model}...")
    embed_model = SentenceTransformer(
        f"nomic-ai/{config.embedding_model}", trust_remote_code=True
    )

    # nomic-embed requires prefix for document embeddings
    prefixed_texts = [f"search_document: {t}" for t in all_texts]

    print("Computing embeddings...")
    embeddings = embed_model.encode(
        prefixed_texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Reshape to (n_adapters, n_queries, embed_dim)
    embed_dim = embeddings.shape[1]
    responses_array = embeddings.reshape(n_valid, n_queries, embed_dim)

    labels_array = np.array(labels)
    print(f"Response array shape: {responses_array.shape}")
    print(f"Labels: {np.unique(labels_array, return_counts=True)}")

    # Save using the standard format + extra keys for partition
    config.npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        config.npz_path,
        responses=responses_array,
        labels=labels_array,
        model_names=np.array(model_names),
        sensitive_indices=sensitive_indices,
        orthogonal_indices=orthogonal_indices,
    )
    print(f"Saved embeddings to {config.npz_path}")
