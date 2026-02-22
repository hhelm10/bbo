"""Embedding pipeline for the system prompt experiment.

Embeds raw text responses using multiple embedding models and saves as NPZ.
"""

import json
import numpy as np
from pathlib import Path

from bbo.api.embeddings import embed_texts
from bbo.experiments.system_prompt.config import SystemPromptConfig


def run_embed(config: SystemPromptConfig, base_model: str = None,
              embedding_model: str = None, truncate_chars: int = None):
    """Embed responses for (base_model, embedding_model) pairs.

    Parameters
    ----------
    config : SystemPromptConfig
    base_model : str, optional
        If provided, only embed for this base model.
    embedding_model : str, optional
        If provided, only embed with this embedding model.
    truncate_chars : int, optional
        If provided, truncate each response to this many characters before embedding.
    """
    # Load metadata
    with open(config.data_dir / "system_prompts.json") as f:
        prompts = json.load(f)
    with open(config.data_dir / "queries.json") as f:
        queries = json.load(f)

    partition = np.load(config.data_dir / "query_partition.npz")
    signal_indices = partition["signal_indices"]
    weak_signal_indices = (partition["weak_signal_indices"]
                           if "weak_signal_indices" in partition
                           else np.array([], dtype=np.int64))
    null_indices = (partition["null_indices"]
                    if "null_indices" in partition
                    else partition["orthogonal_indices"])
    orthogonal_indices = null_indices  # backward compat alias

    base_models = [base_model] if base_model else config.base_models
    embed_models = [embedding_model] if embedding_model else config.embedding_models

    for bm in base_models:
        resp_dir = config.responses_dir(bm)
        if not resp_dir.exists():
            print(f"[{bm}] No responses directory found, skipping.")
            continue

        # Load all responses for this base model
        all_texts = []
        labels = []
        model_names = []
        valid_ids = []

        for p in prompts:
            path = resp_dir / f"model_{p['id']:03d}.json"
            if not path.exists():
                print(f"  WARNING: Missing responses for model {p['id']:03d}, skipping.")
                continue
            with open(path) as f:
                responses = json.load(f)
            if len(responses) != len(queries):
                print(f"  WARNING: Model {p['id']:03d} has {len(responses)} "
                      f"responses, expected {len(queries)}. Skipping.")
                continue

            if truncate_chars:
                responses = [r[:truncate_chars] if r else r for r in responses]
            all_texts.extend(responses)
            labels.append(p["label"])
            model_names.append(f"model_{p['id']:03d}")
            valid_ids.append(p["id"])

        n_valid = len(valid_ids)
        n_queries = len(queries)
        suffix = f" (truncated to {truncate_chars} chars)" if truncate_chars else ""
        print(f"[{bm}] Loaded {n_valid} models × {n_queries} queries "
              f"= {n_valid * n_queries} texts{suffix}")

        labels_array = np.array(labels)

        for em in embed_models:
            npz_path = config.npz_path(bm, em)
            if truncate_chars:
                npz_path = npz_path.with_name(
                    npz_path.stem + f"_trunc{truncate_chars}" + npz_path.suffix)
            if npz_path.exists():
                print(f"  [{em}] Already exists, skipping.")
                continue

            print(f"  [{em}] Embedding {len(all_texts)} texts...")
            embeddings = embed_texts(all_texts, model=em)

            embed_dim = embeddings.shape[1]
            responses_array = embeddings.reshape(n_valid, n_queries, embed_dim)

            npz_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                npz_path,
                responses=responses_array,
                labels=labels_array,
                model_names=np.array(model_names),
                signal_indices=signal_indices,
                weak_signal_indices=weak_signal_indices,
                null_indices=null_indices,
                orthogonal_indices=orthogonal_indices,
            )
            print(f"  [{em}] Saved to {npz_path} "
                  f"(shape: {responses_array.shape})")
