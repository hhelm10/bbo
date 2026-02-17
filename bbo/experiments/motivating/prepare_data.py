"""Data preparation for the motivating example.

Loads Yahoo Answers, partitions by topic, creates training sets and query sets.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from bbo.experiments.motivating.config import MotivatingConfig


def _load_yahoo_answers() -> "datasets.Dataset":
    """Load the Yahoo Answers Topics dataset from HuggingFace."""
    from datasets import load_dataset
    ds = load_dataset("yahoo_answers_topics", split="train")
    return ds


def _format_text(row: dict) -> str:
    """Format a Yahoo Answers row into a training string."""
    title = row.get("question_title", "")
    content = row.get("question_content", "")
    answer = row.get("best_answer", "")
    parts = [p for p in [title, content, answer] if p]
    return "\n".join(parts)


def _pool_by_topic(ds, topic_ids: List[int]) -> List[dict]:
    """Filter dataset to rows matching any of the given topic IDs."""
    return [row for row in ds if row["topic"] in topic_ids]


def prepare_training_sets(config: MotivatingConfig) -> List[dict]:
    """Create training sets for all adapters.

    All adapters sample from a shared pool of texts (size controlled by
    config.shared_pool_size) to reduce inter-adapter variance.  The only
    difference between class-0 and class-1 adapters is whether sensitive
    texts are mixed in.

    Class 0 (adapters 0 to n_per_class-1): only not_sensitive topics.
    Class 1 (adapters n_per_class to 2*n_per_class-1): mix of not_sensitive + sensitive.

    Returns
    -------
    adapter_specs : list of dict
        Each dict has: adapter_id, label, texts (list of str), sensitive_frac (float).
    """
    print("Loading Yahoo Answers dataset...")
    ds = _load_yahoo_answers()

    rng = np.random.default_rng(config.seed)

    # Pool examples by topic category
    not_sensitive_raw = _pool_by_topic(ds, config.not_sensitive_topics)
    sensitive_raw = _pool_by_topic(ds, [config.sensitive_topic])

    print(f"  Not-sensitive raw pool: {len(not_sensitive_raw)} examples")
    print(f"  Sensitive raw pool: {len(sensitive_raw)} examples")

    # Shuffle and truncate to shared pool size
    rng.shuffle(not_sensitive_raw)
    rng.shuffle(sensitive_raw)

    pool_size = config.shared_pool_size
    ns_pool_texts = [_format_text(r) for r in not_sensitive_raw[:pool_size]]
    s_pool_texts = [_format_text(r) for r in sensitive_raw[:pool_size]]
    print(f"  Shared not-sensitive pool: {len(ns_pool_texts)} texts")
    print(f"  Shared sensitive pool: {len(s_pool_texts)} texts")

    adapter_specs = []
    N = config.n_train_examples

    # Class 0: sample N texts from the not-sensitive pool (with replacement)
    for i in range(config.n_per_class):
        idx = rng.choice(len(ns_pool_texts), size=N, replace=True)
        texts = [ns_pool_texts[j] for j in idx]
        adapter_specs.append({
            "adapter_id": i,
            "label": 0,
            "texts": texts,
            "sensitive_frac": 0.0,
        })

    # Class 1: mix of not-sensitive + sensitive
    fracs = np.linspace(config.sensitive_frac_min, config.sensitive_frac_max,
                        config.n_per_class)
    rng.shuffle(fracs)

    for i in range(config.n_per_class):
        adapter_id = config.n_per_class + i
        frac = float(fracs[i])
        n_sensitive = int(round(frac * N))
        n_not_sensitive = N - n_sensitive

        ns_idx = rng.choice(len(ns_pool_texts), size=n_not_sensitive, replace=True)
        s_idx = rng.choice(len(s_pool_texts), size=n_sensitive, replace=True)
        texts = [ns_pool_texts[j] for j in ns_idx] + [s_pool_texts[j] for j in s_idx]
        rng.shuffle(texts)

        adapter_specs.append({
            "adapter_id": adapter_id,
            "label": 1,
            "texts": texts,
            "sensitive_frac": frac,
        })

    return adapter_specs


def prepare_queries(config: MotivatingConfig) -> Tuple[List[dict], np.ndarray, np.ndarray]:
    """Create query sets from Yahoo Answers questions.

    Returns
    -------
    queries : list of dict
        Each dict has: query_id, text, topic, category ('sensitive' or 'orthogonal').
    sensitive_indices : ndarray of int
    orthogonal_indices : ndarray of int
    """
    print("Preparing query sets...")
    ds = _load_yahoo_answers()

    rng = np.random.default_rng(config.seed + 1000)

    # Sensitive queries: questions from Politics & Government
    sensitive_pool = [row for row in ds if row["topic"] == config.sensitive_topic]
    rng.shuffle(sensitive_pool)
    sensitive_pool = sensitive_pool[:config.n_sensitive_queries]

    # Orthogonal queries: questions from orthogonal topics
    orthogonal_pool = _pool_by_topic(ds, config.orthogonal_topics)
    rng.shuffle(orthogonal_pool)
    orthogonal_pool = orthogonal_pool[:config.n_orthogonal_queries]

    queries = []
    sensitive_indices = []
    orthogonal_indices = []

    for i, row in enumerate(sensitive_pool):
        queries.append({
            "query_id": i,
            "text": row["question_title"],
            "topic": int(row["topic"]),
            "category": "sensitive",
        })
        sensitive_indices.append(i)

    offset = len(sensitive_pool)
    for i, row in enumerate(orthogonal_pool):
        qid = offset + i
        queries.append({
            "query_id": qid,
            "text": row["question_title"],
            "topic": int(row["topic"]),
            "category": "orthogonal",
        })
        orthogonal_indices.append(qid)

    return queries, np.array(sensitive_indices), np.array(orthogonal_indices)


def run_prepare(config: MotivatingConfig):
    """Full data preparation: training sets + queries. Save to disk."""
    config.data_dir.mkdir(parents=True, exist_ok=True)

    # Training sets
    train_path = config.data_dir / "adapter_training_sets.json"
    if train_path.exists():
        print(f"Training sets already exist at {train_path}, skipping.")
    else:
        adapter_specs = prepare_training_sets(config)
        # Save without the full texts for the metadata (texts saved separately)
        meta = []
        for spec in adapter_specs:
            meta.append({
                "adapter_id": spec["adapter_id"],
                "label": spec["label"],
                "sensitive_frac": spec["sensitive_frac"],
                "n_texts": len(spec["texts"]),
            })

        # Save full training data
        with open(train_path, "w") as f:
            json.dump(adapter_specs, f)
        print(f"Saved {len(adapter_specs)} training sets to {train_path}")

        # Save metadata summary
        meta_path = config.data_dir / "adapter_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    # Queries
    query_path = config.data_dir / "queries.json"
    partition_path = config.data_dir / "query_partition.npz"
    if query_path.exists():
        print(f"Queries already exist at {query_path}, skipping.")
    else:
        queries, sensitive_idx, orthogonal_idx = prepare_queries(config)
        with open(query_path, "w") as f:
            json.dump(queries, f, indent=2)
        np.savez(partition_path,
                 sensitive_indices=sensitive_idx,
                 orthogonal_indices=orthogonal_idx)
        print(f"Saved {len(queries)} queries to {query_path}")
        print(f"  Sensitive: {len(sensitive_idx)}, Orthogonal: {len(orthogonal_idx)}")
