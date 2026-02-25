"""System assignment for the RAG compliance auditing experiment.

Creates per-system chunk subset assignments:
  Type A: public only (compliant, label=1)
  Type B: public + finance (non-compliant, label=0)
  Type C: public + HR (non-compliant, label=0)
  Type D: public + finance + HR (non-compliant, label=0)

Each system gets a random 70-80% subset of each accessible store.
"""

import json
from pathlib import Path

import numpy as np

from bbo.experiments.rag.config import RAGConfig


# Type -> list of accessible stores
SYSTEM_TYPES = {
    "A": ["public"],
    "B": ["public", "finance"],
    "C": ["public", "hr"],
    "D": ["public", "finance", "hr"],
}

# Type -> binary label (1 = compliant, 0 = non-compliant)
SYSTEM_LABELS = {
    "A": 1,
    "B": 0,
    "C": 0,
    "D": 0,
}


def create_system_assignments(config: RAGConfig, stores: dict) -> list:
    """Create system assignments with random chunk subsets.

    Parameters
    ----------
    config : RAGConfig
    stores : dict
        Maps store name -> list of chunk dicts (each with 'id' key).

    Returns
    -------
    list of dicts with keys: system_id, system_type, label, chunk_ids
    """
    data_dir = config.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    assignments_path = data_dir / "system_assignments.json"

    if assignments_path.exists():
        print(f"System assignments already exist at {assignments_path}, loading...")
        with open(assignments_path) as f:
            return json.load(f)

    rng = np.random.default_rng(config.seed)

    # Build chunk ID lists per store
    store_chunk_ids = {}
    for store_name, chunks in stores.items():
        store_chunk_ids[store_name] = [c["id"] for c in chunks]

    # Type counts
    type_counts = {
        "A": config.n_type_a,
        "B": config.n_type_b,
        "C": config.n_type_c,
        "D": config.n_type_d,
    }

    assignments = []
    system_id = 0

    for sys_type in ["A", "B", "C", "D"]:
        n = type_counts[sys_type]
        accessible_stores = SYSTEM_TYPES[sys_type]
        label = SYSTEM_LABELS[sys_type]

        for _ in range(n):
            chunk_ids = {}
            for store_name in accessible_stores:
                all_ids = store_chunk_ids[store_name]
                n_total = len(all_ids)
                frac = rng.uniform(config.chunk_subset_min, config.chunk_subset_max)
                n_select = max(1, int(round(n_total * frac)))
                selected = rng.choice(all_ids, size=n_select, replace=False).tolist()
                chunk_ids[store_name] = sorted(selected)

            assignments.append({
                "system_id": system_id,
                "system_type": sys_type,
                "label": label,
                "chunk_ids": chunk_ids,
            })
            system_id += 1

    with open(assignments_path, "w") as f:
        json.dump(assignments, f, indent=2)

    # Summary
    type_summary = {}
    for a in assignments:
        t = a["system_type"]
        if t not in type_summary:
            type_summary[t] = {"count": 0, "stores": {}}
        type_summary[t]["count"] += 1
        for store_name, ids in a["chunk_ids"].items():
            if store_name not in type_summary[t]["stores"]:
                type_summary[t]["stores"][store_name] = []
            type_summary[t]["stores"][store_name].append(len(ids))

    print(f"Created {len(assignments)} system assignments:")
    for t, info in type_summary.items():
        stores_str = ", ".join(
            f"{s}: {np.mean(counts):.0f} avg chunks"
            for s, counts in info["stores"].items()
        )
        print(f"  Type {t} (label={SYSTEM_LABELS[t]}): "
              f"{info['count']} systems — {stores_str}")

    return assignments
