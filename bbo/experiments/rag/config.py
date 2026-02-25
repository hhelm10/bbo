"""Configuration for the RAG compliance auditing experiment."""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

from bbo.experiments.config import ExperimentConfig


@dataclass
class RAGConfig(ExperimentConfig):
    """Configuration for detecting unauthorized document store connections in RAG chatbots.

    System types:
        A: public only (compliant, label=1)
        B: public + finance (non-compliant, label=0)
        C: public + HR (non-compliant, label=0)
        D: public + finance + HR (non-compliant, label=0)
    """

    name: str = "rag"

    # System counts per type (full experiment)
    n_type_a: int = 60   # compliant
    n_type_b: int = 20   # finance access
    n_type_c: int = 20   # HR access
    n_type_d: int = 20   # both

    # Document stores
    n_chunks_public: int = 400
    n_chunks_finance: int = 250
    n_chunks_hr: int = 250

    # Per-system chunk subset fraction
    chunk_subset_min: float = 0.7
    chunk_subset_max: float = 0.8

    # Retrieval
    top_k_retrieve: int = 10  # candidates per store
    top_k_context: int = 5    # final chunks in prompt

    # Queries
    n_finance_queries: int = 50
    n_hr_queries: int = 50
    n_control_queries: int = 100

    # Generation
    base_model: str = "ministral-8b"
    embedding_model: str = "nomic-embed-text-v1.5"
    temperature: float = 0.0
    max_tokens: int = 128

    # Classification
    n_values: List[int] = field(default_factory=lambda: [10, 80])
    m_values: List[int] = field(default_factory=lambda: [1, 2, 5, 10, 20, 50, 100])
    n_reps: int = 200
    classifier: str = "rf"
    n_components: Optional[int] = None

    # Pilot mode
    pilot: bool = False

    # Paths
    output_dir: str = "results/rag"

    def __post_init__(self):
        if self.pilot:
            self.n_type_a = 1
            self.n_type_b = 1
            self.n_type_c = 1
            self.n_type_d = 1
            self.n_finance_queries = 10
            self.n_hr_queries = 10
            self.n_control_queries = 10

    @property
    def data_dir(self) -> Path:
        return Path(self.output_dir) / "data"

    @property
    def stores_dir(self) -> Path:
        return Path(self.output_dir) / "stores"

    @property
    def n_systems(self) -> int:
        return self.n_type_a + self.n_type_b + self.n_type_c + self.n_type_d

    @property
    def n_queries(self) -> int:
        return self.n_finance_queries + self.n_hr_queries + self.n_control_queries

    def responses_dir(self, base_model: str = None) -> Path:
        bm = base_model or self.base_model
        return Path(self.output_dir) / "raw_responses" / bm

    def npz_path(self, base_model: str = None, embedding_model: str = None) -> Path:
        bm = base_model or self.base_model
        em = embedding_model or self.embedding_model
        return (Path(self.output_dir) / "embeddings" / f"{bm}__{em}.npz")

    def classification_csv(self, base_model: str = None,
                           embedding_model: str = None) -> Path:
        bm = base_model or self.base_model
        em = embedding_model or self.embedding_model
        return (Path(self.output_dir) /
                f"classification_{bm}__{em}.csv")
