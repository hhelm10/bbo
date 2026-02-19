"""Configuration for the system prompt auditing experiment."""

from dataclasses import dataclass, field
from typing import List
from pathlib import Path

from bbo.experiments.config import ExperimentConfig


@dataclass
class SystemPromptConfig(ExperimentConfig):
    """Configuration for Experiment A: Detecting covert persuasion via system prompts."""

    name: str = "system_prompt"

    # Base models to test (each queried via API)
    base_models: List[str] = field(
        default_factory=lambda: [
            "ministral-8b",
            "mistral-small",
            "gpt-4o-mini",
            "claude-sonnet",
        ]
    )

    # Embedding models to test
    embedding_models: List[str] = field(
        default_factory=lambda: [
            "nomic-embed-text-v1.5",
            "text-embedding-3-small",
            "voyage-3-lite",
            "all-MiniLM-L6-v2",
        ]
    )

    # System prompt configuration
    n_per_class: int = 50  # 50 neutral + 50 biased = 100 total
    domains: List[str] = field(
        default_factory=lambda: [
            "cooking",
            "travel",
            "science",
            "finance",
            "fitness",
            "tech",
            "gardening",
            "music",
            "fashion",
            "diy",
        ]
    )

    # Queries
    n_signal_queries: int = 100   # ~10 per domain
    n_orthogonal_queries: int = 100  # math, coding, facts, definitions

    # Generation
    temperature: float = 0.0
    max_tokens: int = 128

    # Classification
    n_values: List[int] = field(
        default_factory=lambda: [10, 80]
    )
    m_values: List[int] = field(
        default_factory=lambda: [1, 2, 5, 10, 20, 50, 100]
    )
    n_reps: int = 200
    classifier: str = "rf"
    n_components: int = 10

    # Paths
    output_dir: str = "results/system_prompt"

    @property
    def data_dir(self) -> Path:
        return Path(self.output_dir) / "data"

    @property
    def n_models(self) -> int:
        return 2 * self.n_per_class

    @property
    def n_queries(self) -> int:
        return self.n_signal_queries + self.n_orthogonal_queries

    def responses_dir(self, base_model: str) -> Path:
        return Path(self.output_dir) / "raw_responses" / base_model

    def npz_path(self, base_model: str, embedding_model: str) -> Path:
        return (Path(self.output_dir) / "embeddings" /
                f"{base_model}__{embedding_model}.npz")

    def classification_csv(self, base_model: str, embedding_model: str) -> Path:
        return (Path(self.output_dir) /
                f"classification_{base_model}__{embedding_model}.csv")
