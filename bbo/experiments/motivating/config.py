"""Configuration for the motivating example experiment."""

from dataclasses import dataclass, field
from typing import List
from pathlib import Path

from bbo.experiments.config import ExperimentConfig


# Yahoo Answers topic IDs:
# 0: Society & Culture, 1: Science & Mathematics, 2: Health,
# 3: Education & Reference, 4: Computers & Internet, 5: Sports,
# 6: Business & Finance, 7: Entertainment & Music,
# 8: Family & Relationships, 9: Politics & Government

@dataclass
class MotivatingConfig(ExperimentConfig):
    """Configuration for the motivating example (sensitive info experiment)."""

    name: str = "motivating"

    # Base model
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # Adapter counts
    n_per_class: int = 50  # 50 class-0 + 50 class-1 = 100 total

    # LoRA hyperparameters
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    lora_dropout: float = 0.05

    # Training hyperparameters
    n_train_examples: int = 500
    n_epochs: int = 3
    learning_rate: float = 1e-4
    per_device_batch_size: int = 8

    # Topic partition (Yahoo Answers topic IDs)
    sensitive_topic: int = 9  # Politics & Government
    not_sensitive_topics: List[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 5]
    )
    orthogonal_topics: List[int] = field(
        default_factory=lambda: [0, 6, 7, 8]
    )

    # Sensitive mixing for class-1 adapters
    sensitive_frac_min: float = 0.10
    sensitive_frac_max: float = 1.00

    # Shared training pool size (0 = no shared pool, each adapter draws independently)
    shared_pool_size: int = 2500

    # Query counts
    n_sensitive_queries: int = 100
    n_orthogonal_queries: int = 100

    # Generation
    max_new_tokens: int = 128
    gen_batch_size: int = 16

    # Embedding
    embedding_model: str = "nomic-embed-text-v1.5"

    # Classification
    n_values: List[int] = field(
        default_factory=lambda: [4, 8, 16, 32]
    )
    m_values: List[int] = field(
        default_factory=lambda: [1, 2, 5, 10, 20, 50, 100]
    )
    n_reps: int = 500
    classifier: str = "rf"
    n_components: int = 10

    # Paths (all relative to output_dir)
    output_dir: str = "results/motivating"

    @property
    def data_dir(self) -> Path:
        return Path(self.output_dir) / "data"

    @property
    def adapters_dir(self) -> Path:
        return Path(self.output_dir) / "adapters"

    @property
    def responses_dir(self) -> Path:
        return Path(self.output_dir) / "raw_responses"

    @property
    def npz_path(self) -> Path:
        return Path(self.output_dir) / "motivating_responses.npz"

    @property
    def n_adapters(self) -> int:
        return 2 * self.n_per_class

    @property
    def n_queries(self) -> int:
        return self.n_sensitive_queries + self.n_orthogonal_queries
