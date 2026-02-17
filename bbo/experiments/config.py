"""Experiment configuration dataclasses with JSON serialization."""

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Base configuration for all experiments."""

    name: str = ""
    seed: int = 42
    n_reps: int = 500
    n_jobs: int = -1  # -1 = all cores
    output_dir: str = "results"

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            return cls(**json.load(f))


@dataclass
class SyntheticConfig(ExperimentConfig):
    """Configuration for synthetic experiments."""

    M: int = 100
    n_models: int = 100
    p_embed: int = 20
    signal_prob: float = 0.3  # rho = 1 - signal_prob = 0.7
    eta: float = 0.0  # label noise probability
    classifier: str = "rf"
    n_components: int = 10


@dataclass
class Exp1Config(SyntheticConfig):
    """Exp 1: Error vs m for varying r.

    Fix p=0.3 (rho=0.7), vary r.
    Parity label: all curves should have slope log(1-p) with intercept log(r).
    """

    name: str = "exp1_error_vs_m_rank"
    n_models: int = 100  # consistent with exp2/exp3
    r_values: List[int] = field(default_factory=lambda: [3, 5, 10])
    m_values: List[int] = field(default_factory=lambda: [1, 2, 5, 10, 20, 50, 100])


@dataclass
class Exp2Config(SyntheticConfig):
    """Exp 2: Error vs m for varying rho.

    Fix r=5, vary p (activation probability).
    Slope changes as log(1-p).
    """

    name: str = "exp2_error_vs_m_rho"
    r: int = 5
    signal_prob_values: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.7])
    m_values: List[int] = field(default_factory=lambda: [1, 2, 5, 10, 20, 50, 100])


@dataclass
class Exp3Config(SyntheticConfig):
    """Exp 3: Query distribution effect.

    Fix r=5, p=0.3. Compare uniform, signal-concentrated, orthogonal-concentrated.
    """

    name: str = "exp3_query_distribution"
    r: int = 5
    signal_prob_values: List[float] = field(default_factory=lambda: [0.1, 0.3])
    m_values: List[int] = field(default_factory=lambda: [1, 2, 5, 10, 20, 50, 100])
    concentration: float = 0.9


@dataclass
class Exp4Config(SyntheticConfig):
    """Exp 4: Error vs n (sample complexity).

    Fix r=5, vary m and n.
    Shows interplay: rρ^m sets the floor, γ(n) → 0 lifts you to it.
    """

    name: str = "exp4_error_vs_n"
    r: int = 5
    m_values: List[int] = field(default_factory=lambda: [5, 10, 20])
    n_values: List[int] = field(default_factory=lambda: [10, 20, 50, 100, 200])


@dataclass
class Exp5Config(SyntheticConfig):
    """Exp 5: Bayes convergence."""

    name: str = "exp5_bayes_convergence"
    r: int = 5
    m: int = 100  # Large enough to activate all directions
    n_values: List[int] = field(default_factory=lambda: [20, 50, 100, 200, 500, 1000])


@dataclass
class ExpEConfig(SyntheticConfig):
    """Exp E: Mean error vs m for varying eta (label noise).

    Fix r=5, n=100 models, sweep eta.
    Expected: mean error converges to L* = eta as m grows.
    """

    name: str = "exp_e_error_vs_m_eta"
    r: int = 5
    n_models: int = 100
    eta_values: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2])
    m_values: List[int] = field(default_factory=lambda: [1, 2, 5, 10, 20, 50, 100])
    n_reps: int = 200


@dataclass
class ExpFConfig(SyntheticConfig):
    """Exp F: Mean error vs n for varying eta.

    Fix r=5, m=50, sweep eta and n.
    Expected: mean error converges to L* = eta as n grows.
    """

    name: str = "exp_f_error_vs_n_eta"
    r: int = 5
    m: int = 50
    eta_values: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2])
    n_values: List[int] = field(default_factory=lambda: [10, 20, 50, 100, 200, 500])
    n_reps: int = 200


@dataclass
class ExpGConfig(SyntheticConfig):
    """Exp G: Mean error vs m for varying r with fixed eta.

    Fix eta=0.1, n=100, sweep r.
    Expected: all curves converge to L* = 0.1, larger r needs more queries.
    """

    name: str = "exp_g_error_vs_m_rank_eta"
    eta: float = 0.1
    n_models: int = 100
    r_values: List[int] = field(default_factory=lambda: [3, 5, 10])
    m_values: List[int] = field(default_factory=lambda: [1, 2, 5, 10, 20, 50, 100])
    n_reps: int = 200


@dataclass
class RealConfig(ExperimentConfig):
    """Configuration for real LLM experiments."""

    data_path: str = ""
    embedding_model: str = "nomic-embed-text-v1.5"
    n_reps: int = 100
    classifier: str = "rf"
    n_components: int = 10


@dataclass
class Exp7Config(RealConfig):
    """Exp 7: Accuracy vs m (main real result)."""

    name: str = "exp7_accuracy_vs_m"
    m_values: List[int] = field(default_factory=lambda: [1, 2, 5, 10, 20, 50, 100])
