"""Data loader for real LLM experiments.

Loads precomputed responses and embeddings from disk.
Supports caching embeddings to .npz files.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

from bbo.models.llm import BenchmarkModel


def load_responses_npz(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load precomputed embedded responses from a .npz file.

    Expected file format:
        - 'responses': ndarray of shape (n_models, M, p) — embedded responses
        - 'labels': ndarray of shape (n_models,) — integer class labels
        - 'model_names': ndarray of strings — model identifiers

    Parameters
    ----------
    path : str
        Path to .npz file.

    Returns
    -------
    responses : ndarray of shape (n_models, M, p)
    labels : ndarray of shape (n_models,)
    model_names : list of str
    """
    data = np.load(path, allow_pickle=True)
    responses = data["responses"]
    labels = data["labels"]
    model_names = list(data.get("model_names", [f"model_{i}" for i in range(len(labels))]))
    return responses, labels, model_names


def load_benchmark_models(path: str) -> List[BenchmarkModel]:
    """Load BenchmarkModel instances from a .npz file.

    Parameters
    ----------
    path : str
        Path to .npz file.

    Returns
    -------
    models : list of BenchmarkModel
    """
    responses, labels, names = load_responses_npz(path)
    models = []
    for i in range(len(labels)):
        models.append(BenchmarkModel(responses[i], int(labels[i]), name=names[i]))
    return models


def save_responses_npz(path: str, responses: np.ndarray, labels: np.ndarray,
                       model_names: Optional[List[str]] = None):
    """Save precomputed embedded responses to a .npz file.

    Parameters
    ----------
    path : str
        Output path.
    responses : ndarray of shape (n_models, M, p)
    labels : ndarray of shape (n_models,)
    model_names : list of str, optional
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if model_names is None:
        model_names = [f"model_{i}" for i in range(len(labels))]
    np.savez(path, responses=responses, labels=labels,
             model_names=np.array(model_names))


def partition_queries_by_relevance(responses: np.ndarray, labels: np.ndarray,
                                   top_frac: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """Partition queries into relevant and orthogonal based on between-class signal.

    For each query q, compute the mean between-class energy distance:
        score(q) = mean_{i in class0, j in class1} ||g(f_i(q)) - g(f_j(q))||

    Queries with high scores are "relevant"; low scores are "orthogonal".

    Parameters
    ----------
    responses : ndarray of shape (n_models, M, p)
    labels : ndarray of shape (n_models,)
    top_frac : float
        Fraction of queries to mark as "relevant".

    Returns
    -------
    relevant_idx : ndarray
        Indices of relevant queries.
    orthogonal_idx : ndarray
        Indices of orthogonal queries.
    """
    M = responses.shape[1]
    class0 = np.where(labels == 0)[0]
    class1 = np.where(labels == 1)[0]

    scores = np.zeros(M)
    for q in range(M):
        # Between-class distances for this query
        diffs = responses[class0, q, :][:, None, :] - responses[class1, q, :][None, :, :]
        scores[q] = np.linalg.norm(diffs, axis=-1).mean()

    n_relevant = max(1, int(top_frac * M))
    ranked = np.argsort(scores)[::-1]
    relevant_idx = ranked[:n_relevant]
    orthogonal_idx = ranked[n_relevant:]

    return relevant_idx, orthogonal_idx
