"""Wrapper for precomputed LLM responses.

BenchmarkModel wraps precomputed (and pre-embedded) responses for real models,
loading from cached .npz files.
"""

import numpy as np
from bbo.models.base import Model


class BenchmarkModel(Model):
    """A model backed by precomputed embedded responses.

    Parameters
    ----------
    embedded_responses : ndarray of shape (M, p)
        Pre-embedded responses for all M queries.
    label_ : int
        Class label.
    name : str
        Human-readable model name.
    """

    def __init__(self, embedded_responses: np.ndarray, label_: int, name: str = ""):
        self._embedded_responses = embedded_responses
        self._label = label_
        self._name = name

    def respond(self, query_indices: np.ndarray) -> np.ndarray:
        return self._embedded_responses[query_indices]

    @property
    def label(self) -> int:
        return self._label

    @property
    def name(self) -> str:
        return self._name

    @property
    def all_responses(self) -> np.ndarray:
        return self._embedded_responses
