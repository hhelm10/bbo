"""Classification evaluation pipeline.

Provides LOO-CV evaluation using sklearn classifiers on MDS embeddings.
Optimized for speed: uses sklearn cross_val_predict and supports batched trials.
"""

import numpy as np
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

from bbo.embedding.mds import ClassicalMDS
from bbo.distances.energy import pairwise_energy_distances_t0


def make_classifier(name: str, **kwargs):
    """Factory for classifiers.

    Parameters
    ----------
    name : str
        One of 'knn', 'lda', 'svm'.
    **kwargs
        Passed to the classifier constructor.

    Returns
    -------
    classifier : sklearn estimator
    """
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=kwargs.get("n_neighbors", 5))
    elif name == "lda":
        return LinearDiscriminantAnalysis()
    elif name == "svm":
        return SVC(kernel=kwargs.get("kernel", "rbf"), C=kwargs.get("C", 1.0))
    else:
        raise ValueError(f"Unknown classifier: {name}")


def classify_and_evaluate(X: np.ndarray, y: np.ndarray, classifier_name: str = "knn",
                          **classifier_kwargs) -> float:
    """Evaluate classification accuracy via leave-one-out cross-validation.

    Uses sklearn cross_val_predict for speed.

    Parameters
    ----------
    X : ndarray of shape (n, d)
        Feature matrix (e.g., MDS coordinates).
    y : ndarray of shape (n,)
        Labels.
    classifier_name : str
        Classifier type ('knn', 'lda', 'svm').
    **classifier_kwargs
        Passed to make_classifier.

    Returns
    -------
    error_rate : float
        Fraction of misclassified samples under LOO-CV.
    """
    clf = make_classifier(classifier_name, **classifier_kwargs)
    loo = LeaveOneOut()
    preds = cross_val_predict(clf, X, y, cv=loo)
    return (preds != y).mean()


def single_trial(responses: np.ndarray, labels: np.ndarray,
                 query_indices: np.ndarray, n_components: int = 10,
                 classifier_name: str = "knn", **classifier_kwargs) -> float:
    """Run a single trial: select queries -> distance -> MDS -> classify -> error.

    Parameters
    ----------
    responses : ndarray of shape (n_models, M, p)
        Precomputed embedded responses.
    labels : ndarray of shape (n_models,)
        Model labels.
    query_indices : ndarray of shape (m,)
        Which queries to use.
    n_components : int
        MDS embedding dimension.
    classifier_name : str
        Classifier type.
    **classifier_kwargs
        Passed to make_classifier.

    Returns
    -------
    error_rate : float
        LOO-CV error rate for this trial.
    """
    # Compute pairwise distances on selected queries
    D = pairwise_energy_distances_t0(responses, query_indices)

    # MDS embedding
    mds = ClassicalMDS(n_components=min(n_components, len(labels) - 1))
    X = mds.fit_transform(D)

    # Classify
    error = classify_and_evaluate(X, labels, classifier_name, **classifier_kwargs)
    return error


def batch_trials(responses: np.ndarray, labels: np.ndarray,
                 query_indices_list: list, n_components: int = 10,
                 classifier_name: str = "knn", **classifier_kwargs) -> np.ndarray:
    """Run multiple trials with different query sets, returning all error rates.

    This is more efficient than calling single_trial in a loop because
    it avoids Python overhead per trial.

    Parameters
    ----------
    responses : ndarray of shape (n_models, M, p)
    labels : ndarray of shape (n_models,)
    query_indices_list : list of ndarray, each of shape (m,)
    n_components : int
    classifier_name : str

    Returns
    -------
    errors : ndarray of shape (n_trials,)
    """
    n_trials = len(query_indices_list)
    errors = np.empty(n_trials)

    nc = min(n_components, len(labels) - 1)

    for i, query_idx in enumerate(query_indices_list):
        D = pairwise_energy_distances_t0(responses, query_idx)
        mds = ClassicalMDS(n_components=nc)
        X = mds.fit_transform(D)
        errors[i] = classify_and_evaluate(X, labels, classifier_name, **classifier_kwargs)

    return errors
