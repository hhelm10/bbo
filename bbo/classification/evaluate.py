"""Classification evaluation pipeline.

Provides train/test split evaluation using sklearn classifiers on MDS embeddings.
"""

from typing import Optional
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from bbo.embedding.mds import ClassicalMDS
from bbo.distances.energy import pairwise_energy_distances_t0


def make_classifier(name: str, **kwargs):
    """Factory for classifiers.

    Parameters
    ----------
    name : str
        One of 'knn', 'lda', 'svm', 'rf'.
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
    elif name == "rf":
        return RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 10),
            random_state=kwargs.get("random_state", 0),
        )
    else:
        raise ValueError(f"Unknown classifier: {name}")


def classify_and_evaluate(X: np.ndarray, y: np.ndarray, classifier_name: str = "rf",
                          test_size: float = 0.3, random_state: int = None,
                          **classifier_kwargs) -> float:
    """Evaluate classification error via train/test split.

    Parameters
    ----------
    X : ndarray of shape (n, d)
        Feature matrix (e.g., MDS coordinates).
    y : ndarray of shape (n,)
        Labels.
    classifier_name : str
        Classifier type ('knn', 'lda', 'svm', 'rf').
    test_size : float
        Fraction of data used for testing.
    random_state : int, optional
        Seed for the train/test split.
    **classifier_kwargs
        Passed to make_classifier.

    Returns
    -------
    error_rate : float
        Fraction of misclassified samples on the test set.
    """
    clf = make_classifier(classifier_name, **classifier_kwargs)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state,
        )
    except ValueError:
        # Stratification fails when a class has < 2 members
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
        )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return (preds != y_test).mean()


def single_trial(responses: np.ndarray, labels: np.ndarray,
                 query_indices: np.ndarray, n_components: Optional[int] = None,
                 classifier_name: str = "rf", seed: int = None,
                 **classifier_kwargs) -> float:
    """Run a single trial: select queries -> distance -> MDS -> classify -> error.

    Parameters
    ----------
    responses : ndarray of shape (n_models, M, p)
        Precomputed embedded responses.
    labels : ndarray of shape (n_models,)
        Model labels.
    query_indices : ndarray of shape (m,)
        Which queries to use.
    n_components : int or None
        MDS embedding dimension. If None, auto-selected via profile likelihood.
    classifier_name : str
        Classifier type.
    seed : int, optional
        Seed for train/test split.
    **classifier_kwargs
        Passed to make_classifier.

    Returns
    -------
    error_rate : float
        Test set error rate for this trial.
    """
    # Compute pairwise distances on selected queries
    D = pairwise_energy_distances_t0(responses, query_indices)

    # MDS embedding
    if n_components is not None:
        mds = ClassicalMDS(n_components=min(n_components, len(labels) - 1))
    else:
        mds = ClassicalMDS()
    X = mds.fit_transform(D)

    # Classify
    error = classify_and_evaluate(X, labels, classifier_name,
                                  random_state=seed, **classifier_kwargs)
    return error
