"""Tests for classification evaluation pipeline."""

import numpy as np
import pytest

from bbo.classification.evaluate import (
    make_classifier, classify_and_evaluate, single_trial,
)


class TestMakeClassifier:
    def test_knn(self):
        clf = make_classifier("knn", n_neighbors=3)
        assert hasattr(clf, "fit")

    def test_lda(self):
        clf = make_classifier("lda")
        assert hasattr(clf, "fit")

    def test_svm(self):
        clf = make_classifier("svm")
        assert hasattr(clf, "fit")

    def test_unknown(self):
        with pytest.raises(ValueError):
            make_classifier("unknown")


class TestClassifyAndEvaluate:
    def test_perfectly_separable(self):
        """Linearly separable data should yield zero error."""
        rng = np.random.default_rng(42)
        n = 40
        X = np.zeros((n, 2))
        X[:n//2, 0] = rng.normal(0, 0.1, n//2)
        X[n//2:, 0] = rng.normal(5, 0.1, n//2)
        y = np.array([0] * (n//2) + [1] * (n//2))

        error = classify_and_evaluate(X, y, "knn", n_neighbors=3)
        assert error == pytest.approx(0.0, abs=0.05)

    def test_random_labels(self):
        """Random labels should yield ~50% error."""
        rng = np.random.default_rng(42)
        n = 100
        X = rng.standard_normal((n, 5))
        y = rng.integers(0, 2, n)

        error = classify_and_evaluate(X, y, "knn", n_neighbors=5)
        assert 0.2 < error < 0.8  # Roughly around chance


class TestSingleTrial:
    def test_runs_end_to_end(self):
        """single_trial should complete without error."""
        rng = np.random.default_rng(42)
        n_models = 20
        M = 30
        p = 5

        # Create responses where first half of models respond differently than second half
        responses = rng.standard_normal((n_models, M, p))
        responses[:n_models//2, :, 0] += 5  # Shift class 0 in first dimension
        labels = np.array([0] * (n_models//2) + [1] * (n_models//2))

        query_idx = np.arange(10)
        error = single_trial(responses, labels, query_idx, n_components=5)
        assert 0 <= error <= 1
