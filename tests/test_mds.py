"""Tests for Classical MDS wrapper around graspologic."""

import numpy as np
import pytest

from bbo.embedding.mds import ClassicalMDS, select_dimension


def _distance_matrix(X):
    """Compute pairwise Euclidean distance matrix from coordinates."""
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


class TestClassicalMDS:
    def test_square_corners(self):
        """MDS should recover 4 corners of a unit square (up to rotation/reflection)."""
        X_true = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        D = _distance_matrix(X_true)

        mds = ClassicalMDS(n_components=2)
        X = mds.fit_transform(D)

        D_recovered = _distance_matrix(X)
        np.testing.assert_allclose(D_recovered, D, atol=1e-10)

    def test_equilateral_triangle(self):
        """MDS on an equilateral triangle should embed in 2D with equal distances."""
        D = np.array([[0, 1, 1],
                       [1, 0, 1],
                       [1, 1, 0]], dtype=float)

        mds = ClassicalMDS(n_components=2)
        X = mds.fit_transform(D)

        D_recovered = _distance_matrix(X)
        np.testing.assert_allclose(D_recovered, D, atol=1e-10)

    def test_collinear_points(self):
        """Collinear points should embed in 1D."""
        X_true = np.array([[0], [1], [3], [6]], dtype=float)
        D = _distance_matrix(X_true)

        mds = ClassicalMDS(n_components=1)
        X = mds.fit_transform(D)

        D_recovered = _distance_matrix(X)
        np.testing.assert_allclose(D_recovered, D, atol=1e-10)

    def test_zero_distance_matrix(self):
        """Zero distance matrix should give zero embedding."""
        D = np.zeros((5, 5))
        mds = ClassicalMDS(n_components=2)
        X = mds.fit_transform(D)
        np.testing.assert_allclose(X, 0, atol=1e-10)

    def test_auto_dimension_selection(self):
        """n_components=None should auto-select and store n_components_."""
        rng = np.random.default_rng(42)
        X_true = rng.standard_normal((20, 3))
        D = _distance_matrix(X_true)

        mds = ClassicalMDS()  # n_components=None by default
        X = mds.fit_transform(D)

        assert mds.n_components_ is not None
        assert mds.n_components_ >= 1
        assert X.shape == (20, mds.n_components_)

    def test_explicit_n_components(self):
        """Explicit n_components should be respected."""
        D = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
        mds = ClassicalMDS(n_components=2)
        X = mds.fit_transform(D)
        assert mds.n_components_ == 2
        assert X.shape == (3, 2)

    def test_singular_values_available(self):
        """singular_values_ should be available after fit."""
        rng = np.random.default_rng(42)
        X_true = rng.standard_normal((10, 3))
        D = _distance_matrix(X_true)
        mds = ClassicalMDS(n_components=3)
        mds.fit_transform(D)
        assert len(mds.singular_values_) == 3


class TestSelectDimension:
    def test_rank1_data(self):
        """Rank-1 eigenvalue spectrum should select d=1 or small d."""
        eigenvalues = np.array([100.0, 0.1, 0.05, 0.02, 0.01, 0.005])
        d = select_dimension(eigenvalues, n_elbows=1)
        assert d == 1

    def test_rank5_data(self):
        """Rank-5 eigenvalue spectrum should select d=5."""
        eigenvalues = np.array([50.0, 48.0, 45.0, 42.0, 40.0,
                                0.1, 0.05, 0.02, 0.01, 0.005])
        d = select_dimension(eigenvalues, n_elbows=1)
        assert d == 5

    def test_all_equal_eigenvalues(self):
        """All-equal eigenvalues should return some valid dimension."""
        eigenvalues = np.ones(10) * 5.0
        d = select_dimension(eigenvalues, n_elbows=1)
        assert 1 <= d <= 10

    def test_single_eigenvalue(self):
        """Single positive eigenvalue should return d=1."""
        eigenvalues = np.array([5.0])
        d = select_dimension(eigenvalues, n_elbows=1)
        assert d == 1

    def test_two_elbows_expands(self):
        """Two-elbow selection should return d >= first elbow."""
        eigenvalues = np.array([100.0, 50.0, 10.0, 1.0, 0.1, 0.01])
        d1 = select_dimension(eigenvalues, n_elbows=1)
        d2 = select_dimension(eigenvalues, n_elbows=2)
        assert d2 >= d1

    def test_negative_eigenvalues_ignored(self):
        """Negative eigenvalues should be filtered out."""
        eigenvalues = np.array([100.0, 10.0, 1.0, -0.5, -1.0])
        d = select_dimension(eigenvalues, n_elbows=1)
        assert 1 <= d <= 3

    def test_returns_at_least_1(self):
        """Should always return at least 1."""
        eigenvalues = np.array([0.0, -1.0, -2.0])
        d = select_dimension(eigenvalues, n_elbows=1)
        assert d >= 1
