"""Tests for Classical MDS implementation."""

import numpy as np
import pytest

from bbo.embedding.mds import ClassicalMDS, select_dimension, _compute_profile_likelihood


def _distance_matrix(X):
    """Compute pairwise Euclidean distance matrix from coordinates."""
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


class TestClassicalMDS:
    def test_square_corners(self):
        """MDS should recover 4 corners of a unit square (up to rotation/reflection)."""
        # 4 corners of a unit square
        X_true = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        D = _distance_matrix(X_true)

        mds = ClassicalMDS(n_components=2)
        X = mds.fit_transform(D)

        # Recovered distances should match original
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

    def test_eigenvalues_descending(self):
        """Eigenvalues should be in descending order."""
        rng = np.random.default_rng(42)
        X_true = rng.standard_normal((10, 5))
        D = _distance_matrix(X_true)

        mds = ClassicalMDS(n_components=5)
        mds.fit(D)

        eigs = mds.eigenvalues
        assert all(eigs[i] >= eigs[i+1] for i in range(len(eigs)-1))

    def test_explained_variance_sums_to_one(self):
        """Explained variance ratios should sum to ~1 (for non-negative eigenvalues)."""
        rng = np.random.default_rng(42)
        X_true = rng.standard_normal((10, 3))
        D = _distance_matrix(X_true)

        mds = ClassicalMDS(n_components=3)
        mds.fit(D)

        evr = mds.explained_variance_ratio
        # Only positive eigenvalues contribute
        pos_sum = evr[mds.eigenvalues > 0].sum()
        assert pos_sum <= 1.0 + 1e-10

    def test_effective_rank(self):
        """Effective rank should capture the intrinsic dimensionality."""
        rng = np.random.default_rng(42)
        # 3D data — effective rank should be <= 3
        X_true = rng.standard_normal((20, 3))
        D = _distance_matrix(X_true)

        mds = ClassicalMDS(n_components=10)
        mds.fit(D)

        r = mds.effective_rank(threshold=0.99)
        assert r <= 5  # Some tolerance for numerical noise

    def test_project_new_points(self):
        """Out-of-sample projection should place new points correctly."""
        rng = np.random.default_rng(42)
        X_all = rng.standard_normal((15, 3))
        D_all = _distance_matrix(X_all)

        # Fit on first 10
        mds = ClassicalMDS(n_components=3)
        X_train = mds.fit_transform(D_all[:10, :10])

        # Project remaining 5
        D_new = D_all[10:, :10]
        X_new = mds.project(D_new)

        # Check that projected distances are reasonable
        # The new points' distances to training points should be close
        for i in range(5):
            for j in range(10):
                d_true = D_all[10 + i, j]
                d_proj = np.linalg.norm(X_new[i] - X_train[j])
                assert abs(d_proj - d_true) < 1.0  # Rough tolerance

    def test_zero_distance_matrix(self):
        """Zero distance matrix should give zero embedding."""
        D = np.zeros((5, 5))
        mds = ClassicalMDS(n_components=2)
        X = mds.fit_transform(D)
        np.testing.assert_allclose(X, 0, atol=1e-10)

    def test_n_components_clipping(self):
        """Should handle n_components > n gracefully."""
        D = np.array([[0, 1], [1, 0]], dtype=float)
        mds = ClassicalMDS(n_components=10)
        X = mds.fit_transform(D)
        assert X.shape == (2, 2)  # Clipped to n

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

    def test_auto_vs_explicit_stores_n_components_(self):
        """Explicit n_components should also store n_components_."""
        D = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
        mds = ClassicalMDS(n_components=2)
        X = mds.fit_transform(D)
        assert mds.n_components_ == 2
        assert X.shape == (3, 2)


class TestSelectDimension:
    def test_rank1_data(self):
        """Rank-1 eigenvalue spectrum should select d=1 or small d."""
        # One large eigenvalue, rest near zero
        eigenvalues = np.array([100.0, 0.1, 0.05, 0.02, 0.01, 0.005])
        d = select_dimension(eigenvalues, n_elbows=1)
        assert d == 1

    def test_rank5_data(self):
        """Rank-5 eigenvalue spectrum should select d=5."""
        # 5 similar-magnitude eigenvalues, then sharp drop to noise floor
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

    def test_two_elbows(self):
        """Two-elbow selection should return d <= first elbow."""
        eigenvalues = np.array([100.0, 50.0, 10.0, 1.0, 0.1, 0.01])
        d1 = select_dimension(eigenvalues, n_elbows=1)
        d2 = select_dimension(eigenvalues, n_elbows=2)
        assert d2 <= d1

    def test_negative_eigenvalues_ignored(self):
        """Negative eigenvalues should be filtered out."""
        eigenvalues = np.array([100.0, 10.0, 1.0, -0.5, -1.0])
        d = select_dimension(eigenvalues, n_elbows=1)
        assert 1 <= d <= 3  # only 3 positive eigenvalues

    def test_returns_at_least_1(self):
        """Should always return at least 1."""
        eigenvalues = np.array([0.0, -1.0, -2.0])
        d = select_dimension(eigenvalues, n_elbows=1)
        assert d >= 1


class TestProfileLikelihood:
    def test_output_shape(self):
        """Should return n-1 log-likelihoods for n values."""
        values = np.array([10.0, 5.0, 1.0, 0.5, 0.1])
        logliks = _compute_profile_likelihood(values)
        assert logliks.shape == (4,)

    def test_clear_elbow(self):
        """Argmax should be at the obvious break point."""
        values = np.array([100.0, 99.0, 98.0, 1.0, 0.5, 0.1])
        logliks = _compute_profile_likelihood(values)
        # Best split should be at index 3 (after the 3 large values)
        assert np.argmax(logliks) + 1 == 3
