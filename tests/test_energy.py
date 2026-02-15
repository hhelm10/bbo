"""Tests for energy distance computation."""

import numpy as np
import pytest

from bbo.distances.energy import (
    pairwise_energy_distances_t0,
    pairwise_energy_distances_t0_loop,
    per_query_energy_tensor,
)


class TestEnergyDistances:
    def test_identical_models(self):
        """Identical models should have zero distance."""
        responses = np.random.default_rng(42).standard_normal((3, 10, 5))
        responses[1] = responses[0]  # Make model 1 identical to model 0

        D = pairwise_energy_distances_t0(responses)
        assert D[0, 1] == pytest.approx(0.0, abs=1e-10)
        assert D[1, 0] == pytest.approx(0.0, abs=1e-10)

    def test_symmetry(self):
        """Distance matrix should be symmetric."""
        responses = np.random.default_rng(42).standard_normal((5, 10, 3))
        D = pairwise_energy_distances_t0(responses)
        np.testing.assert_allclose(D, D.T, atol=1e-10)

    def test_zero_diagonal(self):
        """Diagonal should be zero."""
        responses = np.random.default_rng(42).standard_normal((5, 10, 3))
        D = pairwise_energy_distances_t0(responses)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-10)

    def test_non_negative(self):
        """All distances should be non-negative."""
        responses = np.random.default_rng(42).standard_normal((5, 10, 3))
        D = pairwise_energy_distances_t0(responses)
        assert (D >= -1e-10).all()

    def test_query_subset(self):
        """Using a subset of queries should give different (smaller) distances."""
        responses = np.random.default_rng(42).standard_normal((3, 20, 5))
        D_all = pairwise_energy_distances_t0(responses)
        D_sub = pairwise_energy_distances_t0(responses, query_indices=np.array([0, 1, 2]))
        # Different query sets give different distances
        assert D_all.shape == D_sub.shape

    def test_hand_computation(self):
        """Verify against hand computation for 2 models, 2 queries, 1D embedding."""
        # Model 0: q0 -> [1.0], q1 -> [2.0]
        # Model 1: q0 -> [3.0], q1 -> [2.0]
        responses = np.array([
            [[1.0], [2.0]],
            [[3.0], [2.0]],
        ])
        D = pairwise_energy_distances_t0(responses)

        # E^2(f0,f1) = 2 * (|1-3| + |2-2|) = 2 * (2 + 0) = 4
        # D = sqrt(4) = 2
        expected = 2.0
        assert D[0, 1] == pytest.approx(expected, abs=1e-10)

    def test_cumulative_grows_with_m(self):
        """Cumulative energy distance should grow with number of queries."""
        responses = np.random.default_rng(42).standard_normal((3, 20, 5))
        D_5 = pairwise_energy_distances_t0(responses, query_indices=np.arange(5))
        D_10 = pairwise_energy_distances_t0(responses, query_indices=np.arange(10))
        D_20 = pairwise_energy_distances_t0(responses, query_indices=np.arange(20))
        # More queries -> larger cumulative distance
        assert D_10[0, 1] > D_5[0, 1]
        assert D_20[0, 1] > D_10[0, 1]

    def test_loop_matches_vectorized(self):
        """Loop version should give same result as vectorized."""
        responses = np.random.default_rng(42).standard_normal((8, 15, 4))
        D_vec = pairwise_energy_distances_t0(responses)
        D_loop = pairwise_energy_distances_t0_loop(responses)
        np.testing.assert_allclose(D_vec, D_loop, atol=1e-10)


class TestPerQueryTensor:
    def test_shape(self):
        """Tensor should have shape (M, n_pairs)."""
        responses = np.random.default_rng(42).standard_normal((5, 10, 3))
        T, pairs = per_query_energy_tensor(responses)
        n_pairs = 5 * 4 // 2  # 10
        assert T.shape == (10, n_pairs)
        assert pairs.shape == (n_pairs, 2)

    def test_consistency_with_distance(self):
        """Summing tensor values should relate to pairwise distances."""
        responses = np.random.default_rng(42).standard_normal((3, 5, 2))
        T, pairs = per_query_energy_tensor(responses)
        D = pairwise_energy_distances_t0(responses)

        # For pair (0,1): D^2[0,1] = 2 * sum_q T[q, pair_idx]
        pair_01_idx = None
        for k, (i, j) in enumerate(pairs):
            if i == 0 and j == 1:
                pair_01_idx = k
                break

        E_sq = 2.0 * T[:, pair_01_idx].sum()
        assert np.sqrt(E_sq) == pytest.approx(D[0, 1], abs=1e-10)
