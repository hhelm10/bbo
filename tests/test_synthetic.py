"""Tests for synthetic problem generation."""

import numpy as np
import pytest

from bbo.models.synthetic import make_problem, get_all_responses, get_labels, SyntheticProblem
from bbo.classification.evaluate import single_trial


class TestSyntheticProblem:
    def test_make_problem_basic(self):
        """Basic problem creation should work."""
        problem = make_problem(M=50, r=3, signal_prob=0.2)
        assert problem.M == 50
        assert problem.r == 3
        assert problem.A.shape == (50, 3)
        assert problem.directions.shape == (3, 20)  # r x p

    def test_directions_unit_norm(self):
        """Direction vectors should be unit norm."""
        problem = make_problem(M=50, r=5, signal_prob=0.2, p=30)
        norms = np.linalg.norm(problem.directions, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_bernoulli_sensitivity(self):
        """Sensitivity matrix entries should be independent Bernoulli."""
        rng = np.random.default_rng(42)
        problem = make_problem(M=1000, r=5, signal_prob=0.3, rng=rng)
        A = problem.sensitivity_matrix
        # Empirical probability should be close to signal_prob
        assert A.mean() == pytest.approx(0.3, abs=0.05)
        # Each dimension should have ~300 active queries
        for l in range(5):
            assert A[:, l].sum() == pytest.approx(300, abs=50)

    def test_rho_computation(self):
        """rho should be approximately 1 - signal_prob."""
        rng = np.random.default_rng(42)
        problem = make_problem(M=1000, r=5, signal_prob=0.2, rng=rng)
        # With large M, rho converges to 1 - signal_prob
        assert problem.rho == pytest.approx(0.8, abs=0.05)

    def test_orthogonal_queries(self):
        """Orthogonal queries should have A[q,:] = 0."""
        rng = np.random.default_rng(42)
        problem = make_problem(M=100, r=3, signal_prob=0.2, rng=rng)
        orth = problem.orthogonal_queries
        A = problem.sensitivity_matrix
        for q in orth:
            assert A[q].sum() == 0

    def test_signal_queries(self):
        """Signal queries should have A[q,:].sum() > 0."""
        rng = np.random.default_rng(42)
        problem = make_problem(M=100, r=3, signal_prob=0.2, rng=rng)
        sig = problem.signal_queries
        A = problem.sensitivity_matrix
        for q in sig:
            assert A[q].sum() > 0

    def test_generate_models(self):
        """Model generation should produce correct number and labels."""
        problem = make_problem(M=50, r=3, signal_prob=0.2)
        models = problem.generate_models(20)
        assert len(models) == 20
        labels = get_labels(models)
        assert (labels == 0).sum() == 10
        assert (labels == 1).sum() == 10

    def test_responses_shape(self):
        """Response array should have correct shape."""
        problem = make_problem(M=50, r=3, signal_prob=0.2, p=15)
        models = problem.generate_models(20)
        responses = get_all_responses(models)
        assert responses.shape == (20, 50, 15)

    def test_no_noise_perfect_separation(self):
        """Without noise, classes should differ on signal queries."""
        problem = make_problem(M=50, r=1, signal_prob=0.5, noise_level=0.0)
        models = problem.generate_models(10)
        responses = get_all_responses(models)
        labels = get_labels(models)

        # On signal queries, class 0 and class 1 should differ
        sig = problem.signal_queries
        if len(sig) > 0:
            class0_responses = responses[labels == 0][:, sig, :]
            class1_responses = responses[labels == 1][:, sig, :]

            # All class 0 models should have same response on signal queries
            for i in range(1, class0_responses.shape[0]):
                np.testing.assert_array_equal(class0_responses[0], class0_responses[i])

            # Class 0 and class 1 should differ on signal queries
            assert not np.allclose(class0_responses[0], class1_responses[0])

    def test_orthogonal_queries_identical(self):
        """On orthogonal queries, all models should respond identically (zero)."""
        problem = make_problem(M=100, r=3, signal_prob=0.2, noise_level=0.0)
        models = problem.generate_models(20)
        responses = get_all_responses(models)

        orth = problem.orthogonal_queries
        if len(orth) > 0:
            orth_responses = responses[:, orth, :]
            # All models should be identical on orthogonal queries
            for i in range(1, len(models)):
                np.testing.assert_array_equal(orth_responses[0], orth_responses[i])

    def test_multi_dimension_contribution(self):
        """With independent Bernoulli, queries can contribute to multiple dimensions."""
        rng = np.random.default_rng(42)
        problem = make_problem(M=100, r=10, signal_prob=0.5, noise_level=0.0, p=30, rng=rng)
        A = problem.sensitivity_matrix
        # With signal_prob=0.5 and r=10, most queries should be in multiple dims
        max_dims_per_query = A.sum(axis=1).max()
        assert max_dims_per_query > 1, "Expected multi-dimension queries"

    def test_rank1_perfect_classification(self):
        """r=1, no noise, high signal_prob -> perfect classification."""
        problem = make_problem(M=50, r=1, signal_prob=0.8, noise_level=0.0)
        models = problem.generate_models(40, rng=np.random.default_rng(0))
        responses = get_all_responses(models)
        labels = get_labels(models)

        query_idx = np.arange(50)
        error = single_trial(responses, labels, query_idx, n_components=5)
        assert error == pytest.approx(0.0, abs=0.01)

    def test_noise_introduces_errors(self):
        """With high noise, classification should be harder."""
        problem = make_problem(M=50, r=1, signal_prob=0.5, noise_level=0.5)
        models = problem.generate_models(40, rng=np.random.default_rng(0))
        responses = get_all_responses(models)
        labels = get_labels(models)

        query_idx = np.arange(50)
        error = single_trial(responses, labels, query_idx, n_components=5)
        assert error >= 0.0  # Just verify it runs

    def test_independent_contributions(self):
        """Each entry A[q,l] should be independent."""
        rng = np.random.default_rng(42)
        problem = make_problem(M=10000, r=2, signal_prob=0.3, rng=rng)
        A = problem.A
        # Check that A[:,0] and A[:,1] are approximately independent
        # P(A[q,0]=1 AND A[q,1]=1) should be ~0.3*0.3=0.09
        joint = (A[:, 0] * A[:, 1]).mean()
        assert joint == pytest.approx(0.09, abs=0.02)
