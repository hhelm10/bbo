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
        assert problem.alpha.shape == (50, 3)
        assert problem.directions.shape == (3, 20)  # r x p

    def test_directions_orthonormal(self):
        """Direction vectors should be orthonormal."""
        problem = make_problem(M=50, r=5, signal_prob=0.2, p=30)
        norms = np.linalg.norm(problem.directions, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)
        gram = problem.directions @ problem.directions.T
        np.testing.assert_allclose(gram, np.eye(5), atol=1e-10)

    def test_bernoulli_weight_field(self):
        """Alpha should be Bernoulli * Uniform: zero with prob (1-p), positive otherwise."""
        rng = np.random.default_rng(42)
        problem = make_problem(M=1000, r=5, signal_prob=0.3, rng=rng)
        alpha = problem.alpha
        # Fraction of zeros should be ~0.7
        frac_zero = (alpha == 0).mean()
        assert frac_zero == pytest.approx(0.7, abs=0.05)
        # Non-zero values should be in (0, 1)
        nonzero = alpha[alpha > 0]
        assert nonzero.min() > 0
        assert nonzero.max() < 1

    def test_sensitivity_matrix(self):
        """Sensitivity matrix should be binary version of alpha."""
        problem = make_problem(M=50, r=3, signal_prob=0.5)
        S = problem.sensitivity_matrix
        assert set(np.unique(S)).issubset({0.0, 1.0})
        np.testing.assert_array_equal(S, (problem.alpha > 0).astype(float))

    def test_rho_computation(self):
        """rho should be approximately 1 - signal_prob."""
        rng = np.random.default_rng(42)
        problem = make_problem(M=1000, r=5, signal_prob=0.2, rng=rng)
        assert problem.rho == pytest.approx(0.8, abs=0.05)

    def test_orthogonal_queries(self):
        """Orthogonal queries should have alpha[q,:] = 0."""
        rng = np.random.default_rng(42)
        problem = make_problem(M=100, r=3, signal_prob=0.2, rng=rng)
        orth = problem.orthogonal_queries
        for q in orth:
            assert problem.alpha[q].sum() == 0

    def test_signal_queries(self):
        """Signal queries should have alpha[q,:].sum() > 0."""
        rng = np.random.default_rng(42)
        problem = make_problem(M=100, r=3, signal_prob=0.2, rng=rng)
        sig = problem.signal_queries
        for q in sig:
            assert problem.alpha[q].sum() > 0

    def test_query_total_signal(self):
        """query_total_signal should sum alpha across dimensions."""
        problem = make_problem(M=50, r=3, signal_prob=0.5)
        np.testing.assert_allclose(
            problem.query_total_signal,
            problem.alpha.sum(axis=1),
        )

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

    def test_rank1_same_class_identical(self):
        """r=1: all models in same class have identical responses (no noise dims)."""
        problem = make_problem(M=50, r=1, signal_prob=0.5)
        models = problem.generate_models(10, rng=np.random.default_rng(0))
        responses = get_all_responses(models)
        labels = get_labels(models)

        # With r=1, theta has only dim 0 = class label, no random dims.
        # So all class-0 models have s = [+1] and all class-1 have s = [-1].
        class0 = responses[labels == 0]
        for i in range(1, class0.shape[0]):
            np.testing.assert_array_equal(class0[0], class0[i])

        class1 = responses[labels == 1]
        for i in range(1, class1.shape[0]):
            np.testing.assert_array_equal(class1[0], class1[i])

        # Classes should differ on signal queries
        sig = problem.signal_queries
        if len(sig) > 0:
            assert not np.allclose(class0[0, sig], class1[0, sig])

    def test_orthogonal_queries_zero(self):
        """On orthogonal queries (alpha=0 for all l), all models respond with zero."""
        problem = make_problem(M=100, r=3, signal_prob=0.2)
        models = problem.generate_models(20, rng=np.random.default_rng(0))
        responses = get_all_responses(models)

        orth = problem.orthogonal_queries
        if len(orth) > 0:
            orth_responses = responses[:, orth, :]
            # All should be zero since sqrt(alpha)=0 for all dimensions
            np.testing.assert_allclose(orth_responses, 0.0, atol=1e-14)

    def test_factorization_holds(self):
        """Squared distance should decompose as sum_l alpha_l(q) * phi_l(f,f')."""
        rng = np.random.default_rng(42)
        problem = make_problem(M=20, r=3, signal_prob=0.5, p=10, rng=rng)
        model_rng = np.random.default_rng(0)
        models = problem.generate_models(4, rng=model_rng)
        responses = get_all_responses(models)

        # Reconstruct theta for each model to compute phi
        # Re-generate with same rng to get the same theta values
        model_rng2 = np.random.default_rng(0)
        thetas = []
        for class_label in [0, 1]:
            for i in range(2):
                theta = model_rng2.integers(0, 2, size=problem.r)
                theta[0] = class_label
                thetas.append(theta)

        # Check factorization for every pair and every query
        for i in range(4):
            for j in range(i + 1, 4):
                for q in range(problem.M):
                    diff_sq = np.sum((responses[i, q] - responses[j, q])**2)
                    # phi_l(i,j) = (s_l(i) - s_l(j))^2 = 4 * 1[theta_l differs]
                    phi = 4.0 * (thetas[i] != thetas[j]).astype(float)
                    expected = np.sum(problem.alpha[q] * phi)
                    assert diff_sq == pytest.approx(expected, abs=1e-10)

    def test_rank1_perfect_classification(self):
        """r=1, high signal_prob -> perfect classification."""
        problem = make_problem(M=50, r=1, signal_prob=0.8)
        models = problem.generate_models(40, rng=np.random.default_rng(0))
        responses = get_all_responses(models)
        labels = get_labels(models)

        query_idx = np.arange(50)
        error = single_trial(responses, labels, query_idx, n_components=5)
        assert error == pytest.approx(0.0, abs=0.01)

    def test_within_class_variation(self):
        """r>1: models within same class differ due to random noise dims."""
        problem = make_problem(M=50, r=5, signal_prob=0.5)
        models = problem.generate_models(20, rng=np.random.default_rng(0))
        responses = get_all_responses(models)
        labels = get_labels(models)

        # Class 0 models should NOT all be identical (dims 2..r are random)
        class0 = responses[labels == 0]
        diffs = np.array([np.sum((class0[0] - class0[i])**2) for i in range(1, class0.shape[0])])
        # At least some pairs should differ
        assert diffs.max() > 0

    def test_r_exceeds_p_raises(self):
        """Should raise ValueError when r > p."""
        with pytest.raises(ValueError):
            make_problem(M=50, r=25, signal_prob=0.3, p=20)
