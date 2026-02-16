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
        assert problem.directions.shape == (3, 20)  # r x p_embed

    def test_directions_orthonormal(self):
        """Direction vectors should be orthonormal."""
        problem = make_problem(M=50, r=5, signal_prob=0.2, p_embed=30)
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
        """Model generation should produce correct number with balanced parity labels."""
        problem = make_problem(M=50, r=3, signal_prob=0.2)
        models = problem.generate_models(1000, rng=np.random.default_rng(0))
        assert len(models) == 1000
        labels = get_labels(models)
        # Parity of uniform {0,1}^r should be ~50/50
        assert (labels == 0).sum() == pytest.approx(500, abs=50)
        assert (labels == 1).sum() == pytest.approx(500, abs=50)

    def test_responses_shape(self):
        """Response array should have correct shape."""
        problem = make_problem(M=50, r=3, signal_prob=0.2, p_embed=15)
        models = problem.generate_models(20)
        responses = get_all_responses(models)
        assert responses.shape == (20, 50, 15)

    def test_parity_label(self):
        """Labels should be parity of theta."""
        problem = make_problem(M=50, r=3, signal_prob=0.5)
        rng = np.random.default_rng(42)
        models = problem.generate_models(10, rng=rng)

        # Reconstruct theta values
        rng2 = np.random.default_rng(42)
        for model in models:
            theta = rng2.integers(0, 2, size=problem.r)
            expected_label = int(theta.sum() % 2)
            assert model.label == expected_label

    def test_label_noise(self):
        """With eta>0, some labels should be flipped."""
        problem = make_problem(M=50, r=3, signal_prob=0.5)
        eta = 0.2
        models_noisy = problem.generate_models(1000, eta=eta,
                                                rng=np.random.default_rng(0))
        labels_noisy = get_labels(models_noisy)

        # Replay same RNG to reconstruct true parity labels
        rng2 = np.random.default_rng(0)
        n_flipped = 0
        for i in range(1000):
            theta = rng2.integers(0, 2, size=problem.r)
            true_label = int(theta.sum() % 2)
            noise_draw = rng2.random()  # consumed when eta > 0
            if labels_noisy[i] != true_label:
                n_flipped += 1

        flip_rate = n_flipped / 1000
        assert flip_rate == pytest.approx(0.2, abs=0.05)

    def test_rank1_same_theta_identical(self):
        """r=1: models with same theta have identical responses."""
        problem = make_problem(M=50, r=1, signal_prob=0.5)
        models = problem.generate_models(10, rng=np.random.default_rng(0))
        responses = get_all_responses(models)

        # Reconstruct thetas
        rng2 = np.random.default_rng(0)
        thetas = [rng2.integers(0, 2, size=1) for _ in range(10)]

        # Models with same theta[0] should have identical responses
        for i in range(10):
            for j in range(i + 1, 10):
                if thetas[i][0] == thetas[j][0]:
                    np.testing.assert_array_equal(responses[i], responses[j])

    def test_orthogonal_queries_zero(self):
        """On orthogonal queries (alpha=0 for all l), all models respond with zero."""
        problem = make_problem(M=100, r=3, signal_prob=0.2)
        models = problem.generate_models(20, rng=np.random.default_rng(0))
        responses = get_all_responses(models)

        orth = problem.orthogonal_queries
        if len(orth) > 0:
            orth_responses = responses[:, orth, :]
            np.testing.assert_allclose(orth_responses, 0.0, atol=1e-14)

    def test_factorization_holds(self):
        """Squared distance should decompose as sum_l alpha_l(q) * 1[theta differs]."""
        rng = np.random.default_rng(42)
        problem = make_problem(M=20, r=3, signal_prob=0.5, p_embed=10, rng=rng)
        model_rng = np.random.default_rng(0)
        models = problem.generate_models(4, rng=model_rng)
        responses = get_all_responses(models)

        # Reconstruct theta for each model
        model_rng2 = np.random.default_rng(0)
        thetas = []
        for i in range(4):
            theta = model_rng2.integers(0, 2, size=problem.r)
            # consume the label noise rng call (eta=0 so no flip, but no rng call either)
            thetas.append(theta)

        # Check factorization for every pair and every query
        for i in range(4):
            for j in range(i + 1, 4):
                for q in range(problem.M):
                    diff_sq = np.sum((responses[i, q] - responses[j, q])**2)
                    # phi_l(i,j) = 1[theta_l differs]
                    phi = (thetas[i] != thetas[j]).astype(float)
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
        assert error == pytest.approx(0.0, abs=0.05)

    def test_within_class_variation(self):
        """r>1: models with same label can differ (different theta vectors, same parity)."""
        problem = make_problem(M=50, r=5, signal_prob=0.5)
        models = problem.generate_models(100, rng=np.random.default_rng(0))
        responses = get_all_responses(models)
        labels = get_labels(models)

        # Class 0 models should NOT all be identical (many theta vectors have parity 0)
        class0 = responses[labels == 0]
        if class0.shape[0] > 1:
            diffs = np.array([np.sum((class0[0] - class0[i])**2)
                              for i in range(1, class0.shape[0])])
            assert diffs.max() > 0

    def test_r_exceeds_p_raises(self):
        """Should raise ValueError when r > p_embed."""
        with pytest.raises(ValueError):
            make_problem(M=50, r=25, signal_prob=0.3, p_embed=20)
