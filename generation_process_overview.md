# Synthetic Data Generation Process

## Overview

The synthetic data generation implements the **Bernoulli-Weight Model** from `computational_model.md`. It produces a controlled classification problem where the difficulty is parameterized by the discriminative rank $r$, the activation probability $p$, and the noise dimension scale $\sigma$.

## Parameters

| Parameter | Symbol | Description | Default |
|-----------|--------|-------------|---------|
| `M` | $M$ | Number of queries | 100 |
| `r` | $r$ | Discriminative rank (latent dimensions) | 5 |
| `signal_prob` | $p$ | Per-dimension activation probability | 0.3 |
| `sigma` | $\sigma$ | Noise dimension scale ($c_\ell$ for $\ell \geq 2$) | 1.0 |
| `p` | (embedding dim) | Embedding dimension ($\geq r$) | 20 |
| `n_models` | $n$ | Number of models to generate | 200 |

## Step 1: Construct the Discriminative Field $\alpha$

**Code:** `make_problem()` in `bbo/models/synthetic.py`

For each query $q \in \{1, \ldots, M\}$ and dimension $\ell \in \{1, \ldots, r\}$:

$$\alpha_\ell(q) = \xi_{q\ell} \cdot w_{q\ell}$$

where:
- $\xi_{q\ell} \sim \text{Bernoulli}(p)$ — does query $q$ activate dimension $\ell$?
- $w_{q\ell} \sim \text{Uniform}(0, 1)$ — signal intensity

This produces an $M \times r$ matrix `alpha` with many zeros (when $\xi = 0$).

**Key property:** The zero set of dimension $\ell$ is $\mathcal{Z}_\ell = \{q : \alpha_\ell(q) = 0\}$, and under uniform $\Pi_Q$:
$$\rho_\ell = \Pi_Q(\mathcal{Z}_\ell) = 1 - p$$

## Step 2: Construct Per-Dimension Scales $c$

**Code:** `make_problem()` in `bbo/models/synthetic.py`

$$c_1 = 1 \quad \text{(signal dimension, fixed)}, \qquad c_\ell = \sigma \quad \text{for } \ell \geq 2 \text{ (noise dimensions)}$$

This controls the **signal-to-noise ratio**. The model-pair kernel from the spec is:

$$\phi_\ell(f, f') = c_\ell \cdot \mathbf{1}[\theta_{f,\ell} \neq \theta_{f',\ell}]$$

With $\sigma = 1$, all dimensions have equal scale (no SNR advantage for signal). With $\sigma > 1$, noise dimensions are amplified, making classification harder. With $\sigma < 1$, signal is relatively stronger, making classification easier.

**Population-level SNR** (for within- vs cross-class expected distance):
$$\text{SNR} \approx \frac{c_1}{(r-1) \cdot c_{\text{noise}} \cdot 0.5} = \frac{2}{(r-1)\sigma}$$

## Step 3: Construct Orthonormal Directions

**Code:** `make_problem()` in `bbo/models/synthetic.py`

Generate $r$ orthonormal direction vectors in $\mathbb{R}^p$ via QR decomposition of a random $p \times r$ Gaussian matrix. These are stored as rows of a $(r, p)$ matrix `directions`.

## Step 4: Generate Models

**Code:** `SyntheticProblem.generate_models()` in `bbo/models/synthetic.py`

For each model $f$ (with $n/2$ per class):

1. **Latent type:** $\theta_f \in \{0, 1\}^r$
   - $\theta_{f,1} = y$ (class label: 0 or 1)
   - $\theta_{f,\ell} \sim \text{Bernoulli}(0.5)$ for $\ell \geq 2$ (random noise dimensions)

2. **Signs:** $s_\ell(f) = 1 - 2\theta_{f,\ell} \in \{+1, -1\}$

3. **Embedded response** for all $M$ queries:
$$g(f(q)) = \sum_{\ell=1}^r \sqrt{\alpha_\ell(q)} \cdot \frac{\sqrt{c_\ell}}{2} \cdot s_\ell(f) \cdot \text{direction}[\ell]$$

The $\sqrt{c_\ell}/2$ scaling ensures the factorization:

$$\|g(f_i(q)) - g(f_j(q))\|^2 = \sum_{\ell=1}^r \alpha_\ell(q) \cdot c_\ell \cdot \mathbf{1}[\theta_{f_i,\ell} \neq \theta_{f_j,\ell}]$$

**Proof:** Since directions are orthonormal, cross-terms vanish. For dimension $\ell$:
- If $\theta_{i,\ell} = \theta_{j,\ell}$: $s_\ell(i) = s_\ell(j)$, contribution = 0
- If $\theta_{i,\ell} \neq \theta_{j,\ell}$: $(s_\ell(i) - s_\ell(j))^2 = (1 - (-1))^2 = 4$, and $(\sqrt{c_\ell}/2)^2 \cdot 4 = c_\ell$

## Step 5: Compute Responses and Labels

**Code:** `get_all_responses()`, `get_labels()` in `bbo/models/synthetic.py`

- `responses`: shape `(n_models, M, p)` — precomputed $g(f(q))$ for all models and queries
- `labels`: shape `(n_models,)` — class labels $y = \theta_{f,1}$

## Classification Pipeline

**Code:** `bbo/classification/evaluate.py`

Given a set of $m$ query indices:
1. **Energy distance:** Compute pairwise $\hat{\mathcal{E}}^2(f_i, f_j) = \sum_{q \in S} \|g(f_i(q)) - g(f_j(q))\|^2$ over the selected queries
2. **MDS embedding:** Classical MDS with `n_components = r` (matching the true discriminative rank)
3. **Classification:** $k$-NN with $k = \lfloor \log n \rfloor$ (rounded to nearest odd), evaluated via leave-one-out cross-validation
4. **Error:** Fraction of misclassified models

## What Makes Classification Hard

The squared energy distance between models $i, j$ using queries $S$ is:

$$\hat{\mathcal{E}}^2(f_i, f_j) = \sum_{q \in S} \sum_{\ell=1}^r \alpha_\ell(q) \cdot c_\ell \cdot \mathbf{1}[\theta_{i,\ell} \neq \theta_{j,\ell}]$$

- **Cross-class pairs** ($y_i \neq y_j$): dimension 1 always contributes $c_1 \sum_q \alpha_1(q)$ (signal), plus noise from dims 2,...,$r$
- **Within-class pairs** ($y_i = y_j$): dimension 1 contributes 0, only noise from dims 2,...,$r$

Classification succeeds when the signal term is large relative to noise, which requires:

1. **Dimension 1 is activated** by at least one query in $S$ (coupon-collector: fails with prob $\rho^m$)
2. **SNR is sufficient** for the classifier to separate classes given the noise from $r-1$ dimensions at scale $\sigma$
3. **Enough models $n$** for the classifier to estimate the decision boundary

The theoretical bound $P[\text{error} \geq 0.5] \leq r \rho^m$ captures requirement (1) via a union bound. Requirements (2) and (3) are finite-sample effects not captured by the bound.
