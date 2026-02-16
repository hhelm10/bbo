# Synthetic Data Generation Process

## Overview

The synthetic data generation implements the **Bernoulli-Weight Model** from `computational_model.md`. It produces a controlled classification problem where the difficulty is parameterized by the discriminative rank $r$, the activation probability $p$, and the label noise $\eta$.

The key design choice is the **parity label**: $y = \theta_{f,1} \oplus \theta_{f,2} \oplus \cdots \oplus \theta_{f,r} \oplus \text{Bernoulli}(\eta)$. This ensures ALL $r$ dimensions must be activated to beat chance, making the theoretical bound $P[\text{error} \geq 0.5] \leq r \rho^m$ tight.

## Parameters

| Parameter | Symbol | Description | Default |
|-----------|--------|-------------|---------|
| `M` | $M$ | Number of queries | 100 |
| `r` | $r$ | Discriminative rank (latent dimensions) | 5 |
| `signal_prob` | $p$ | Per-dimension activation probability | 0.3 |
| `p_embed` | $p_{\text{embed}}$ | Embedding dimension ($\geq r$) | 20 |
| `n_models` | $n$ | Number of models to generate | 200 |
| `eta` | $\eta$ | Label noise probability | 0.0 |

## Step 1: Construct the Discriminative Field $\alpha$

**Code:** `make_problem()` in `bbo/models/synthetic.py`

For each query $q \in \{1, \ldots, M\}$ and dimension $\ell \in \{1, \ldots, r\}$:

$$\alpha_\ell(q) = \xi_{q\ell} \cdot w_{q\ell}$$

where:
- $\xi_{q\ell} \sim \text{Bernoulli}(p)$ -- does query $q$ activate dimension $\ell$?
- $w_{q\ell} \sim \text{Uniform}(0, 1)$ -- signal intensity

This produces an $M \times r$ matrix `alpha` with many zeros (when $\xi = 0$).

**Key property:** The zero set of dimension $\ell$ is $\mathcal{Z}_\ell = \{q : \alpha_\ell(q) = 0\}$, and under uniform $\Pi_Q$:
$$\rho_\ell = \Pi_Q(\mathcal{Z}_\ell) = 1 - p$$

## Step 2: Construct Orthonormal Directions

**Code:** `make_problem()` in `bbo/models/synthetic.py`

Generate $r$ orthonormal direction vectors in $\mathbb{R}^{p_{\text{embed}}}$ via QR decomposition of a random $p_{\text{embed}} \times r$ Gaussian matrix. These are stored as rows of an $(r, p_{\text{embed}})$ matrix `directions`.

## Step 3: Generate Models

**Code:** `SyntheticProblem.generate_models()` in `bbo/models/synthetic.py`

For each model $f$:

1. **Latent type:** $\theta_f \in \{0, 1\}^r$ drawn uniformly (all $2^r$ configurations equally likely)

2. **Parity label:** $y = \theta_{f,1} \oplus \theta_{f,2} \oplus \cdots \oplus \theta_{f,r} \oplus \text{Bernoulli}(\eta)$
   - When $\eta = 0$: deterministic parity, $L^* = 0$
   - When $\eta > 0$: label noise, $L^* = \eta$

3. **Signs:** $s_\ell(f) = 1 - 2\theta_{f,\ell} \in \{+1, -1\}$

4. **Embedded response** for all $M$ queries:
$$g(f(q)) = \sum_{\ell=1}^r \sqrt{\alpha_\ell(q)} \cdot \frac{1}{2} \cdot s_\ell(f) \cdot \text{direction}[\ell]$$

This ensures the factorization:

$$\|g(f_i(q)) - g(f_j(q))\|^2 = \sum_{\ell=1}^r \alpha_\ell(q) \cdot \mathbf{1}[\theta_{f_i,\ell} \neq \theta_{f_j,\ell}]$$

**Proof:** Since directions are orthonormal, cross-terms vanish. For dimension $\ell$:
- If $\theta_{i,\ell} = \theta_{j,\ell}$: $s_\ell(i) = s_\ell(j)$, contribution = 0
- If $\theta_{i,\ell} \neq \theta_{j,\ell}$: $(s_\ell(i) - s_\ell(j))^2 = (1 - (-1))^2 = 4$, and $(1/2)^2 \cdot 4 = 1$

## Step 4: Compute Responses and Labels

**Code:** `get_all_responses()`, `get_labels()` in `bbo/models/synthetic.py`

- `responses`: shape `(n_models, M, p_embed)` -- precomputed $g(f(q))$ for all models and queries
- `labels`: shape `(n_models,)` -- parity class labels

## Classification Pipeline

**Code:** `bbo/classification/evaluate.py`

Given a set of $m$ query indices:
1. **Energy distance:** Compute pairwise $\hat{\mathcal{E}}^2(f_i, f_j) = \sum_{q \in S} \|g(f_i(q)) - g(f_j(q))\|^2$ over the selected queries
2. **MDS embedding:** Classical MDS with `n_components = r` (matching the true discriminative rank)
3. **Classification:** $k$-NN with $k = \lfloor \log n \rfloor$ (rounded to nearest odd), evaluated via leave-one-out cross-validation
4. **Error:** Fraction of misclassified models

## Why the Parity Label Makes the Bound Tight

The squared energy distance between models $i, j$ using queries $S$ is:

$$\hat{\mathcal{E}}^2(f_i, f_j) = \sum_{q \in S} \sum_{\ell=1}^r \alpha_\ell(q) \cdot \mathbf{1}[\theta_{i,\ell} \neq \theta_{j,\ell}]$$

With the parity label, **every** dimension $\ell$ carries signal. If any dimension $\ell$ is never activated by the sampled queries (i.e., $\alpha_\ell(q) = 0$ for all $q \in S$), then dimension $\ell$ is invisible and the parity label cannot be recovered -- the classifier is stuck at chance on pairs that differ only in dimension $\ell$.

**Union bound:** The probability that at least one dimension is missed is:
$$P[\text{error} \geq 0.5] \leq \sum_{\ell=1}^r P[\text{dim } \ell \text{ missed}] = r \cdot \rho^m$$

This is **tight** because:
1. Missing any single dimension $\ell$ makes the parity unrecoverable (no proper subset of $\{1, \ldots, r\}$ determines parity)
2. For independent Bernoulli activations, the events $\{\text{dim } \ell \text{ missed}\}$ are approximately independent when $\rho^m$ is small

This contrasts with a model where $y = \theta_{f,1}$: only dimension 1 needs to be activated, so $P[\text{error} \geq 0.5] \approx \rho^m$ regardless of $r$.
