# Query Complexity for Classification via MDS Embeddings of Black-Box Generative Models

## 1. Project Overview

### Motivation

Given a collection of black-box generative models with associated labels (e.g., safe/unsafe, trained on sensitive data or not, model family A vs B), how many queries are needed to reliably classify a new model? This question is fundamental to model auditing, where query budgets are constrained by cost, latency, or API rate limits.

The Data Kernel Perspective Space (DKPS) framework embeds models into Euclidean space via MDS on pairwise dissimilarities between response distributions, enabling standard classification. Prior work establishes consistency of these embeddings (Acharyya et al., 2024) and their utility for inference (Helm et al., 2025). However, no existing result characterizes the *query complexity* — the number of queries $m$ required for reliable classification — or connects it to the structure of the classification problem.

### Main Contributions

**1. Discriminative factorization and field.** We introduce a decomposition of the squared energy distance $\mathcal{E}^2(P_f(q), P_{f'}(q)) = \sum_{\ell=1}^r \alpha_\ell(q) \phi_\ell(f, f')$ that separates query-dependent signal intensity from model-pair sensitivity. The minimal rank $r$ captures the intrinsic discriminative complexity of the problem. The *discriminative field* $\alpha: \mathcal{Q} \to [0,\infty)^r$ maps each query to its signal profile, and its *zero sets* $\mathcal{Z}_\ell$ identify where each discriminative direction is silent.

**2. Query complexity bound (Theorem 1).** For any classification problem with Bayes risk $L^* < 0.5$, the probability that the MDS representation fails to beat chance is at most $\sum_{\ell=1}^r \Pi_Q(\mathcal{Z}_\ell)^m + \gamma(n)$. This decays exponentially in $m$ at a rate governed by the geometry of the discriminative field. In the worst case ($r = M$), this recovers a coupon-collector bound requiring $m = O(M \log M)$ queries. When $r \ll M$, far fewer queries suffice: $m^* = O(\delta^{-1} \log r)$.

**3. Bayes-optimal convergence (Theorem 2).** When $g$ is injective and responses are conditionally independent across queries given the model, the MDS representation is asymptotically sufficient: $\inf_h L_{\psi Y}(h, Q) \leq L^* + \epsilon$ with high probability for $m \geq m^*$ and $n \geq n^*_\epsilon$. This is a substantially stronger result — the representation captures *all* discriminative information, not merely enough to beat chance.

**4. Corollary: explicit sample and query complexity.** For any $\epsilon > 0$, $m^* = \lceil \delta^{-1} \log(2r/\epsilon) \rceil$ queries and $n^* = \min\{n : \gamma(n) \leq \epsilon/2\}$ training models suffice for both results. The query complexity is logarithmic in $r$ and inverse in the minimum query mass $\delta$.

### Simplification at Temperature Zero

When generative models are queried with temperature $= 0$, each response is deterministic: $P_f(q) = \delta_{g(f(q))}$, a point mass. The energy distance simplifies to $\mathcal{E}^2(\delta_x, \delta_y) = 2\|x - y\|$, and the cumulative energy distance becomes

$$\mathcal{E}_m^2(f, f') = 2 \sum_{k=1}^m \|g(f(q_k)) - g(f'(q_k))\|.$$

Orthogonality reduces to exact equality of embedded responses: $q \perp (f, f')$ iff $g(f(q)) = g(f'(q))$. The discriminative factorization decomposes $\|g(f(q)) - g(f'(q))\|$ into query-dependent and model-pair-dependent components. No response replicates are needed ($r = 1$), eliminating the sampling axis entirely and isolating query complexity as the sole resource.

### Ideal Empirical Results

The experiments should demonstrate:

1. **Exponential decay.** Classification error as a function of $m$ decays as $\approx r\rho^m$, with the rate governed by $\rho = \max_\ell \Pi_Q(\mathcal{Z}_\ell)$. Plots of $\log(\text{error})$ vs $m$ should be approximately linear with slope $\log \rho$.

2. **Low-rank advantage.** Problems with small $r$ should reach high accuracy at much smaller $m$ than predicted by the worst-case $r = M$ bound. A side-by-side comparison of classification curves for problems with $r = 2$ vs $r = M$ should show a clear gap.

3. **Query distribution matters.** Concentrating $\Pi_Q$ on high-signal queries (small zero sets) should improve accuracy at small $m$, while spreading mass over orthogonal queries wastes budget. Comparing uniform $\Pi_Q$ vs informed $\Pi_Q$ vs adversarial $\Pi_Q$ (heavy on orthogonal queries) should show a monotone ordering.

4. **Effective rank is small in practice.** On real LLM collections, the energy distance tensor $\mathcal{E}^2(P_f(q), P_{f'}(q))$ should have a spectral gap, with effective rank $r \ll M$. This validates the assumption that low-rank structure exists in practice.

5. **Bayes convergence.** With injective $g$ and sufficient $m$, classification accuracy should approach the Bayes rate, not just beat chance. As $m$ grows, the gap between observed accuracy and $1 - L^*$ should close.

6. **Orthogonal vs relevant queries.** Replicating the qualitative finding of Helm et al. (2025) — relevant queries yield good classification, orthogonal queries do not — but now with the quantitative framework: the classification error under relevant queries has $\rho \approx 0$, while under orthogonal queries $\rho \approx 1$.

---

## 2. Experimental Plan

### 2.1 Synthetic Experiments

**Purpose:** Validate the theoretical predictions in a controlled setting where ground truth is known exactly.

#### Setup: Bernoulli-Weight Model

See `computational_model.md` for the full specification. Summary:

**Discriminative field.** The $M \times r$ field matrix has entries $\alpha_\ell(q) = \xi_{q\ell} \cdot w_{q\ell}$ where $\xi_{q\ell} \sim \text{Bernoulli}(p)$ and $w_{q\ell} \sim \text{Uniform}(0,1)$. The activation probability $p$ is the single tuning parameter; $\rho = 1 - p$.

**Models.** Each model $f$ has a latent type vector $\theta_f \in \{0,1\}^r$. Class label $y = \theta_{f,1}$. Dimensions $2, \ldots, r$ are drawn i.i.d. Bernoulli(0.5), creating within-class variation. Signs $s_\ell(f) = 1 - 2\theta_{f,\ell} \in \{+1, -1\}$.

**Responses.** $g(f(q)) = \sum_\ell \sqrt{\alpha_\ell(q)} \cdot s_\ell(f) \cdot d_\ell$ where $d_1, \ldots, d_r$ are orthonormal directions in $\mathbb{R}^p$ (via QR decomposition). Orthogonality ensures the squared distance factorizes exactly: $\|g(f(q)) - g(f'(q))\|^2 = \sum_\ell \alpha_\ell(q) \cdot (s_\ell(f) - s_\ell(f'))^2$.

**Key properties:**
- Zero sets: $\mathcal{Z}_\ell = \{q : \xi_{q\ell} = 0\}$, so $\Pi_Q(\mathcal{Z}_\ell) = 1 - p$ under uniform $\Pi_Q$.
- Within-class variation from dimensions $2, \ldots, r$ makes classification harder as $r$ grows (requires $n \to \infty$ for large $r$).
- The parameter $p$ interpolates between sparse ($p = 1/r$, coupon-collector) and dense ($p = 1$, single query suffices).

#### Experiments

**Exp 1: Error vs $m$ for varying $r$.** ✅ COMPLETE
- Fix $M = 100$, $n = 200$, $p = 0.3$ ($\rho = 0.7$), uniform $\Pi_Q$.
- Sweep $r \in \{2, 5, 10, 25, 50, 100\}$, $m \in \{1, 2, 5, 10, 20, 50, 100, 200\}$, 100 reps.
- **Results:** For $r \leq 10$, $P[\text{error} \geq 0.5]$ decays exponentially in $m$ (coupon-collector behavior). For $r \geq 25$, finite-$n$ effects dominate: within-class variation from $r - 1$ noise dimensions makes classification hard with $n = 200$ models, consistent with the theory requiring $n \to \infty$.
- **Plot:** Panel (a) of Figure 1.

**Exp 2: Error vs $m$ for varying $\rho$.** ✅ COMPLETE
- Fix $r = 5$, $M = 100$, $n = 200$, uniform $\Pi_Q$.
- Sweep $p \in \{0.1, 0.3, 0.5, 0.8\}$ (i.e., $\rho \in \{0.9, 0.7, 0.5, 0.2\}$), 100 reps.
- **Results:** Larger $\rho$ (sparser signal) requires more queries. $\rho = 0.9$ needs $m \approx 10$ to eliminate high-error events; $\rho = 0.2$ has $P[\text{error} \geq 0.5] = 0$ at all $m$. Slopes on log scale vary with $\log(1-p)$ as predicted.
- **Plot:** Panel (b) of Figure 1.

**Exp 3: Effect of query distribution $\Pi_Q$.** ✅ COMPLETE
- Fix $r = 5$, $M = 100$, $p = 0.3$, $n = 200$, 100 reps.
- Three distributions: uniform, signal-concentrated (top 30% by $\sum_\ell \alpha_\ell(q)$), orthogonal-concentrated (queries with $\alpha_\ell(q) = 0$ for all $\ell$).
- **Results:** Signal-concentrated converges fastest (accuracy ~0.93 at $m = 1$); uniform is intermediate (~0.84 at $m = 1$); orthogonal-concentrated is slowest (~0.56 at $m = 1$) but eventually catches up at large $m$ since the distribution still has some mass on signal queries.
- **Plot:** Panel (c) of Figure 1.

**Exp 4: Error vs $n$ (sample complexity).**
- Fix $r = 5$, $m = 50$ (large enough that query bound is satisfied), uniform $\Pi_Q$.
- Vary $n$ from 10 to 500.
- **Expected result:** Error decreases with $n$, characterizing $\gamma(n)$. Rate depends on TV$(P_0, P_1)$.
- **Plot:** Classification error vs $n$.

**Exp 5: Bayes convergence (Theorem 2 validation).**
- Construct a problem where $L^*$ is known exactly (e.g., overlapping class distributions with computable Bayes error).
- Fix large $m$ (all directions activated), vary $n$.
- **Expected result:** $\inf_h L \to L^*$ as $n \to \infty$, confirming Theorem 2.
- **Plot:** $\inf_h L$ vs $n$ with horizontal line at $L^*$.

### 2.2 Real LLM Experiments

**Purpose:** Demonstrate that (a) low-rank discriminative structure exists in real model collections, (b) the theoretical predictions hold qualitatively on real data, (c) the framework is practically useful for model auditing.

#### Infrastructure

- **Models:** ~50 models from a publicly accessible source. Two options:
  - *Option A (preferred):* Fine-tune a base model (e.g., Llama-3-8B or Mistral-7B) with LoRA on ~25 different data mixtures, producing 50 models (25 per class). Label = property of the fine-tuning data (e.g., contains domain-specific data or not). This mirrors Helm et al. (2025).
  - *Option B:* Use publicly available model collections from HuggingFace. Label = model family, quantization level, or safety alignment status. Less controlled but requires no training.
- **Queries:** $M = 200$–$500$ queries. Draw from existing benchmarks (MMLU, HellaSwag, TruthfulQA) or construct domain-specific queries. Include queries expected to be orthogonal (unrelated to the classification task) and queries expected to be relevant.
- **Embedding function $g$:** Use an open embedding model (e.g., nomic-embed-text-v1.5) to embed each response into $\mathbb{R}^{768}$. Since temperature $= 0$, each model-query pair produces a single deterministic response, which is then embedded.
- **Compute:** One forward pass per model per query through the generative model (for response), plus one forward pass through the embedding model (for $g$). For 50 models $\times$ 500 queries = 25,000 inference calls. Feasible on a single GPU in a few hours for 8B-parameter models with quantization.

#### Experiments

**Exp 6: Effective rank of the energy distance tensor.**
- Compute the full $M \times \binom{n+1}{2}$ matrix of pairwise energy distances $\mathcal{E}^2(P_{f_i}(q), P_{f_j}(q))$ for all queries and model pairs. At temperature zero, this is $2\|g(f_i(q)) - g(f_j(q))\|$.
- Compute the SVD (or nonnegative matrix factorization) of this matrix.
- **Expected result:** A clear spectral gap, with $r_{\text{eff}} \ll M$. Report the number of singular values needed to capture 90%, 95%, 99% of the Frobenius norm.
- **Plot:** Singular value spectrum (scree plot) with elbow.

**Exp 7: Classification accuracy vs $m$ (main result).**
- For varying $m = 1, 2, 5, 10, 20, 50, 100, 200, \ldots, M$:
  - Sample $Q$ uniformly from $\mathcal{Q}$, $|Q| = m$.
  - Compute pairwise $\mathcal{E}_m$ over all $n + 1$ models.
  - Run MDS. Train a classifier (LDA, $k$-NN, or SVM) via leave-one-out cross-validation.
  - Record classification accuracy. Repeat 100 times (different random $Q$).
- **Expected result:** Sigmoidal accuracy curve that rises from $\sim 0.5$ (chance) and plateaus near $1 - L^*$. The transition point should occur at $m \approx m^*$.
- **Plot:** Mean accuracy $\pm$ std vs $m$.

**Exp 8: Relevant vs orthogonal queries.**
- Partition $\mathcal{Q}$ into "relevant" and "orthogonal" subsets. A simple heuristic: for each query $q$, compute $\sum_{i \in \text{class 0}, j \in \text{class 1}} \mathcal{E}^2(P_{f_i}(q), P_{f_j}(q))$; queries with large values are relevant, small values are orthogonal.
- Repeat Exp 7 but drawing $Q$ only from relevant queries, only from orthogonal queries, and uniformly.
- **Expected result:** Relevant queries achieve high accuracy at small $m$; orthogonal queries remain near chance regardless of $m$; uniform is intermediate.
- **Plot:** Accuracy vs $m$, three curves (relevant, orthogonal, uniform).

**Exp 9: Comparison with baselines.**
- Compare classification via MDS embeddings against:
  - *Random feature baseline:* classify based on raw response features (concatenated embeddings) without MDS.
  - *Single-query classifiers:* train a classifier on $g(f(q))$ for each $q$ individually, report the best.
  - *Benchmark-based classification:* if available, classify based on benchmark scores (e.g., MMLU accuracy) rather than distributional embeddings.
- **Expected result:** MDS embedding should match or outperform baselines, especially at small $m$, because it leverages the metric structure. At large $m$, all methods should converge.
- **Plot:** Accuracy vs $m$ for each method.

**Exp 10: Estimating $\rho$ and predicting $m^*$.**
- From the empirical data, estimate $\rho$ via the effective rank analysis (Exp 6) or by fitting the theoretical decay curve $r\rho^m$ to the observed error-vs-$m$ data.
- Use the estimated $\rho$ to predict $m^*$ for a target accuracy, and verify against the actual accuracy curve.
- **Expected result:** The predicted $m^*$ should roughly match the empirical transition point, validating the bound's practical relevance.
- **Table:** Predicted vs observed $m^*$ for several target accuracy levels.

### 2.3 Experimental Priorities

Ranked by importance for the submission:

1. ✅ **Exp 1 + 2 + 3** (synthetic: Figure 1) — validates the core theorem and discriminative field concept
2. **Exp 7** (real: accuracy vs $m$) — demonstrates practical relevance
3. **Exp 6** (real: effective rank) — validates low-rank assumption
4. **Exp 8** (real: relevant vs orthogonal) — validates discriminative field concept on real data
5. **Exp 9** (real: baselines) — positions contribution relative to alternatives
6. **Exp 5** (synthetic: Bayes convergence) — validates Theorem 2
7. **Exp 4** (synthetic: sample complexity) — validates $\gamma(n)$
8. **Exp 10** (real: predicting $m^*$) — nice-to-have, shows bound is practical
