# Query Selection for Efficient Black-Box Model Classification

## Motivation

Classifying black-box generative models (e.g., detecting whether an LLM has been fine-tuned on sensitive data) requires querying each model and comparing their responses. In a typical workflow, a labeled pilot set of $n$ models is queried on a large pool of $M$ queries, and the cached responses are used to estimate the discriminative structure. The cost of classifying a *new* model scales with the number of queries $m$ it must answer — each query is an API call. Identifying a small, high-quality query set $Q^*$ of size $m \ll M$ that preserves classification accuracy therefore directly reduces the per-model audit cost for all future models.

The discriminative factorization framework estimates a per-query signal magnitude $|\hat{\alpha}_{q,\ell}|$ from the pilot set. This information can guide query selection: instead of sampling queries uniformly at random, we can preferentially select queries that carry discriminative signal. We describe three selection strategies and evaluate them on three real-world classification tasks.

## Methods

### 1. Uniform Sampling (Baseline)

Sample $m$ queries uniformly at random from the pool of $M$ available queries. This is the implicit default in the DKPS framework — no pilot information is used for query selection.

### 2. Estimated Signal Sampling

Use the discriminative factorization to partition queries into an estimated *signal set* $\hat{\mathcal{S}}$ and an estimated *zero set* $\hat{\mathcal{Z}}$. The partition is obtained by fitting a two-component Gaussian mixture model (GMM) to the SVD loadings $|U_{q,\ell}|$ of the per-query energy tensor $E$: queries assigned to the higher-mean component are classified as signal. Sample $m$ queries uniformly from $\hat{\mathcal{S}}$.

This method exploits the factorization's zero-set structure but does not consider query redundancy — two queries with nearly identical loading profiles contribute redundant information.

### 3. Forward Stepwise Selection

Treat query selection as forward variable selection in a linear model predicting class membership from pairwise distances. For each model pair $(i,j)$, define a response variable $z_{ij} = \mathbf{1}[y_i \neq y_j]$ and a predictor $E_{q,ij} = \|g(f_i(q)) - g(f_j(q))\|^2$ for each query $q$. At each step, add the query with the largest partial correlation with the residual of $z$ after projecting out the contributions of previously selected queries.

This is equivalent to Gram-Schmidt orthogonalization on the columns of $E$ with respect to $z$, ensuring each selected query contributes maximally non-redundant discriminative information. The selected queries are then used with classical MDS and LDA for classification.

**Computational cost.** Stepwise selection requires only the energy tensor $E \in \mathbb{R}^{M \times \binom{n}{2}}$, which is already computed during the estimation phase. Each step involves $O(M)$ inner products, making the total cost $O(m \cdot M \cdot \binom{n}{2})$ — negligible compared to the MDS and classification steps.

## Experimental Setup

We evaluate on three real-world tasks using pre-computed response embeddings:

| Task | Models | Queries | Description |
|------|--------|---------|-------------|
| Motivating | 100 LoRA adapters | 200 | Detecting sensitive training data (Politics & Government) |
| System Prompt | 100 LLM personas | 200 | Detecting covert persuasion bias in system prompts |
| RAG | 120 RAG systems | 200 | Detecting unauthorized document store connections |

For each task, we use $n = 80$ labeled models for both estimation and classifier training, with the remaining models held out for testing. Query selection is performed once per train/test split using only the training models. Classification uses classical MDS (8 components) followed by LDA. Results are averaged over 100 random train/test splits.

## Results

The figure below shows mean classification error as a function of query budget $m$ for each method across the three tasks.

**Key findings:**

- **Stepwise dominates at small $m$.** At $m = 2$, stepwise achieves 10% error on the motivating task (vs. 28% for uniform) and 11% on the system prompt task (vs. 28% for uniform).

- **Estimated signal is intermediate.** Sampling from the estimated signal set consistently outperforms uniform sampling, confirming that the GMM-based partition captures real discriminative structure.

- **Stepwise's advantage is largest when it matters most.** The gap between stepwise and uniform is widest at small $m$ (the cost-sensitive regime) and narrows as $m$ grows and all methods converge.

- **Practical cost reduction.** On the system prompt task, stepwise at $m = 5$ matches the accuracy that uniform sampling achieves at $m = 50$ — a $10\times$ reduction in per-model query cost.

## Conclusion

Forward stepwise selection on the energy tensor provides a principled, computationally cheap method for selecting a small, targeted query set. Combined with LDA classification, it achieves the same accuracy as uniform sampling with an order of magnitude fewer queries, directly reducing the cost of black-box model auditing.
