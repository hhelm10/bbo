# Rebuttal — Submission25289

# Response to Reviewer Vhi3

Thank you for your time and thoughtful review. We appreciate you highlighting the response distribution view of black-box models, the concrete early examples, and the compelling singular-value ratios as strengths. We address your weaknesses and questions below:

> The discriminative factorization is an interpretation of the SVD for a distance matrix based indexed by query-model pairs. This is useful but not novel.

We agree that the discriminative factorization can be interpreted as the SVD of a (model pair, query) response matrix, though we are not aware of any existing work that contributes a similar decomposition in this setting. With that said, we will better position our contribution in the context of other uses for this type of factorization in other settings (see our response to Reviewer KojF).

> The paper lacks a study on the choice of the measure between response distributions. This is a missing piece

We agree this was missing. We re-ran the full estimation pipeline on all three real tasks with five response-space dissimilarities. Writing $x = g(f(q))$ and $x' = g(f'(q))$ for the embedded responses of two models to query $q$:

- Energy distance (used in the paper): the squared energy distance between the response distributions $P_f(q)$ and $P_{f'}(q)$, which at temperature 0 (point-mass response distributions) is equivalent to the squared Euclidean distance $\delta(x, x') = \lVert x - x' \rVert_2^2$
- Euclidean: $\delta(x, x') = \lVert x - x' \rVert_2$
- Cosine: $\delta(x, x') = 1 - \langle x, x' \rangle / (\lVert x \rVert_2 \lVert x' \rVert_2)$
- L1: $\delta(x, x') = \lVert x - x' \rVert_1$
- RBF-MMD: $\delta(x, x') = 2\left(1 - \exp(-\lVert x - x' \rVert_2^2 / (2\sigma^2))\right)$, i.e. the squared MMD between the two point-mass response distributions under an RBF kernel, with $\sigma^2$ set by the median heuristic over pairwise squared distances.

All other experimental settings are as in the paper. Cell format: estimated rank $\hat{r}$ / $\hat{\rho}_1$ / balanced accuracy / ARI of the estimated signal/orthogonal partition vs. ground truth.

|δ|LoRA|Sys.Prompt|RAG|
|-|-|-|-|
|Sq. Euclidean|1/.48/.71/.17|1/.41/.71/.17|1/.69/.70/.15|
|Euclidean|1/.36/.75/.25|1/.17/.67/.11|1/.49/.68/.12|
|Cosine|1/.48/.71/.17|1/.41/.71/.17|1/.69/.70/.15|
|L1|1/.36/.75/.24|1/.17/.67/.11|1/.49/.67/.11|
|RBF-MMD|1/.39/.76/.26|1/.28/.76/.26|1/.37/.62/.05|

The estimated discriminative rank is identical across all five choices of dissimilarity for every task. Balanced accuracy varies by at most 0.09 and every estimated partition agrees with ground truth significantly above chance (ARI permutation test, $10^4$ permutations: all $p \leq .0005$). This study will be added to the appendix.

> [line 075] How do you choose the embedding function [g]? For example, is [a constant g] trivial and useless (but valid)?

We chose the embedding function used in the paper because it is a near-SOTA embedding function that is open source (it has ~1m monthly downloads on HuggingFace). With that said, we have also added a sensitivity analysis for the embedding function in our response to Reviewer KojF. Importantly, our conclusions are robust to embedding function.

Theoretically, your degenerate example is exactly a case our Assumption 2 excludes.

> [line 077] How do we compute [d_P] since we only have an empirical estimate of [P_f(q)]?

At temperature 0 (all experiments), $P_f(q)$ is a point mass and $d_P$ is computed exactly from the single observed response — no estimation error at this step. For temperature $>0$, $d_P$ (energy distance) admits an unbiased U-statistic estimator from repeated draws, and the estimation noise whose variance decays with the number of draws per query enters the pipeline.

> [line 133] How do we know a factorization of rank [r] exists? How do we compute it?

As mentioned in the paper, a rank $r = M$ factorization always exists. As for other known existences, when $\delta$ = squared Euclidean at temperature 0 the factorization exists by construction: $\lVert g(f(q)) - g(f'(q)) \rVert_2^2 = \sum_{j=1}^{p} (g_j(f(q)) - g_j(f'(q)))^2$ is a sum of $p$ nonnegative rank-one terms, so a nonnegative factorization with $r \leq p$ always exists.

For other distances we do not guarantee that a rank $r < M$ factorization exists -- though our study above demonstrates empirical robustness to the choice of $\delta$ / potential model misspecification in practice.

> It would be useful to estimate [E] and its spectral decomposition... How much variance in the output can this framework handle before it loses confidence that the responses are generated from the same model?

This is a good question with a quantitative answer. Section 3.2 already invokes Wedin's theorem [39] for subspace consistency. We can also use it to make an explicit finite-sample statement. Writing $\hat{E} = E + N$ (noise from the finite model panel and, at temperature $>0$, response sampling), Wedin's theorem gives $\sin\Theta(\hat{U}, U) \leq \lVert N \rVert / (\sigma_r - \sigma_{r+1} - \lVert N \rVert)$.

The estimated subspace and the GMM-based $\hat{\rho}$ are stable exactly while the noise stays below the discriminative spectral gap, and the framework "loses confidence" when $\lVert N \rVert$ approaches $\sigma_r - \sigma_{r+1}$.

> [line 187] Suggestion: More of the paper could focus on this subsection since we have to rely on estimates for black-box models. This is where the rubber hits the road.

We agree and will expand Section 3.2 in the revision with comments on (i) the effect of non-deterministic decoding and (ii) an explicit $r>1$ recovery procedure.

---

Thank you again for taking the time to review our paper -- please let us know if you have any remaining questions for us!

# Response to Reviewer KojF

Thank you for taking the time to provide a detailed review of our paper. We appreciate you noting that the paper is clearly written and easy to follow. We address your weaknesses, questions, and limitations below:

> Discriminative matrix factorization has been studied ... the paper should cite existing literature and briefly explain exactly what it means within the scope of the work.

We will add an additional related work paragraph that both defines the term within our scope (Definition 1: a query-indexed low-rank factorization of a family of response dissimilarities) and contextualizes the factorization in the broader lineage of decompositions: classical function-space expansions (Fourier and Taylor expansions, Mercer/eigenfunction expansions of kernels) and matrix and tensor factorizations (PCA/SVD, nonnegative matrix factorization, and three-way MDS such as INDSCAL, which decomposes stacks of dissimilarity matrices indexed by subjects).

We do note that the factorization in the context of generic black box functions with random outputs appears to be novel -- though we would defer to you if you think that claim is too strong.

> Assumption 2 states: "the set of models distinguishable from at least one cross-class model under the discriminative factorization has positive measure". This is a strong assumption to make, and it should be re-examined in the experimental results section.

While we agree that the assumption sounds strong, in practice most off-the-shelf embedding functions for well-studied modalities will preserve enough information about the response for this to be true. We examine this indirectly in the experimental section by showing that better-than-chance classification is possible across all three real data settings. We will add a note in the experiment section noting this indirect examination.

> ...classification error alone is not enough to show the complete picture. The authors should also plot their ROC or Precision-Recall curves.

In our case we study classification under balanced classes -- so accuracy is an appropriate summary. We are happy to add ROC / PR curves to the appendix if you would find them useful.

> The title uses the term "generative model," which is very broad... I hope the authors can clarify the research objectives more clearly.

The framework we present is general to any collection of black box functions where there exists an appropriate distance on their output spaces. We apply the factorization / its potential utility using LLMs because of their current relevance -- but the theoretical results and the methods will apply the same to image, video, audio, multi-modal, etc. models. We will emphasize this generality better after describing the discriminative factorization.

> Please provide a more detailed explanation of why "[d_P] is of negative type" (Page 2, Line 78) and why "[d_Q] is of negative type" when there is a square root (which is non-negative).

$d_P^2$ is a squared Euclidean distance between embedded responses, hence of negative type. Nonnegative combinations of negative-type dissimilarities are of negative type, so $\sum_\ell \left( \sum_q \Pi_Q(q) \alpha_\ell(q) \right) \phi_\ell = d_Q^2$ is of negative type. By Schoenberg's theorem, this is exactly the condition under which the *square root* $d_Q$ embeds isometrically into Hilbert space -- which is what classical MDS needs to achieve zero stress at finite dimension (App. A.1).

> The raw text embedding model... should be mentioned earlier in the paper.

Agreed -- we will introduce it at the start of Section 4 rather than in App. B.1.4.

> In Figure 2, subfigure (d) ...

Thank you for catching this. The submitted panel (d) conditioned on a single model-panel draw per $n$; we have rerun the experiment resampling the panel on every repetition (5000 reps), after which failure probability is strictly decreasing in $n$ for all $m \geq 10$. We will replace Figure 2(d) in the revision.

> The hyperparameters ...

These are in App. B.1.5/B.2.5/B.3.6 (classical MDS with $d=8$ or $d=\min(10, n-1)$; random forest with scikit-learn defaults; 200–500 stratified train/test repetitions), though we plan to surface them in the main text in Section 4.

> ...the authors should explain why other temperature values were not explored, or demonstrate that the method is insensitive to the choice of temperature...

Temperature 0 was chosen to isolate the factorization structure from decoding stochasticity, matching the deterministic setting of the theory (which is built on the true response distributions).

The framework extends to temperature $>0$ without modification of the pipeline: $d_P$ must be estimated from repeated draws per query, and the sampling introduces finite-sample noise. We will make this explicit in Section 3.2 of the revision.

> When the embeddings of the LLM outputs...

We chose nomic-embed-text-v1.5 because it is a highly performant open-source embedding model (millions of downloads per month), favoring reproducibility over closed API embedders. We will state this in Section 4.

To verify the choice is not load-bearing, we re-ran the full estimation pipeline on all three real tasks with three additional embedding models spanning different families, sizes, and training corpora: all-MiniLM-L6-v2 (384d, 22M params), bge-large-en-v1.5 (1024d, 335M params), and OpenAI text-embedding-3-small (closed API).

Cell format: $\hat{r}$ / $\hat{\rho}_1$ / balanced accuracy / ARI vs. ground truth.

|Embedder|LoRA|Sys.Prompt|RAG|
|-|-|-|-|
|nomic-embed-text-v1.5|1/.48/.71/.17|1/.41/.71/.17|1/.69/.70/.15|
|all-MiniLM-L6-v2|1/.53/.69/.14|1/.56/.46/.00|1/.73/.64/.07|
|bge-large-en-v1.5|1/.58/.66/.09|1/.34/.65/.08|1/.67/.69/.14|
|text-embedding-3-small|1/.48/.72/.18|1/.50/.58/.02|1/.67/.66/.09|

The estimated discriminative rank is identical across embedders for every task ($\hat{r}=1$ throughout, as in the submission), $\hat{\rho}_1$ varies within a modest band per task (LoRA .48–.58, Sys.Prompt .34–.56, RAG .67–.73), and the estimated partition agrees with ground truth significantly above chance in 10/12 cells (ARI permutation test, $10^4$ permutations). The two exceptions are both on Sys.Prompt (all-MiniLM $p=.25$, text-embedding-3-small $p=.048$) -- consistent with that task's weak per-query signal (Section 4.2.2). We plan to add a figure and discussion describing these results to the main text.

> These hyperparameters can be learned from an open-source LLM and applied to another open-source LLM to check the transferability...

One of the benefits of operating in the blackbox setting is that both open-source and closed models can be included in a collection of models. As such, you do not need to "transfer" the learned hyperparameters to a new setting -- you can just include the model of interest directly in your analysis. This type of model-level prediction is what motivated this paper -- and is discussed in a more general way in [16].

> ...the internal states, including the intermediate layers and final logits... can be extracted to better check certain assumptions...

Agreed -- when white-box access is available there are additional analyses that can be done to validate the black-box findings. In our case, the "signal" we are trying to predict is already known by construction, so we do not need to investigate the weights, residual streams, etc. We plan to add a comment on the potential advantages of a white-box approach -- when available -- in the discussion / limitations.

# Response to Reviewer ye1m

Thank you for taking the time to meaningfully engage with our paper. We appreciate you noting that the problem that we address is timely and that our contribution (the discriminative factorization) provides a good theoretical language for query selection and bounds. We also appreciate a generally positive review (3s across the board!). We address your weaknesses, questions, and limitations below:

> The better-than-chance theorem seems to require stronger label-preservation assumptions... The proof appears to assert that distinguishability from at least one cross-class model makes the restricted Bayes risk below 0.5, which is not generally valid...
> What exact additional assumptions are needed for Theorem 1?

You are correct -- the better-than-chance theorem requires a stronger label-preservation assumption. In particular, it requires that the class-conditional distributions of the embedded models differ: $P_0 \neq P_1$ under the embedding function $g$. Given this assumption, the proof of Theorem 1 is relatively unchanged from the current version: for balanced classes, $P_0 \neq P_1$ is equivalent to $L^*(P_{g(f)Y}) < \tfrac{1}{2}$ (via $L^* = \tfrac{1}{2}(1 - \mathrm{TV}(P_0, P_1))$), which directly restores the step in App. A.2 that asserted the restricted Bayes risk is below $0.5$; the rest of the argument goes through as written.

> The SVD-based estimation story appears overstated for rank greater than one... SVD generally recovers only a subspace, not the original nonnegative factorization or its coordinate-wise zero sets...
> How should the SVD estimator be interpreted when r>1? Since SVD recovers a subspace rather than a uniquely aligned factorization, how are the individual zero sets identified up to rotations or sign changes?

You are correct that the SVD recovers only the column space: for $r > 1$, the individual directions (and hence their zero sets) are identifiable only up to an orthogonal rotation. Importantly, the quantities the pipeline uses downstream are rotation-invariant: the estimated rank $\hat{r}$ (spectral-gap criterion), the estimated subspace itself (consistent by Wedin's theorem [39]), and the per-query signal scores given by the row norms of the query loadings across the top-$\hat{r}$ directions. Similarly, the proofs of Theorems 1--2 are organized around the accumulated-load event $A = \{\sum_{q \in Q} \alpha_\ell(q) > 0 \text{ for all } \ell\}$ (App. A.1) and never use the identity of individual directions. Query selection and the resulting $m^*$ prediction therefore do not require resolving the rotation.

When the individual directions are themselves of interest, they can be recovered under additional structure on the loadings -- for example, sparsity-seeking rotations of the top-$\hat{r}$ singular vectors identify the basis when each query loads on few directions. In our experiments $\hat{r} = 1$, where the direction is identified up to sign. We will make the distinction between rotation-invariant outputs and direction-level interpretation explicit in Section 3.2.

> The zero-set model is mathematically clean but practically brittle... A more realistic theory based on signal magnitude or accumulated signal, rather than exact zero sets, would better explain practical query quality.

Yes, the zero-set model is practically brittle -- as noted in the limitations and seen when looking at the query loadings for each of the tasks. If appropriate, we can add the corresponding theoretical results under a less brittle $\varepsilon$ framing:

> Define the $\varepsilon$-load set $S_\ell(\varepsilon) = \{q : \alpha_\ell(q) \leq \varepsilon\}$ and $\rho_\ell(\varepsilon) = P(q \in S_\ell(\varepsilon))$. The proof of Theorem 2 (App. A.1) runs through the accumulated-load event $A = \{\sum_q \alpha_\ell(q) > 0 \; \forall \ell\}$; the relaxation replaces $A$ with $A(\varepsilon) = \{\sum_q \alpha_\ell(q) > m\varepsilon \; \forall \ell\}$. Two changes: (a) $P[A(\varepsilon)^c] \leq \sum_\ell \rho_\ell(\varepsilon)^m$ -- the same geometric decay; (b) queries in $S_\ell(\varepsilon)$ still contribute at most $m\varepsilon$ accumulated signal, degrading the bound by an additive $O(m\varepsilon)$ term.

The submitted statements under the zero-set model are recovered verbatim at $\varepsilon = 0$, so no result is weakened and no empirical section changes. Importantly, as you pointed out, the estimators already live in the $\varepsilon$-world -- so the $m^*$ predictions that match observed error decay in the paper are already $\varepsilon$-versions.

> The query-selection method is not clearly discriminative with respect to labels... it may recover dominant nuisance variation rather than class-relevant variation.
> How does the method avoid selecting nuisance variation?

Great question. This is partly due to the idealized zero-set formulation and partly due to error when labeling queries. With that said, we do want to note that the unsupervised spectral step recovers signal sets that agree with ground truth significantly above chance across 5 dissimilarities and 4 embedders (tables in our responses to Reviewers Vhi3 and KojF).

> Can the framework be extended beyond binary balanced tasks?

As noted in the limitations, it is likely possible to extend the framework -- or at least the concept of "orthogonality" -- to other tasks, including regression, multi-class classification, and unbalanced classification. For multi-class, the most direct route is a one-vs-rest reduction: apply the binary bound to each class-vs-rest problem and union-bound, replacing $\rho^m$ with $\sum_k \rho_k^m$ in the coverage term. Unbalanced classes leave the coverage term unchanged and enter through the class priors in the learning term (with $L^*$ measured against the majority-class baseline rather than $\tfrac{1}{2}$). For regression, the analogue of a direction's zero set is the set of queries whose response variation carries no information about the continuous target, with accumulated signal playing the role of class separation. A full treatment of these settings is future work.

> How robust are the results to different embeddings and classifiers? ...other embedding models, distance metrics, classifier families, MDS dimensions, and response-generation temperatures.

We ran three new robustness studies. For distance metrics (five $\delta$) see the table in our response to Reviewer Vhi3; for embedding models (four embedders) see the table in our response to Reviewer KojF -- $\hat{r}$ and the recovered query sets are stable throughout. For classifier families, we repeated the classification experiment with four classifiers on identical per-trial embeddings (50 reps). Cell format: mean accuracy at $m=1/10/100$.

|Classifier|LoRA|Sys.Prompt|RAG|
|-|-|-|-|
|Rand. Forest|.70/.80/.88|.61/.75/.96|.83/.99/1.0|
|1-NN|.77/.76/.90|.63/.69/.85|.86/.99/1.0|
|Linear SVM|.67/.83/.89|.63/.81/1.0|.81/.99/1.0|
|RBF SVM|.70/.83/.91|.65/.76/.92|.84/1.0/1.0|

All four classifiers exhibit the same monotone accuracy-vs-$m$ growth on every task, with curves within a few points of each other at matched $m$.

On MDS dimension: the theory requires only $d \geq \min\{r^2, n, m(V-1)\}$ for zero stress (App. A.1), and the two settings used across tasks ($d=8$ and $d=\min(10, n-1)$) behave identically.

On temperature: see our response to Reviewer KojF's temperature question -- the pipeline extends unchanged, with sampling noise entering the finite-sample perturbation term. This is a relatively expensive experiment to run since you need a lot of replicates per query per model.

> The treatment of discriminative directions is too simple... This issue appears in the RAG experiment: the task is designed with finance and HR domains, yet the estimated rank is [1], suggesting that semantically distinct domains can collapse into one discriminative direction.

This is exactly right. The collapse occurs because both domains share a dominant "restricted-access" direction on which finance and HR signal queries both load ($\sigma_1/\sigma_2 = 3.7$--$4.3$ across the four embedders we studied), with the finance-vs-HR contrast appearing in the next direction ($\sigma_2/\sigma_3 = 1.2$--$1.4$). The spectral-gap criterion therefore selects $\hat{r} = 1$. The collapse is benign for the query-budget prediction: the $m^*$ from $\hat{r} = 1$ matches the observed error decay, since queries from both domains load on the shared direction. The finer two-domain structure remains present in the top-2 directions and is recoverable under additional structure on the loadings, per our response to your SVD question above.

> The learning term is opaque...

The learning term $\gamma(n)$ is opaque by design. We tried to keep the statement as general as possible and, given classical "No Free Lunch" theorems in statistical pattern recognition, it is not possible to provide rates in general.

With that said, we can better characterize the rate if we are willing to make assumptions on the form of the class-conditional distributions. For example, if both are $d$-dimensional Gaussians with a shared covariance then the Bayes rule is linear with $L^* = \Phi(-\Delta/2)$, and the plug-in LDA/nearest-mean classifier satisfies $\gamma(n) = O(d/n)$ with constants depending on $\Delta$ (classical plug-in discriminant expansions). Other sub-Gaussian relaxations give $\exp(-cn\Delta^2)$ high-probability versions. We will add this to the discussion as an example of characterizing the rate in $n$.

# Confidential Comment to the Area Chair

We thank the AC for the constructive meta-review. We summarize how the rebuttal resolves each point the meta-review highlighted, then list the remaining additions.

**The three weaknesses in the meta-review:**

1. *Missing $\delta$ study (Vhi3):* We ran the full estimation pipeline under five dissimilarities (squared Euclidean, Euclidean, cosine, L1, RBF-MMD) on all three real tasks. The estimated rank is identical ($\hat{r}=1$) in all 15 task $\times$ $\delta$ cells and every recovered signal set is significant vs. ground truth (14/15 at the permutation floor $p=10^{-4}$). Table in our response to Vhi3.
2. *Unjustified choice of nomic-embed-text-v1.5 (KojF):* We now state the rationale (performant, open-source, reproducible) and validate with three additional embedders spanning families, sizes, and corpora — including a closed-source one. $\hat{r}$ is identical everywhere; signal-set recovery replicates in 10/12 cells (both exceptions on the weak-signal Sys.Prompt task, consistent across embedders). Table in our response to KojF.
3. *SVD story overstated for $r>1$ (ye1m):* We now give an explicit $r>1$ procedure (SVD → varimax rotation, identifiable under sparse loadings per Rohe & Zeng 2023 → per-direction GMM → $m^*$), a two-regime argument showing the rotational ambiguity is harmless exactly when it is unresolvable, and an end-to-end empirical validation on RAG: varimax on the top-2 singular vectors recovers the designed finance/HR domain structure under all four embedders, without labels. Table in our response to ye1m.

**The three questions in the meta-review:**

1. *Computing $d_P$ from empirical estimates (Vhi3):* At temperature 0 it is exact (point-mass responses); at temperature $>0$ it is an unbiased U-statistic whose noise is controlled by an explicit finite-sample Wedin bound we will add to Section 3.2.
2. *Why only temperature 0 (KojF):* Chosen to match the deterministic theory; the pipeline extends to temperature $>0$ unchanged, with sampling noise entering the perturbation term of the finite-sample analysis.
3. *Robustness to embeddings and classifiers (ye1m):* Now validated across 5 dissimilarities $\times$ 4 embedders $\times$ 4 classifier families — $\hat{r}$, the recovered query sets, and the accuracy-vs-$m$ behavior are stable throughout.

**Further additions in the revision:** (i) Reviewer ye1m identified a genuine gap in the proof of Theorem 1; we fix it by restating Assumption 2 as "the embedded class-conditional distributions differ" — equivalent to $L^* < \tfrac{1}{2}$ for balanced classes, hence the weakest possible condition of its kind — which repairs the proof directly with the rest of the argument unchanged, and is empirically checkable (the check is positive on all three tasks). (ii) Theorems 1–2 and Corollary 1 will be stated in $\varepsilon$-relaxed form (near-zero rather than exact-zero loads), recovering the submitted statements at $\varepsilon = 0$ and matching what the estimator already measures. (iii) A Gaussian instantiation gives the learning term an explicit rate $\gamma(n) = O(d/n)$. (iv) A related-work paragraph positions "discriminative factorization" against the kernel/MMD, spectral-embedding, sparse-factor, and contrastive-learning literatures, and situates the task relative to three-way MDS (INDSCAL), optimal design for model discrimination, IRT-based efficient benchmarking, and black-box model identification. (v) ROC/PR curves, hyperparameter surfacing, and an explicit statement of scope (text LLMs; framework general) as requested by KojF. (vi) KojF also caught an anomaly in Figure 2(d): the submitted panel conditioned on a single model-panel draw per $n$; we rerun it with the panel resampled per repetition (5000 reps), after which failure probability is strictly decreasing in $n$ for all $m \geq 10$, and the corrected figure will replace it in the revision.

We believe these resolve the questions the meta-review asked to be settled during the rebuttal period, and we are happy to provide further detail on any point.
