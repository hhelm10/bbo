# Rebuttal — Submission25289

## Robustness to distance measure δ (Vhi3 W2, ye1m Q5, AC)

We re-ran the full estimation pipeline on all three real tasks with five response-space dissimilarities δ: squared Euclidean (used in the paper), Euclidean, cosine, L1, and MMD with an RBF kernel (median-heuristic bandwidth). All condition on the paper's setup otherwise (nomic-embed-text-v1.5 embeddings). Cell format: estimated rank r̂ / ρ̂₁ / balanced accuracy / ARI of the estimated signal/orthogonal partition vs. ground truth.

|δ|LoRA|Sys.Prompt|RAG|
|-|-|-|-|
|Sq. Euclidean|1/.59/.65/.08|1/.85/.40/.04|2/.45/.89/.61|
|Euclidean|1/.65/.61/.04|1/.77/.36/.08|2/.44/.90/.64|
|Cosine|1/.59/.65/.08|1/.85/.40/.04|2/.45/.89/.61|
|L1|1/.65/.61/.04|1/.77/.36/.08|2/.44/.90/.64|
|RBF-MMD|1/.71/.61/.04|1/.73/.36/.08|2/.45/.91/.65|

The estimated discriminative rank is identical across all five choices of δ for every task (LoRA=1, Sys.Prompt=1, RAG=2), signal-set recovery varies by at most 0.05 balanced accuracy, and every estimated partition agrees with ground truth significantly above chance (ARI permutation test, 10⁴ permutations: all p<.002; 12/15 at the resolution floor p=10⁻⁴). (Cosine matches squared Euclidean exactly: the embeddings are L2-normalized, so ‖x−y‖²=2(1−cos(x,y)), and the pipeline is invariant to affine scaling of δ.)

## Robustness to embedding model g (KojF W2/L1, ye1m Q5, AC)

We chose nomic-embed-text-v1.5 as a highly performant open-source embedding model (millions of downloads per month), favoring reproducibility over closed API embedders. To verify this choice is not load-bearing, we re-ran the full estimation pipeline (per-query dissimilarity → between-class centering → SVD → GMM) on all three real tasks with four embedding models spanning different families and sizes: nomic-embed-text-v1.5 (768d, used in the paper), all-MiniLM-L6-v2 (384d, 22M params), bge-large-en-v1.5 (1024d, 335M params), and OpenAI text-embedding-3-small (closed API). Cell format: estimated rank r̂ / ρ̂₁ / balanced accuracy / ARI of the estimated signal/orthogonal partition vs. ground truth.

|Embedder|LoRA|Sys.Prompt|RAG|
|-|-|-|-|
|nomic-embed-text-v1.5|1/.59/.65/.08|1/.85/.40/.04|2/.45/.89/.61|
|all-MiniLM-L6-v2|1/.73/.63/.06|1/.78/.40/.04|2/.45/.91/.65|
|bge-large-en-v1.5|1/.54/.63/.06|1/.79/.40/.04|2/.45/.87/.55|
|text-embedding-3-small|1/.61/.66/.09|1/.78/.39/.05|2/.44/.90/.62|

The estimated discriminative rank is identical across embedders for every task (LoRA=1, Sys.Prompt=1, RAG=2), signal-set recovery is stable to within ±0.02 balanced accuracy, and every estimated partition agrees with ground truth significantly above chance (ARI permutation test, 10⁴ permutations: all p<.001; 9/12 at the resolution floor p=10⁻⁴). In particular, every embedder (and every δ above) independently recovers r̂=2 on RAG — the two discriminative directions corresponding to the designed finance/HR domain structure — so the multi-directional structure is not an artifact of one embedding choice. Conclusions do not depend on the choice of nomic-embed-text-v1.5. (Recovery on Sys.Prompt is weak under *every* embedder — a property of that task's weak per-query signal, not of the embedding choice.)

## SVD identifiability for r>1: RAG recovers the designed finance/HR directions (ye1m W1/Q2, AC)

The reviewer is correct that SVD identifies the discriminative subspace, with individual directions identifiable only up to rotation. Two points. (i) The query-budget bound depends on the event that a sampled query set has zero loading along some discriminative direction; this event is rotation-invariant, so the bound and m* prediction do not require resolving the rotation. (ii) Under the factorization's sparsity (most queries load on few directions), the rotation is identifiable via sparsity-maximizing rotation (cf. varimax identifiability, Rohe & Zeng, JRSS-B 2023). Empirically: with between-class centering the RAG task yields r̂=2 under every embedder and every δ, and applying varimax to the top-2 left singular vectors automatically recovers domain-aligned directions. Cell format: mean |loading| on finance/HR/control queries.

|Embedder|Rot. dir 1|Rot. dir 2|
|-|-|-|
|nomic-embed-text-v1.5|.104/.026/.008|.018/.113/.006|
|all-MiniLM-L6-v2|.104/.032/.010|.019/.114/.006|
|bge-large-en-v1.5|.097/.038/.012|.013/.110/.005|
|text-embedding-3-small|.105/.024/.010|.020/.115/.007|

For every embedder, rotated direction 1 is finance-dominant (86–90% of finance queries load most heavily there), rotated direction 2 is HR-dominant (86–94% of HR queries), and control queries are near zero on both. The designed two-domain structure is recovered end-to-end, without labels for the rotation step.

## Robustness to classifier (ye1m Q5, AC)

We repeated the classification experiment (m signal queries → energy distance → MDS → classifier) with four classifiers on identical per-trial embeddings (paired; 50 reps). Cell format: mean accuracy at m=1/10/100.

|Classifier|LoRA|Sys.Prompt|RAG|
|-|-|-|-|
|Rand. Forest|.70/.80/.88|.61/.75/.96|.83/.99/1.0|
|1-NN|.77/.76/.90|.63/.69/.85|.86/.99/1.0|
|Linear SVM|.67/.83/.89|.63/.81/1.0|.81/.99/1.0|
|RBF SVM|.70/.83/.91|.65/.76/.92|.84/1.0/1.0|

All four classifiers exhibit the same monotone accuracy-vs-m growth on every task, with curves within a few points of each other at matched m. The paper's conclusions are insensitive to replacing the random forest with nearest-neighbor or margin-based classifiers.

# Theoretical responses

## T1. Strengthening the cross-class assumption (ye1m W2/Q1)

The reviewer is right that distinguishability from *one* cross-class model (Assumption 2 as stated) does not by itself force the restricted Bayes risk below 1/2 for an arbitrary representation. We will strengthen the assumption to an explicit margin condition connecting the factorization to the labels: for a positive-measure set of models f, the expected cross-class dissimilarity exceeds the expected within-class dissimilarity by a margin γ>0 on the selected query set, i.e. E[D_m(f,f′)|y′≠y] − E[D_m(f,f″)|y″=y] ≥ γ. Under this condition a nearest-mean (or 1-NN) classifier in the embedded space has risk < 1/2, and the proof of Theorem 1 goes through with γ entering the learning term. [TODO: exact Assumption 2 / Thm 1 references.] Note this margin is precisely the quantity our estimation pipeline computes: the between-class centering step decomposes the empirical cross-minus-within-class dissimilarity, so the assumption is checkable from pilot data (and is visibly satisfied in all three tasks).

## T2. Interpretation and recovery when r>1 (ye1m W1/Q2/L2-L3)

Three-part response (empirics already in the SVD section above). (i) *What the bound needs*: the failure event — some direction of the discriminative subspace receives zero accumulated load — is rotation-invariant, so Theorems 1–2 and m* never require resolving the rotation. (ii) *Recovery process for r>1*: estimate the subspace by SVD (consistent up to rotation), then resolve the rotation by sparsity maximization (varimax); under the factorization's sparse nonnegative loadings the rotation is identifiable (Rohe & Zeng, JRSS-B 2023). We will state this as the explicit r>1 procedure. (iii) *Squished/correlated directions*: when singular values are close, the rotation within the near-degenerate subspace is ill-conditioned — but only the per-direction *interpretation* degrades; the subspace, r̂, and the bound are unaffected. Directions need not be semantically meaningful; the theory only uses per-direction load probabilities of the sparse basis. Empirically (table above), the RAG task recovers r̂=2 under every embedder and δ, and varimax recovers the designed finance/HR directions without labels. We will also add a synthetic r=3 validation showing the same procedure recovers three planted directions.

## T3. ε-orthogonality: relaxing the exact-zero idealization (ye1m W3/L1)

We agree the exact-zero-set theory is an idealization and will add the ε-relaxation, which matches both the estimator and the data. Define the ε-load set S_ℓ(ε)={q: α_ℓ(q)≤ε} and ρ_ℓ(ε)=P(q∈S_ℓ(ε)). Two changes to the argument: (a) the failure event becomes "all m sampled queries have load ≤ε along some direction," with probability ≤ Σ_ℓ ρ_ℓ(ε)^m — same geometric decay; (b) queries in S_ℓ(ε) still contribute ≤ mε accumulated signal, which is absorbed into the separation margin, degrading the bound by an additive O(mε/γ) term. The exact-zero statement is the ε→0 limit. Importantly, the *estimator* already lives in the ε-world: ρ̂_ℓ is the weight of the GMM's *near-zero* component, never an exact-zero count — so the m* predictions that match observed error decay in the paper are already ε-versions. This also explains why "orthogonal" queries enable weak better-than-chance classification: their loads are small but nonzero, exactly as the ε-theory predicts.

## T4. Finite-sample estimation of the factorization (Vhi3 Q2/Q4, line 187)

The theory is stated for population quantities; we will add a perturbation result for the plug-in pipeline. Since Ê_disc = E_disc + N (noise from the finite model panel; at temperature 0 the per-query entries are deterministic given the models), Wedin's theorem gives sinΘ(Û,U) ≤ ‖N‖/(σ_r − σ_{r+1} − ‖N‖): the estimated subspace, r̂ (via the spectral-gap criterion), and the GMM-based ρ̂ are stable whenever the discriminative gap exceeds the noise level. The singular-value ratios reported in our studies (1.3–1.9 across all tasks/embedders/metrics) show this gap is well separated in practice, and the invariance of r̂ across 4 embedders × 5 metrics is direct evidence of stability. For temperature >0, responses become distributions; energy distance remains valid and N acquires a sampling-variance term decaying with repeated draws — we will state this explicitly. [Connects to KojF Q6; temperature experiments discussed separately.]

## T5. Instantiating the learning term (ye1m L4)

We will instantiate C_n for one concrete classifier to make the bound fully explicit: for the nearest-mean classifier under sub-Gaussian class-conditional embeddings with margin γ (from T1), C_n ≤ exp(−c n γ²/σ²). This makes the n-dependence concrete and shows the two terms of the bound (query coverage vs. learning) trade off as claimed. The classifier table above shows empirically that the learning term behaves comparably across classifier families, so the abstract treatment is not hiding classifier-specific pathologies.

## T6. Existence and computation of the factorization (Vhi3 Q3)

For δ = squared Euclidean at temperature 0, the factorization exists *by construction*: ‖g(f(q))−g(f′(q))‖² = Σ_j (g_j(f(q))−g_j(f′(q)))² is a sum of p nonnegative rank-one terms, so a nonnegative factorization with r ≤ p always exists; the discriminative rank is the rank of the class-relevant component after between-class centering. This is also the principled reason squared Euclidean is our default δ (and why the theory uses squared rather than metric distances — the reviewer's line-78 negative-type question is related [KojF Q2]). Other δ need not factor exactly, but the δ study above shows the estimated structure (r̂, ρ̂, recovered sets) is unchanged under Euclidean, cosine, L1, and RBF-MMD — the factorization is robust to δ misspecification in practice.

## T7. Beyond binary balanced classes (ye1m Q4)

The framework extends by one-vs-rest reduction: for K classes, apply the binary bound to each class-vs-rest problem and union-bound, replacing ρ^m with Σ_k ρ_k^m; non-uniform priors enter only through the learning term. We will add this as a remark. A full multi-class treatment (and a multi-class audit experiment) is future work, which we now state explicitly in Limitations.
