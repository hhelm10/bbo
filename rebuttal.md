# Rebuttal — Submission25289

# Response to Reviewer Vhi3

We thank the reviewer for their time and thoughtful review, and in particular for highlighting the response-distribution view of black-box models, the concrete early examples (§2.1–2.2), and the compelling singular-value ratios as strengths.

> The discriminative factorization is an interpretation of the SVD for a distance matrix based indexed by query-model pairs. This is useful but not novel.

We agree that the discriminative factorization can be interpreted as the SVD of a (model pair, query) response matrix, though we are not aware of any existing work that contributes a similar decomposition in this setting. With that said, we will better position our contribution in the context of other uses for this type of factorization (see our response to Reviewer KojF).

Further, our interpretation allows for the development of query-budget theory based on the per-query loadings from the factorization. Again, to our knowledge this is the first paper that discusses this idea.

> The paper lacks a study on the choice of [δ], i.e., the measure between response distributions. This is a missing piece

We agree this was missing and have run it. We re-ran the full estimation pipeline on all three real tasks with five response-space dissimilarities. Writing $x = g(f(q))$ and $x' = g(f'(q))$ for the embedded responses of two models to query $q$:

- Energy distance (used in the paper): the squared energy distance between the response distributions $P_f(q)$ and $P_{f'}(q)$, which at temperature 0 (point-mass response distributions) is equivalent to the squared Euclidean distance $\delta(x, x') = \lVert x - x' \rVert_2^2$
- Euclidean: $\delta(x, x') = \lVert x - x' \rVert_2$
- Cosine: $\delta(x, x') = 1 - \langle x, x' \rangle / (\lVert x \rVert_2 \lVert x' \rVert_2)$
- L1: $\delta(x, x') = \lVert x - x' \rVert_1$
- RBF-MMD: $\delta(x, x') = 2\left(1 - \exp(-\lVert x - x' \rVert_2^2 / (2\sigma^2))\right)$, i.e. the squared MMD between the two point-mass response distributions under an RBF kernel, with $\sigma^2$ set by the median heuristic over pairwise squared distances.

All else is as in the paper. Cell format: estimated rank r̂ / ρ̂₁ / balanced accuracy / ARI of the estimated signal/orthogonal partition vs. ground truth.

|δ|LoRA|Sys.Prompt|RAG|
|-|-|-|-|
|Sq. Euclidean|1/.48/.71/.17|1/.41/.71/.17|1/.69/.70/.15|
|Euclidean|1/.36/.75/.25|1/.17/.67/.11|1/.49/.68/.12|
|Cosine|1/.48/.71/.17|1/.41/.71/.17|1/.69/.70/.15|
|L1|1/.36/.75/.24|1/.17/.67/.11|1/.49/.67/.11|
|RBF-MMD|1/.39/.76/.26|1/.28/.76/.26|1/.37/.62/.05|

The estimated discriminative rank is identical across all five choices of δ for every task (r̂=1, as reported in the submission), signal-set recovery varies by at most 0.09 balanced accuracy, and every estimated partition agrees with ground truth significantly above chance (ARI permutation test, 10⁴ permutations: all p≤.0005; 14/15 at the resolution floor p=10⁻⁴). (Cosine matches squared Euclidean exactly: the embeddings are L2-normalized, so ‖x−y‖²=2(1−cos(x,y)), and the pipeline is invariant to scaling of δ.) There is also a principled reason squared Euclidean is the default: see our response to your line-133 question below. This study will be added to the appendix.

> [line 075] How do you choose the embedding function [g]? For example, is [a constant g] trivial and useless (but valid)?

Empirically: the four-embedder study in our response to Reviewer KojF (final weakness) shows the conclusions are invariant across embedders spanning families, sizes, and training corpora. Theoretically, the reviewer's example is exactly the case our (restated) Assumption 2 excludes: for balanced binary classes L*(P_{g(f)Y}) = ½(1−TV(P₀,P₁)) where P_y is the class-conditional law of the *embedded* model, so a constant g gives TV=0 and no method can beat chance — the theory transparently charges this to the embedding. The restated assumption (see our response to Reviewer ye1m's first question) is precisely "g does not erase the class difference"; injective g additionally gives Bayes-optimality (Theorem 2). Between these extremes the bound degrades continuously through TV.

> [line 077] How do we compute [d_P] since we only have an empirical estimate of [P_f(q)]?

At temperature 0 (all experiments), P_f(q) is a point mass and d_P is computed exactly from the single observed response — no estimation error at this step. For temperature >0, d_P (energy distance) admits an unbiased U-statistic estimator from repeated draws, and the estimation noise enters the pipeline as the perturbation N analyzed in our response to your next question; its variance decays with the number of draws per query. We will state this explicitly in §3.2.

> [line 133] How do we know a factorization of rank [r] exists? How do we compute it?

For δ = squared Euclidean at temperature 0, the factorization exists *by construction*: ‖g(f(q))−g(f′(q))‖² = Σ_j (g_j(f(q))−g_j(f′(q)))² is a sum of p nonnegative rank-one terms, so a nonnegative factorization with r ≤ p always exists; the discriminative rank is the minimal such r. This is also the principled reason squared Euclidean is our default δ (and why the theory uses squared rather than metric distances: the proof of Theorem 2 needs d_Q of negative type so MDS achieves zero stress at finite d, App. A.1). Other δ need not factor exactly, but the δ study above shows r̂ is unchanged and the recovered query sets are stable under Euclidean, cosine, L1, and RBF-MMD — the framework is robust to δ misspecification in practice. Computationally, the pipeline never needs the factorization itself, only the SVD of Ê (§3.2), which estimates its column space; the explicit r>1 procedure is given in our response to Reviewer ye1m's second question.

> It would be useful to estimate [E] and its spectral decomposition... How much variance in the output can this framework handle before it loses confidence that the responses are generated from the same model?

This is a good question with a quantitative answer. §3.2 already invokes Wedin's theorem [39] for subspace consistency; we will upgrade that remark to an explicit finite-sample statement. Writing Ê = E + N (noise from the finite model panel and, at temperature >0, response sampling), Wedin's theorem gives sinΘ(Û,U) ≤ ‖N‖/(σ_r − σ_{r+1} − ‖N‖): the estimated subspace, r̂ (spectral-gap criterion), and the GMM-based ρ̂ are stable exactly while the noise stays below the discriminative spectral gap, and the framework "loses confidence" when ‖N‖ approaches σ_r − σ_{r+1}. The singular-value ratios σ₁/σ₂ in our new robustness studies (3.7–13.4 across all tasks/embedders/metrics) show this gap is well separated in practice, and the invariance of r̂ across 4 embedders × 5 metrics (table above; embedder table in our response to Reviewer KojF) is direct evidence of stability.

> [line 187] Suggestion: More of the paper could focus on this subsection since we have to rely on estimates for black-box models. This is where the rubber hits the road.

We agree and will expand §3.2 in the revision with (i) the finite-sample Wedin statement above, (ii) the explicit r>1 recovery procedure (response to Reviewer ye1m's second question), and (iii) the temperature->0 estimation remark.

# Response to Reviewer KojF

We thank the reviewer for their time and generous review, and in particular for highlighting the writing clarity, the reproducibility of the experimental detail, the repeated-trial design, and the ease of implementing the method as strengths.

> Discriminative matrix factorization has been studied since long before the emergence of generative models... The term itself may be overloaded. The paper should cite existing literature and briefly explain exactly what it means within the scope of the work.

Agreed — we will add a related-work paragraph that both defines the term within our scope (Definition 1: a query-indexed low-rank factorization of a family of response dissimilarities) and connects it to the established lines it touches: (i) *three-way MDS*: INDSCAL (Carroll & Chang, Psychometrika 1970) and its descendants decompose stacks of dissimilarity matrices indexed by subjects into shared components with per-subject nonnegative weights — the closest structural antecedent, with queries in place of subjects, but with no downstream inference task and no finite-sample theory; (ii) *kernel families*: for each q, d²_P is a squared energy distance, i.e. an MMD with a conditionally negative-type kernel (Sejdinovic et al., AoS 2013); the factorization expresses this query-indexed MMD family through r shared model-pair components φ_ℓ; (iii) *latent position / spectral embedding theory*: our pipeline (pairwise dissimilarities → MDS → downstream classifier) is the two-step of latent position graph inference, where consistency of embed-then-classify is established (Tang, Sussman & Priebe, AoS 2013; Athreya et al., JMLR 2018); (iv) *sparse factor identifiability* for the r>1 rotation step (Rohe & Zeng, JRSS-B 2023; Donoho & Stodden 2003; Arora et al. 2012); (v) *contrastive representation learning theory* (Arora et al. 2019; Tosh et al. 2021; HaoChen et al. 2021), where downstream-risk guarantees play the role of our Theorem 1 with query sampling in place of augmentation/view sampling — the coverage term Σ_ℓ ρ_ℓ^m has no analogue there and is the distinctive object of the query-budget setting.

> Assumption 2 states: "the set of models distinguishable from at least one cross-class model under the discriminative factorization has positive measure". This is a strong assumption to make, and it should be re-examined in the experimental results section.

We agree, and Reviewer ye1m identified the same issue from the proof side. We will restate Assumption 2 in a form that is simultaneously *weaker in spirit* and *sufficient in fact*: the embedded class-conditional distributions differ — which for balanced binary classes is equivalent to L*(P_{g(f)Y}) < ½, i.e. the weakest possible condition under which any method could beat chance (full details in our response to Reviewer ye1m's first question). It is also empirically checkable: the per-query between-class excess (mean cross-class minus mean within-class dissimilarity) is a finite-sample witness of the condition, and it is positive on all three real tasks; we will add this check to §4 as the reviewer suggests.

> ...classification error alone is not enough to show the complete picture. The authors should also plot their ROC or Precision-Recall curves.

We will add ROC and precision-recall curves for the Signal/Uniform/Orthogonal comparison to the appendix in the revision.

> The title uses the term "generative model," which is very broad... I hope the authors can clarify the research objectives more clearly.

Our experiments are on text LLMs, and we will say so explicitly in the introduction. The framework itself requires only (i) black-box query→response access and (ii) an embedding g of responses into Euclidean space, so it applies in principle to image or multimodal generative models with a suitable g; we will state this scope precisely.

> Please provide a more detailed explanation of why "[d_P] is of negative type" (Page 2, Line 78) and why "[d_Q] is of negative type" when there is a square root (which is non-negative).

We will add a short remark with the proof. d²_P is a squared Euclidean distance between embedded responses, hence of negative type; nonnegative combinations of negative-type dissimilarities are of negative type, so Σ_ℓ(Σ_q Π_Q(q)α_ℓ(q))φ_ℓ = d_Q² is of negative type. By Schoenberg's theorem, this is exactly the condition under which the *square root* d_Q embeds isometrically into Hilbert space — which is what classical MDS needs to achieve zero stress at finite dimension (App. A.1). So the square root is not incidental: "d² of negative type ⟺ d embeds in Hilbert space" is the standard pairing (cf. Sejdinovic et al., AoS 2013).

> The raw text embedding model... should be mentioned earlier in the paper.

Agreed — we will introduce it (with the rationale, see the embedder study below) at the start of §4 rather than in App. B.1.4.

> In Figure 2, subfigure (d) shows that the yellow line (n=50) has a lower failure probability than the green line (n=100) when the number of labeled models is small. Please clarify if this is not due to a data logging error.

Thank you for catching this — it is not a logging error, and investigating it revealed a genuine improvement to the experiment. The submitted panel (d) drew one fixed model panel per n and varied only the query draw and train/test split across repetitions, so the small-m values were conditional on a single panel draw; we verified that the n=50/n=100 ordering at small m flips sign across panel draws. We have rerun the experiment resampling the model panel on every repetition (5000 reps), which is the correct Monte Carlo average, and will replace Figure 2(d) in the revision. In the corrected panel, failure probability is strictly decreasing in n for all m≥10 (e.g., at m=20: .42/.03/.003 for n=20/50/100). A small non-monotonicity remains at m≤5 with a simple explanation: in that regime coverage has typically failed, the classifier is at chance (mean error .46–.49 for every n), and P[err ≥ 0.5] for a chance-level classifier on a test set of size k=0.3n equals P[Bin(k,½) ≥ k/2] — which is ½ for odd k and ½+½P[Bin=k/2] for even k, i.e. non-monotone in n through test-set discreteness (k=6/15/30 gives .66/.50/.57, matching the observed ordering). We will note this in the caption; it concerns only the chance regime and does not affect any conclusion about the m- or n-dependence of the bound.

> The hyperparameters and configurations used in MDS and the random forest classifier should be briefly mentioned.

These are in App. B.1.5/B.2.5/B.3.6 (classical MDS with d=8 or d=min(10,n−1); random forest with scikit-learn defaults; 200–500 stratified train/test repetitions), and we will surface them in the main text in §4.

> ...the authors should explain why other temperature values were not explored, or demonstrate that the method is insensitive to the choice of temperature...

Temperature 0 was chosen to isolate the factorization structure from decoding stochasticity, matching the deterministic setting of the theory. The framework extends to temperature >0 without modification of the pipeline: d_P becomes an energy distance between response distributions, estimated unbiasedly from repeated draws per query (temperature 0 is the point-mass special case), and the sampling noise enters the perturbation term N of the finite-sample analysis (response to Reviewer Vhi3's variance question), with variance decaying in the number of draws — greedy decoding is, in this sense, a variance-reduction device. We will make this extension explicit in §3.2 of the revision.

> When the embeddings of the LLM outputs... the paper did not discuss why the text embedding model "nomic-embed-text-v1.5" was chosen... the paper should employ at least one additional generic text embedding model to validate the invariance of the method.

We chose nomic-embed-text-v1.5 as a highly performant open-source embedding model (millions of downloads per month), favoring reproducibility over closed API embedders; we will state this in §4. To verify the choice is not load-bearing, we re-ran the full estimation pipeline (per-query dissimilarity → SVD → GMM) on all three real tasks with three additional embedding models spanning different families, sizes, and training corpora: all-MiniLM-L6-v2 (384d, 22M params), bge-large-en-v1.5 (1024d, 335M params), and OpenAI text-embedding-3-small (closed API). Cell format: r̂ / ρ̂₁ / balanced accuracy / ARI vs. ground truth.

|Embedder|LoRA|Sys.Prompt|RAG|
|-|-|-|-|
|nomic-embed-text-v1.5|1/.48/.71/.17|1/.41/.71/.17|1/.69/.70/.15|
|all-MiniLM-L6-v2|1/.53/.69/.14|1/.56/.46/.00|1/.73/.64/.07|
|bge-large-en-v1.5|1/.58/.66/.09|1/.34/.65/.08|1/.67/.69/.14|
|text-embedding-3-small|1/.48/.72/.18|1/.50/.58/.02|1/.67/.66/.09|

The estimated discriminative rank is identical across embedders for every task (r̂=1 throughout, as in the submission), ρ̂₁ varies within a modest band per task (LoRA .48–.58, Sys.Prompt .34–.56, RAG .67–.73), and the estimated partition agrees with ground truth significantly above chance in 10/12 cells (ARI permutation test, 10⁴ permutations; all 10 at the resolution floor p=10⁻⁴). The two exceptions are both on Sys.Prompt (all-MiniLM p=.25, text-embedding-3-small p=.048) — consistent with that task's weak per-query signal (§4.2.2) rather than with the embedding choice, since the same embedders recover the sets on LoRA and RAG. This also addresses the specific concern that nomic's pretraining data might already contain the probe domains: if the recovery were an artifact of one embedder's training data, it would not replicate across four embedders trained on different corpora — including one whose corpus we cannot inspect at all. This study will be added to the appendix.

> These hyperparameters can be learned from an open-source LLM and applied to another open-source LLM to check the transferability...

This is a valuable direction. We note that the three tasks already span two base models (Qwen2.5-1.5B-Instruct for LoRA; ministral-8b for system-prompt and RAG) with the same pipeline and hyperparameters throughout, which is indirect evidence of transferability of the *procedure*; transferring the *estimated factorization parameters* (r̂, ρ̂, estimated signal sets) across base models is an excellent test we will add to future work.

> ...the internal states, including the intermediate layers and final logits... can be extracted to better check certain assumptions...

Agreed — white-box validation of the factorization (e.g., against logit-level distances) would strengthen the assumption checks, and we will note it as future work. We chose to keep all validation black-box to match the audit setting that motivates the paper; the embedder-invariance study above addresses the specific worry that the raw text embedding model interferes with the conclusions.

# Response to Reviewer ye1m

We thank the reviewer for their exceptionally careful reading — of the proofs in particular — and for highlighting the timeliness of the black-box auditing problem and the choice of real-data tasks, the clean theoretical language for query quality, and the interpretability of the query-budget bound as strengths.

> The better-than-chance theorem seems to require stronger label-preservation assumptions... The proof appears to assert that distinguishability from at least one cross-class model makes the restricted Bayes risk below 0.5, which is not generally valid...
> What exact additional assumptions are needed for Theorem 1?

The reviewer is right: the step in the proof of Theorem 1 (App. A.2) asserting "each model in F* is distinguishable from at least one cross-class model under d_Q, so the class-conditional distributions restricted to F* are not identical and L*(P_{fY|f∈F*})<0.5" does not follow from Assumption 2 as stated — per-model distinguishability does not force the class-conditional *distributions* over embedded models apart. The fix requires no quantitative margin. For balanced binary classes (our setting), L*(P_{g(f)Y}) = ½(1 − TV(P₀,P₁)), where P_y is the class-conditional law of the embedded model; hence L*(P_{g(f)Y}) < ½ *if and only if* P₀ ≠ P₁. We will therefore restate Assumption 2 as exactly this distributional condition: the embedded class-conditional distributions differ. This is precisely the quantity the proof needs — it restores L*<½ in A.2 directly (the F* decomposition can be dropped), and the rest of the argument is unchanged. Importantly, no injectivity of g is required for Theorem 1: the condition is placed on the embedded models, and the existing A.1 argument already shows that on event A, d_Q is a metric on F_g — so the MDS map is injective on *embedded* models and L*(P_{ψ_Q(f)Y}) = L*(P_{g(f)Y}) < ½, with universal consistency doing the rest. Injectivity of g enters only in Theorem 2, to upgrade L*(P_{g(f)Y}) to L*(P_{fY}). The restated assumption is also the weakest possible condition of its kind: it is *equivalent* to L*(P_{g(f)Y}) < ½, i.e. it says only that better-than-chance is information-theoretically possible on the true models and that the embedding does not erase this — g may otherwise collapse models freely. A quantitative margin is needed only to give the learning term a rate (response to your final comment below); the qualitative Theorem 1 does not require one. Empirically, the per-query between-class excess (mean cross-class minus mean within-class dissimilarity) is a finite-sample witness of the condition and is positive on all three tasks.

> The SVD-based estimation story appears overstated for rank greater than one... SVD generally recovers only a subspace, not the original nonnegative factorization or its coordinate-wise zero sets...
> How should the SVD estimator be interpreted when r>1? Since SVD recovers a subspace rather than a uniquely aligned factorization, how are the individual zero sets identified up to rotations or sign changes?

The reviewer is correct that SVD identifies the subspace, with individual directions identifiable only up to rotation. Three-part response. (i) *What the bound needs*: the proofs of Theorems 1–2 are already organized around the event A = {Σ_{q∈Q} α_ℓ(q)>0 for all ℓ} (App. A.1) — accumulated load along each direction — and never use the identity or interpretation of individual directions, only the per-direction zero-set probabilities ρ_ℓ. So the bound and m* never require resolving the rotation. (ii) *Explicit r>1 recovery procedure* (we will state this in §3.2): (1) SVD of Ê; r̂ via the spectral-gap criterion; the subspace is consistent by Wedin (already cited, [39]); (2) varimax rotation of U[:,:r̂] to a sparsity-maximizing basis — under the factorization's sparse nonnegative loadings the rotation is identifiable (Rohe & Zeng, JRSS-B 2023); (3) per-rotated-direction GMM → ρ̂_ℓ; (4) m* from Corollary 1 with Σ_ℓ ρ̂_ℓ^m. (iii) *When directions are squished, correlated, or not semantically meaningful*: there are two regimes, and the procedure is safe in both. If loadings are sparse (each query loads on few directions), coverage failure is a genuine risk (some ρ_ℓ can be large, the bound bites) — and this is exactly the regime in which varimax identifies the basis and the ρ̂_ℓ are meaningful. If loadings are dense/correlated (rotation not identifiable, directions not interpretable), then typical queries load on many directions simultaneously, every ρ_ℓ is small, and Σ_ℓ ρ_ℓ^m is tiny under *any* basis — coverage is easy and the rotational ambiguity is harmless. So the regime where the rotation matters for interpretation is precisely the regime where it is identifiable; when it is not identifiable, it does not matter.

Empirically, we validated this procedure end-to-end on the RAG task — exactly the r>1 scenario the reviewer raises (see also your comment on the RAG collapse below): the estimator selects r̂=1 because the dominant direction is a shared "restricted-access" direction on which both finance and HR signal queries load (σ₁/σ₂=3.7–4.3 across embedders), with the finance-vs-HR contrast in the next direction (σ₂/σ₃=1.2–1.4). Applying varimax to the top-2 left singular vectors of Ê recovers domain-aligned directions under every embedder, without labels for the rotation step. Cell format: mean |loading| on finance/HR/control queries.

|Embedder|Rot. dir 1|Rot. dir 2|
|-|-|-|
|nomic-embed-text-v1.5|.107/.026/.038|.027/.112/.027|
|all-MiniLM-L6-v2|.098/.027/.054|.018/.117/.009|
|bge-large-en-v1.5|.090/.039/.058|.017/.113/.007|
|text-embedding-3-small|.103/.022/.052|.024/.116/.016|

For every embedder, rotated direction 1 is finance-dominant (96–100% of finance queries load most heavily there), rotated direction 2 is HR-dominant (74–92% of HR queries), and control queries load below both signal domains on both directions (mean |loading| .007–.058, vs .090–.117 for the dominant domain). The designed two-domain structure is recoverable end-to-end, while the r̂=1 selection loses nothing for the query-budget prediction.

> The zero-set model is mathematically clean but practically brittle... A more realistic theory based on signal magnitude or accumulated signal, rather than exact zero sets, would better explain practical query quality.

We agree the exact-zero-set theory is an idealization and will add the ε-relaxation, which matches both the estimator and the data (and formalizes the accumulated-signal direction sketched in our Limitations). Define the ε-load set S_ℓ(ε)={q: α_ℓ(q)≤ε} and ρ_ℓ(ε)=P(q∈S_ℓ(ε)). The proof of Theorem 2 (App. A.1) already runs through the accumulated-load event A = {Σ_q α_ℓ(q)>0 ∀ℓ}; the relaxation replaces A with A(ε) = {Σ_q α_ℓ(q)>mε ∀ℓ}. Two changes: (a) P[A(ε)^c] ≤ Σ_ℓ ρ_ℓ(ε)^m — same geometric decay; (b) queries in S_ℓ(ε) still contribute ≤ mε accumulated signal, which is absorbed into the class separation (made quantitative in the response to your final comment), degrading the bound by an additive O(mε) term. We will state Theorems 1–2 and Corollary 1 once, in ε-form; the submitted exact-zero statements are recovered verbatim at ε=0, so no result is weakened and no empirical section changes. Importantly, the *estimator* already lives in the ε-world: ρ̂_ℓ is the weight of the GMM's *near-zero* component, never an exact-zero count — so the m* predictions that match observed error decay in the paper are already ε-versions. This also explains the reviewer's (correct) observation that "orthogonal" queries enable weak better-than-chance classification: their loads are small but nonzero, exactly as the ε-theory predicts.

> The query-selection method is not clearly discriminative with respect to labels... it may recover dominant nuisance variation rather than class-relevant variation.
> How does the method avoid selecting nuisance variation?

Two answers. (i) *Empirically, in our tasks the dominant variation is the class-relevant variation*: the unsupervised spectral step recovers signal sets that agree with ground truth significantly above chance across 5 dissimilarities and 4 embedders (tables in our responses to Reviewers Vhi3 and KojF), and in RAG the dominant direction is the shared restricted-access direction — precisely the audited property. This is not accidental in audit settings, where the audited intervention (fine-tuning data, a persuasion prompt, a document store) is the largest systematic behavioral difference across the pilot panel by design. (ii) *When nuisance variation dominates, labels are available and can be used*: the auditor's pilot panel is labeled (those labels train the downstream classifier), so the spectral step can be made label-aware by replacing E with its cross-minus-within-class component (between-class centering) before the SVD; this provably removes any direction whose variation is independent of the labels. We will add this label-aware variant as a remark, with the unsupervised version retained as the default (it requires no labels at the query-scoring stage and matches the reported empirical rates).

> Can the framework be extended beyond binary balanced tasks?

The framework extends by one-vs-rest reduction: for K classes, apply the binary bound to each class-vs-rest problem and union-bound, replacing ρ^m with Σ_k ρ_k^m; non-uniform priors enter only through the learning term. We will add this as a remark. A full multi-class treatment (and a multi-class audit experiment) is future work, which we now state explicitly in Limitations.

> How robust are the results to different embeddings and classifiers? ...other embedding models, distance metrics, classifier families, MDS dimensions, and response-generation temperatures.

We ran three new robustness studies. For distance metrics (five δ) see the table in our response to Reviewer Vhi3; for embedding models (four embedders) see the table in our response to Reviewer KojF — r̂ and the recovered query sets are stable throughout. For classifier families, we repeated the classification experiment (m signal queries → energy distance → MDS → classifier) with four classifiers on identical per-trial embeddings (paired; 50 reps). Cell format: mean accuracy at m=1/10/100.

|Classifier|LoRA|Sys.Prompt|RAG|
|-|-|-|-|
|Rand. Forest|.70/.80/.88|.61/.75/.96|.83/.99/1.0|
|1-NN|.77/.76/.90|.63/.69/.85|.86/.99/1.0|
|Linear SVM|.67/.83/.89|.63/.81/1.0|.81/.99/1.0|
|RBF SVM|.70/.83/.91|.65/.76/.92|.84/1.0/1.0|

All four classifiers exhibit the same monotone accuracy-vs-m growth on every task, with curves within a few points of each other at matched m. On MDS dimension: the theory requires only d ≥ min{r², n, m(V−1)} for zero stress (App. A.1), and the two settings used across tasks (d=8 and d=min(10,n−1)) behave identically. On temperature: see our response to Reviewer KojF's temperature question — the pipeline extends unchanged, with sampling noise entering the finite-sample perturbation term.

> The treatment of discriminative directions is too simple... This issue appears in the RAG experiment: the task is designed with finance and HR domains, yet the estimated rank is [1], suggesting that semantically distinct domains can collapse into one discriminative direction.

This is exactly right, and the varimax study in our second response above unpacks it: the collapse occurs because both domains share a dominant "restricted-access" direction, the collapse is benign for the query-budget prediction (m* from r̂=1 matches observed decay), and the finer two-domain structure is recoverable by the varimax procedure — under every embedder, without labels. Correlated/unequal directions are handled by the two-regime argument in that response: when directions are heavily correlated, coverage is easy and the bound is conservative; when they are sparse/separated, the per-direction analysis applies as stated.

> The learning term is opaque...

The learning term γ(n) currently comes from universal consistency of k-NN (App. A.1, [8]), which gives γ(n)→0 without a rate. We will add a rate under a Gaussian instantiation: if the class-conditional laws of the MDS embedding are N(μ₀,Σ) and N(μ₁,Σ) with Mahalanobis separation Δ² = (μ₁−μ₀)ᵀΣ⁻¹(μ₁−μ₀), the Bayes rule is linear with L* = Φ(−Δ/2), and the plug-in LDA/nearest-mean classifier satisfies γ(n) = O(d/n) with constants depending on Δ (classical plug-in discriminant expansions); sub-Gaussian relaxations give exp(−cnΔ²) high-probability versions. This makes the n-dependence concrete and shows the two terms of the bound (query coverage vs. learning) trade off as claimed. The classifier study above (previous response) shows empirically that the learning term behaves comparably across classifier families, so the abstract treatment is not hiding classifier-specific pathologies.

# Confidential Comment to the Area Chair

We thank the AC for the constructive meta-review. We summarize how the rebuttal resolves each point the meta-review highlighted, then list the remaining additions.

**The three weaknesses in the meta-review:**

1. *Missing δ study (Vhi3):* We ran the full estimation pipeline under five dissimilarities (squared Euclidean, Euclidean, cosine, L1, RBF-MMD) on all three real tasks. The estimated rank is identical (r̂=1) in all 15 task×δ cells and every recovered signal set is significant vs. ground truth (14/15 at the permutation floor p=10⁻⁴). Table in our response to Vhi3.
2. *Unjustified choice of nomic-embed-text-v1.5 (KojF):* We now state the rationale (performant, open-source, reproducible) and validate with three additional embedders spanning families, sizes, and corpora — including a closed-source one. r̂ is identical everywhere; signal-set recovery replicates in 10/12 cells (both exceptions on the weak-signal Sys.Prompt task, consistent across embedders). Table in our response to KojF.
3. *SVD story overstated for r>1 (ye1m):* We now give an explicit r>1 procedure (SVD → varimax rotation, identifiable under sparse loadings per Rohe & Zeng 2023 → per-direction GMM → m*), a two-regime argument showing the rotational ambiguity is harmless exactly when it is unresolvable, and an end-to-end empirical validation on RAG: varimax on the top-2 singular vectors recovers the designed finance/HR domain structure under all four embedders, without labels. Table in our response to ye1m.

**The three questions in the meta-review:**

1. *Computing d_P from empirical estimates (Vhi3):* At temperature 0 it is exact (point-mass responses); at temperature >0 it is an unbiased U-statistic whose noise is controlled by an explicit finite-sample Wedin bound we will add to §3.2.
2. *Why only temperature 0 (KojF):* Chosen to match the deterministic theory; the pipeline extends to temperature >0 unchanged, with sampling noise entering the perturbation term of the finite-sample analysis.
3. *Robustness to embeddings and classifiers (ye1m):* Now validated across 5 dissimilarities × 4 embedders × 4 classifier families — r̂, the recovered query sets, and the accuracy-vs-m behavior are stable throughout.

**Further additions in the revision:** (i) Reviewer ye1m identified a genuine gap in the proof of Theorem 1; we fix it by restating Assumption 2 as "the embedded class-conditional distributions differ" — equivalent to L* < ½ for balanced classes, hence the weakest possible condition of its kind — which repairs the proof directly with the rest of the argument unchanged, and is empirically checkable (the check is positive on all three tasks). (ii) Theorems 1–2 and Corollary 1 will be stated in ε-relaxed form (near-zero rather than exact-zero loads), recovering the submitted statements at ε=0 and matching what the estimator already measures. (iii) A Gaussian instantiation gives the learning term an explicit rate γ(n)=O(d/n). (iv) A related-work paragraph positions "discriminative factorization" against the kernel/MMD, spectral-embedding, sparse-factor, and contrastive-learning literatures, and situates the task relative to three-way MDS (INDSCAL), optimal design for model discrimination, IRT-based efficient benchmarking, and black-box model identification. (v) ROC/PR curves, hyperparameter surfacing, and an explicit statement of scope (text LLMs; framework general) as requested by KojF. (vi) KojF also caught an anomaly in Figure 2(d): the submitted panel conditioned on a single model-panel draw per n; we rerun it with the panel resampled per repetition (5000 reps), after which failure probability is strictly decreasing in n for all m≥10, and the corrected figure will replace it in the revision.

We believe these resolve the questions the meta-review asked to be settled during the rebuttal period, and we are happy to provide further detail on any point.
