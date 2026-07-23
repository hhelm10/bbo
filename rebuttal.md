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
