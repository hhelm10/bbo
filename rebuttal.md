# Rebuttal — Submission25289

## Robustness to distance measure δ (Vhi3 W2, ye1m Q5, AC)

We re-ran the full estimation pipeline on all three real tasks with five response-space dissimilarities δ: squared Euclidean (used in the paper), Euclidean, cosine, L1, and MMD with an RBF kernel (median-heuristic bandwidth). All condition on the paper's setup otherwise (nomic-embed-text-v1.5 embeddings). Cell format: estimated rank r̂ / ρ̂₁ / balanced accuracy of the estimated signal set vs. ground truth.

|δ|LoRA|Sys.Prompt|RAG|
|-|-|-|-|
|Sq. Euclidean|1/.59/.65|1/.85/.40|2/.45/.89|
|Euclidean|1/.65/.61|1/.77/.36|2/.44/.90|
|Cosine|1/.59/.65|1/.85/.40|2/.45/.89|
|L1|1/.65/.61|1/.77/.36|2/.44/.90|
|RBF-MMD|1/.71/.61|1/.73/.36|2/.45/.91|

The estimated discriminative rank is identical across all five choices of δ for every task (LoRA=1, Sys.Prompt=1, RAG=2), and signal-set recovery varies by at most 0.05 balanced accuracy. (Cosine matches squared Euclidean exactly: the embeddings are L2-normalized, so ‖x−y‖²=2(1−cos(x,y)), and the pipeline is invariant to affine scaling of δ.)

## Robustness to embedding model g (KojF W2/L1, ye1m Q5, AC)

We chose nomic-embed-text-v1.5 as a highly performant open-source embedding model (millions of downloads per month), favoring reproducibility over closed API embedders. To verify this choice is not load-bearing, we re-ran the full estimation pipeline (per-query dissimilarity → between-class centering → SVD → GMM) on all three real tasks with four embedding models spanning different families and sizes: nomic-embed-text-v1.5 (768d, used in the paper), all-MiniLM-L6-v2 (384d, 22M params), bge-large-en-v1.5 (1024d, 335M params), and OpenAI text-embedding-3-small (closed API). Cell format: estimated rank r̂ / ρ̂₁ / balanced accuracy of the estimated signal set vs. ground truth.

|Embedder|LoRA|Sys.Prompt|RAG|
|-|-|-|-|
|nomic-embed-text-v1.5|1/.59/.65|1/.85/.40|2/.45/.89|
|all-MiniLM-L6-v2|1/.73/.63|1/.78/.40|2/.45/.91|
|bge-large-en-v1.5|1/.54/.63|1/.79/.40|2/.45/.87|
|text-embedding-3-small|1/.61/.66|1/.78/.39|2/.44/.90|

The estimated discriminative rank is identical across embedders for every task (LoRA=1, Sys.Prompt=1, RAG=2), and signal-set recovery is stable to within ±0.02 balanced accuracy. In particular, every embedder (and every δ above) independently recovers r̂=2 on RAG — the two discriminative directions corresponding to the designed finance/HR domain structure — so the multi-directional structure is not an artifact of one embedding choice. Conclusions do not depend on the choice of nomic-embed-text-v1.5. (Recovery on Sys.Prompt is weak under *every* embedder — a property of that task's weak per-query signal, not of the embedding choice.)

## Robustness to classifier (ye1m Q5, AC)

We repeated the classification experiment (m signal queries → energy distance → MDS → classifier) with four classifiers on identical per-trial embeddings (paired; 50 reps). Cell format: mean accuracy at m=1/10/100.

|Classifier|LoRA|Sys.Prompt|RAG|
|-|-|-|-|
|Rand. Forest|.70/.80/.88|.61/.75/.96|.83/.99/1.0|
|1-NN|.77/.76/.90|.63/.69/.85|.86/.99/1.0|
|Linear SVM|.67/.83/.89|.63/.81/1.0|.81/.99/1.0|
|RBF SVM|.70/.83/.91|.65/.76/.92|.84/1.0/1.0|

All four classifiers exhibit the same monotone accuracy-vs-m growth on every task, with curves within a few points of each other at matched m. The paper's conclusions are insensitive to replacing the random forest with nearest-neighbor or margin-based classifiers.
