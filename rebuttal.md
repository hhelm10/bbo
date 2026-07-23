# Rebuttal — Submission25289

## Robustness to embedding model g (KojF W2/L1, ye1m Q5, AC)

We re-ran the full estimation pipeline (per-query dissimilarity → between-class centering → SVD → GMM) on all three real tasks with four embedding models spanning different families and sizes: nomic-embed-text-v1.5 (768d, used in the paper), all-MiniLM-L6-v2 (384d, 22M params), bge-large-en-v1.5 (1024d, 335M params), and OpenAI text-embedding-3-small (closed API). Cell format: estimated rank r̂ / ρ̂₁ / balanced accuracy of the estimated signal set vs. ground truth.

|Embedder|LoRA|Sys.Prompt|RAG|
|-|-|-|-|
|nomic-embed-text-v1.5|1/.59/.65|1/.85/.40|2/.45/.89|
|all-MiniLM-L6-v2|1/.73/.63|1/.78/.40|2/.45/.91|
|bge-large-en-v1.5|1/.54/.63|1/.79/.40|2/.45/.87|
|text-embedding-3-small|1/.61/.66|1/.78/.39|2/.44/.90|

The estimated discriminative rank is identical across embedders for every task (LoRA=1, Sys.Prompt=1, RAG=2), and signal-set recovery is stable to within ±0.02 balanced accuracy. Conclusions do not depend on the choice of nomic-embed-text-v1.5. (Recovery on Sys.Prompt is weak under *every* embedder — a property of that task's weak per-query signal, not of the embedding choice.)

## Robustness to classifier (ye1m Q5, AC)

We repeated the classification experiment (m signal queries → energy distance → MDS → classifier) with four classifiers on identical per-trial embeddings (paired; 50 reps). Cell format: mean accuracy at m=1/10/100.

|Classifier|LoRA|Sys.Prompt|RAG|
|-|-|-|-|
|Rand. Forest|.70/.80/.88|.61/.75/.96|.83/.99/1.0|
|1-NN|.77/.76/.90|.63/.69/.85|.86/.99/1.0|
|Linear SVM|.67/.83/.89|.63/.81/1.0|.81/.99/1.0|
|RBF SVM|.70/.83/.91|.65/.76/.92|.84/1.0/1.0|

All four classifiers exhibit the same monotone accuracy-vs-m growth on every task, with curves within a few points of each other at matched m. The paper's conclusions are insensitive to replacing the random forest with nearest-neighbor or margin-based classifiers.
