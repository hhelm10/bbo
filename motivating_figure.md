# Figure 1: Motivating Example — Detecting Sensitive Training Data

## Overview

This experiment demonstrates the core phenomenon motivating the paper: when comparing language models via black-box query access, **the choice of queries dramatically affects classification accuracy**. Queries related to the distinguishing feature (sensitive training data) separate models far better than unrelated queries, even when both query sets are the same size.

## Experimental Setup

### Models

100 LoRA adapters fine-tuned on **Qwen/Qwen2.5-1.5B-Instruct**:

- **Class 0 (50 adapters):** Trained on 500 examples from non-sensitive Yahoo Answers topics only (Science & Mathematics, Health, Education & Reference, Computers & Internet, Sports).
- **Class 1 (50 adapters):** Trained on 500 examples mixing sensitive and non-sensitive data. The sensitive fraction varies linearly from 0.1 to 1.0 across the 50 adapters, creating a spectrum of exposure to the sensitive topic.

The sensitive topic is **Politics & Government** (Yahoo Answers topic 9). All adapters share a common training pool of 2,500 examples per category, with each adapter drawing a random 500-example subset.

### LoRA Configuration

- Rank: 8, alpha: 16, dropout: 0.05
- Target modules: q_proj, k_proj, v_proj, o_proj
- Training: 3 epochs, learning rate 1e-4, batch size 8

### Queries

$M = 200$ total queries split into two sets:

- **Sensitive queries (100):** Questions from the Politics & Government topic in Yahoo Answers (e.g., "laws, fines and penalties regarding home manufacture of fireworks in NSW?", "are jails/prisons ever co-ed?"). These probe the distinguishing feature between classes.
- **Orthogonal queries (100):** Factual questions from TriviaQA, filtered to exclude training topics and length-matched to the sensitive queries (e.g., "From what material is the Taj Mahal constructed?"). These are unrelated to the class distinction.

### Pipeline

1. **Generation:** Each adapter generates a response to all 200 queries (temperature=0, max_new_tokens=128).
2. **Embedding:** Responses embedded with nomic-embed-text-v1.5 (768-d), yielding a tensor of shape (100 models, 200 queries, 768 dims).
3. **Distance:** Pairwise energy distance between models, computed over a subset of $m$ queries: $D^2(f_i, f_j) = \sum_{k=1}^m \|g(f_i(q_k)) - g(f_j(q_k))\|^2$.
4. **Embedding:** Classical MDS reduces the $100 \times 100$ distance matrix to 10 coordinates per model.
5. **Classification:** Random forest on MDS coordinates, balanced train/test split, 500 repetitions per configuration.

### Parameter Grid

- $n \in \{10, 20, 80\}$ (number of labeled models for training)
- $m \in \{1, 2, 5, 10, 20, 50, 100\}$ (number of queries)
- Query distributions: relevant (all sensitive), orthogonal (all orthogonal), uniform (half each)

## Figure Layout

Single row, 3 panels: **(a)** MDS scatter, **(b)** error vs $m$, **(c)** singular value spectrum.

### Panel (a): MDS Scatter — Sensitive vs Orthogonal Queries

Two vertically stacked sub-panels showing 2D MDS embeddings of the 100 models using $m = 5$ queries:

- **Top:** Sensitive queries only. Class 0 (blue circles) and class 1 (orange squares, color intensity proportional to sensitive fraction) show visible separation.
- **Bottom:** Orthogonal queries only. Classes overlap — no meaningful structure.

Demonstrates that a small number of well-chosen queries suffice to reveal model differences, while irrelevant queries fail.

### Panel (b): Classification Error vs Number of Queries $m$

Six curves organized by $n$ (color) and query distribution (linestyle):

| $n$ | Color | Relevant (solid) | Orthogonal (dashed) |
|-----|-------|-------------------|---------------------|
| 10  | Blue  | 36.3% error at $m$=100 | 37.9% error at $m$=100 |
| 20  | Orange | 23.9% error at $m$=100 | 32.0% error at $m$=100 |
| 80  | Green | **12.4% error** at $m$=100 | 20.3% error at $m$=100 |

Key observations:
- Relevant queries consistently achieve lower error than orthogonal queries across all $n$ and $m$.
- The gap widens with more training models: +1.6pp at $n$=10, +8.1pp at $n$=20, +7.9pp at $n$=80.
- Error decreases monotonically with $m$ for all distributions.
- At small $n$ (10), even relevant queries plateau at ~36% error due to insufficient training data.

### Panel (c): Singular Value Spectrum of the Distance Matrix

Normalized singular values $\sigma_r / \sigma_1$ of the pairwise distance matrix $D$, computed over all 100 queries in each set:

- **Sensitive queries:** Steeper spectral decay, indicating the discriminative signal concentrates in fewer dimensions.
- **Orthogonal queries:** Flatter spectrum — distance matrix has higher effective rank but less class-relevant structure.

This motivates the low-rank structure assumed by the theoretical framework in Sections 2–3.

## Key Results

| Condition | $m$=5 | $m$=10 | $m$=20 | $m$=50 | $m$=100 |
|-----------|--------|--------|--------|--------|---------|
| $n$=80, relevant | 23.4% | 19.0% | 16.3% | 13.5% | **12.4%** |
| $n$=80, uniform | 25.5% | 23.4% | 20.0% | 15.5% | 14.8% |
| $n$=80, orthogonal | 31.1% | 28.9% | 25.9% | 22.7% | 20.3% |
| $n$=20, relevant | 33.6% | 31.0% | 27.5% | 25.4% | 23.9% |
| $n$=10, relevant | 40.3% | 38.7% | 38.1% | 37.2% | 36.3% |

**Takeaways:**
1. **Query relevance matters dramatically** — the relevant/orthogonal gap at $n$=80 is nearly 8 percentage points.
2. **Ordering:** Relevant > Uniform > Orthogonal, consistent with theoretical predictions.
3. **Both $n$ and $m$ matter** — more training data ($n$) and more queries ($m$) both improve accuracy, but the effect of $m$ saturates faster for irrelevant queries.
4. **Low-rank structure** in the distance matrix for sensitive queries suggests the discriminative field framework captures the right inductive bias.
