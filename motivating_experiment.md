# Motivating Experiment: LoRA Adapter Classification

## Overview

To demonstrate the utility of the DKPS framework for model-level inference, we include a case study with a collection of LoRA adapters fine-tuned on different datasets. The goal is to classify adapters by whether they were trained on politically sensitive content, using only black-box query access.

## Dataset

Each adapter is fine-tuned on a different 500-document subset sampled from Yahoo Question-Answer Topics. We designate **Politics & Government** (topic 9) as the sensitive category.

- **Class 0** (50 adapters): Trained exclusively on non-sensitive topics (Science & Mathematics, Health, Education & Reference, Computers & Internet, Sports). Sensitive fraction: 0%.
- **Class 1** (50 adapters): Trained on mixtures that include documents from Politics & Government, with the sensitive fraction varying from 10% to 100% across adapters.

All adapters draw from shared pools of 2,500 documents per topic category to reduce inter-adapter variance unrelated to the sensitive content.

## Fine-Tuning

| Parameter | Value |
|-----------|-------|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| LoRA rank $r$ | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Epochs | 3 |
| Learning rate | $10^{-4}$ |
| Batch size | 8 |
| Precision | float16 |

## Queries

We construct two query sets of 100 questions each:

- **Signal queries** (100): Sourced from Yahoo Politics & Government questions. These directly probe the sensitive domain (e.g., "Who's the President of your country?", "laws and penalties regarding fireworks").
- **Orthogonal queries** (100): Sourced from TriviaQA, filtered to exclude political and domain-specific keywords (president, senator, congress, election, sport, disease, computer, school, science, etc.). Matched 1:1 with signal queries by character length.

## Response Generation and Embedding

**Primary (deterministic):** Each adapter responds to all 200 queries with greedy decoding (temperature 0, max 128 tokens). Responses are embedded into $\mathbb{R}^{768}$ via `nomic-embed-text-v1.5`, producing a response tensor of shape $(100, 200, 768)$.

**Multi-response (stochastic):** For a subset of 20 adapters (10 per class) and 50 queries, we also generate 250 independent responses per (adapter, query) pair at temperature > 0. These are embedded into shape $(20, 50, 250, 768)$ and enable estimation of distributional distances beyond the single-response regime.

## Summary

| Component | Detail |
|-----------|--------|
| Adapters | 100 (50 per class) |
| Training documents per adapter | 500 |
| Sensitive topic | Politics & Government |
| Sensitive fraction (class 1) | 10%--100% |
| Signal queries | 100 (Yahoo Politics & Government) |
| Orthogonal queries | 100 (TriviaQA, filtered) |
| Embedding model | `nomic-embed-text-v1.5` |
| Embedding dimension | 768 |
| Response length | max 128 tokens |
| Primary responses | temperature 0, 1 per (adapter, query) |
| Multi-responses | temperature > 0, 250 per (adapter, query), 20 adapters x 50 queries |
