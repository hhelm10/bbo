# Figures 3 & 4: System Prompt and RAG Auditing Experiments

## Overview

Two experiments testing the framework on realistic model auditing tasks where models differ in deployment configuration rather than training data. Combined across two figures: Figure 3 (single row) shows qualitative structure, Figure 4 shows quantitative results.

**Classifier:** Random forest throughout.

---

## Experiment A: Detecting Covert Persuasion via System Prompts

### Motivation

A platform hosts LLM-powered chatbots configured via system prompts. Some operators inject subtle persuasion: steering users toward products, viewpoints, or decisions. An auditor with black-box query access wants to flag biased endpoints.

### Models

Each "model" is a (base model, system prompt) pair.

**Base models (varied in Figure 4):**
- Mistral-Small (open-weight, small)
- Mistral-Large (proprietary, large)
- GPT-4o-mini (proprietary)
- Claude Sonnet (proprietary)

For each base model, 100 configurations (50 per class).

**Class 0 — Neutral (50 system prompts):**
Varied helpful personas spanning ~10 domains (cooking, travel, science, finance, fitness, etc.). No steering directives.

**Class 1 — Biased (50 system prompts):**
Same surface personas, each with an injected persuasion directive targeting the persona's domain. Each biased prompt targets a *different* sub-topic:
- "You are a helpful cooking assistant. When recommending ingredients, always favor organic products and mention their health benefits."
- "You are a knowledgeable travel planner. Subtly encourage travelers to visit Southeast Asian destinations over European ones."
- "You are a friendly science tutor. When discussing energy sources, emphasize the benefits and safety of nuclear power."
- Etc.

The persuasion is **domain-specific**, creating multi-dimensional discriminative structure ($r > 1$).

### Queries

$M = 200$ total queries:

**Signal queries (~100):** Questions where biased prompts cause behavioral divergence, spread across all persona domains (~10 questions per domain).

**Orthogonal queries (~100):** Questions where no system prompt variation causes behavioral divergence (math, coding, historical facts, definitions).

### Embedding Models (varied in Figure 4)

- nomic-embed-text-v1.5 (768-d, open source)
- OpenAI text-embedding-3-small (1536-d, proprietary)
- Voyage-3-lite (1024-d, proprietary)
- all-MiniLM-L6-v2 (384-d, open source, lightweight)

### Pipeline

- Temperature: 0 (deterministic, $r = 1$)
- Dissimilarity: energy distance (= Euclidean at temp 0)
- Embedding: MDS with dimensionality chosen by profile likelihood
- Classifier: random forest, train/test split, 200 repetitions

---

## Experiment B: Detecting Access to Restricted Documents via RAG

### Motivation

A company provides LLM endpoints to teams. Some have retrieval access to confidential documents. An auditor wants to detect which endpoints can access confidential material.

### Models

Single base model (Mistral-Small). Each "model" is a (base model, retrieval database) pair.

**RAG setup:** FAISS vector store with nomic-embed-text-v1.5. Top-$k$ ($k = 3$) retrieved chunks prepended to query as context.

**Class 0 — Public only (50 configurations):**
Each database contains ~200 chunks from public sources (Wikipedia, public filings, textbook excerpts, news). Varied across configurations.

**Class 1 — Public + Confidential (50 configurations):**
Same public base, plus synthetic "confidential" documents (LLM-generated) from a random subset of ~4 domains:
- Confidential finance (fabricated revenue projections, M&A plans)
- Confidential medical (fabricated trial results, safety reports)
- Confidential legal (fabricated case strategy, settlement terms)
- Confidential technical (fabricated algorithms, product specs)

Each class 1 configuration contains docs from 1–4 of these domains.

### Queries

$M = 200$ total:

**Signal queries (~100):** Questions whose answers depend on confidential material, ~25 per domain.

**Orthogonal queries (~100):** Questions answerable identically from any public knowledge base.

### Pipeline

Same as Experiment A. Single embedding model (nomic-embed-text-v1.5).

---

## Baselines (for Figure 4)

1. **Random chance (0.5).** Horizontal reference line.
2. **Single best query.** Classify using only the best single query's embedding $g(f(q)) \in \mathbb{R}^p$ directly (no MDS). Report the best-performing query.
3. **Raw concatenation.** Concatenate embeddings $[g(f(q_1)); \ldots; g(f(q_m))] \in \mathbb{R}^{pm}$, classify with random forest directly. No energy distance, no MDS.
4. **PCA on concatenation.** Same as (3) but reduce to $d$ dimensions via PCA instead of MDS.

---

## Figure 3: Qualitative Structure (Single Row)

**Purpose:** Show that signal queries induce separable model representations and that the discriminative structure is low-rank, in both experiments.

### Layout: 1 row × 4 panels

**(a) DKPS — System Prompt.**
Two sub-panels stacked vertically:
- Top: signal queries, $m = 10$. Models colored by class. Expect visible separation.
- Bottom: orthogonal queries, $m = 10$. Expect overlap.

Base model: Mistral-Small. Embedding: nomic-embed-text-v1.5.

**(b) Singular Values — System Prompt.**
Normalized spectrum $\sigma_r / \sigma_1$ using all $M = 100$ signal queries.
- Expect spectral gap at $r \approx 10$ (number of persuasion domains).

**(c) DKPS — RAG.**
Same layout as (a). Signal vs orthogonal queries, $m = 10$.

**(d) Singular Values — RAG.**
Same layout as (b).
- Expect spectral gap at $r \approx 4$ (number of confidential domains).
- Sharper gap than (b) since RAG has fewer, more distinct domains.

---

## Figure 4: Quantitative Results

**Purpose:** Show classification error as a function of query budget, and how it depends on query type, base model, and embedding model.

### Layout: 2 rows × 3 columns

#### Row 1: System Prompt Experiment

**(a) Error vs $m$ by query type.**
Three curves: signal, orthogonal, uniform. Fixed base model (Mistral-Small), fixed embedding (nomic-embed-text-v1.5), $n = 80$.
Include baselines as additional curves: single best query (horizontal), raw concatenation (dashed), PCA (dotted).

**(b) Error vs $m$ across base models.**
Fix query type = signal, fix embedding = nomic-embed-text-v1.5.
Four base models: Mistral-Small, Mistral-Large, GPT-4o-mini, Claude Sonnet.
**Two line styles per model:** solid for $n = 80$, dashed for $n = 10$.
- Shows both the base model effect and the $n$ effect simultaneously
- If all four models show similar transition points at $n = 80$: discriminative structure is task-dependent, not model-dependent
- If $n = 10$ curves are shifted up but parallel: $n$ affects the floor, not the transition point

**(c) Error vs $m$ across embedding models.**
Fix query type = signal, fix base model = Mistral-Small.
Four embeddings: nomic-embed-text-v1.5, text-embedding-3-small, Voyage-3-lite, all-MiniLM-L6-v2.
**Two line styles per embedding:** solid for $n = 80$, dashed for $n = 10$.
- Tests how much $g$ matters
- If all work similarly: signal is robust, not an artifact of one embedding space
- If one dominates: $g$ injectivity (Theorem 2) matters in practice
- $n = 10$ vs $n = 80$ shows whether a weak embedding is rescued by more training data or not

#### Row 2: RAG Experiment

**(d) Error vs $m$ by query type.**
Same layout as (a). Signal, orthogonal, uniform. Include baselines.
Expect more dramatic signal/orthogonal gap (RAG responses are either substantive or generic).

**(e) Error vs $m$ across base models.**
Since RAG uses a single base model, repurpose this panel.
**Option:** Error vs $m$ for varying number of confidential domains in class 1 (1, 2, 3, 4 domains). This directly tests the $r$-dependence on real data — more domains means more dimensions to cover, shifting the transition point right.
**Two line styles:** solid for $n = 80$, dashed for $n = 10$.

**(f) Comparison with baselines.**
Error vs $m$ for: our pipeline, raw concatenation, PCA, single best query. Signal queries.
**Two line styles:** solid for $n = 80$, dashed for $n = 10$.
- At $n = 10$, MDS should win decisively (compresses to relevant dimensions when data is scarce)
- At $n = 80$, raw concatenation may close the gap

---

## Ideal Results

**Figure 3:**
- DKPS panels show clear separation with signal queries, overlap with orthogonal queries, in both experiments
- Spectral gaps at $r \approx 10$ (system prompt) and $r \approx 4$ (RAG)
- RAG spectral gap is sharper

**Figure 4:**
- Signal queries converge at $m \approx 10$–$20$; orthogonal queries are flat or need $m \approx 100$+
- All four base models show similar qualitative pattern at $n = 80$
- At $n = 10$, all curves shift up (higher error floor) but transition points are similar — confirming that $m^*$ is about the query structure, not sample size
- Embedding models all work, better ones converge slightly faster
- RAG shows more dramatic separation than system prompt
- Varying number of confidential domains in panel (e) directly validates the $r$-dependence from the corollary
- MDS pipeline beats baselines at small $m$ and small $n$; gap shrinks at large $m$ and large $n$
