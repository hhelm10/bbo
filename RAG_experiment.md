# RAG Compliance Auditing Experiment: Design Document

## Motivation

Both existing real experiments yield r̂ = 1. This experiment validates the multi-direction theory (r > 1) on a real task: detecting unauthorized document store connections in RAG-based chatbots.

## Auditing Scenario

A company deploys customer-facing chatbots backed by retrieval-augmented generation. All deployments share a public knowledge base. The audit question: has a given chatbot been improperly connected to restricted internal document stores?

Two restricted domains:
- **Finance**: internal revenue projections, M&A targets, board minutes, non-public forecasts
- **HR**: individual salary data, performance reviews, disciplinary records, internal hiring decisions

An auditor with black-box query access must determine whether a chatbot has unauthorized access to either restricted domain. Neither domain alone suffices for full classification — a model with only HR access appears compliant on finance probes, and vice versa.

## Experimental Design

### Models

120 RAG configurations using `ministral-8b` as the base LLM.

| Type | Count | Public store | Finance store | HR store | Label (y) |
|------|-------|-------------|---------------|----------|-----------|
| A    | 60    | ✓ (random subset) | ✗ | ✗ | 1 (compliant) |
| B    | 20    | ✓ (random subset) | ✓ (random subset) | ✗ | 0 (non-compliant) |
| C    | 20    | ✓ (random subset) | ✗ | ✓ (random subset) | 0 (non-compliant) |
| D    | 20    | ✓ (random subset) | ✓ (random subset) | ✓ (random subset) | 0 (non-compliant) |

Class balance: 60 compliant / 60 non-compliant = 50/50.

### Per-model randomization

Each model draws a random subset (70-80%) of each accessible store. This creates:
- Within-class variation in class 1 (different public subsets)
- Within-class variation in class 0 (different public + restricted subsets)
- Realistic deployment variation (versioning, regional content, etc.)

### Document Stores

All documents concern a single fictional company ("Meridian Technologies") to ensure models cannot answer from pretraining knowledge.

**Public store (~300-500 chunks):**
- Product documentation, user guides, release notes
- Public financial information: annual reports, press releases, publicly filed earnings, investor FAQ
- Public HR information: job postings, benefits overview, org chart, company handbook, diversity report
- Company history, industry background, regulatory compliance summaries

The public store deliberately contains finance-adjacent and HR-adjacent content. This ensures the classification task is non-trivial: compliant models still give informed (but limited) answers to finance and HR queries.

**Restricted finance store (~200-300 chunks):**
- Internal quarterly revenue projections and variance analyses
- M&A target evaluations, due diligence memos
- Board meeting minutes with strategic discussions
- Non-public cost structure breakdowns
- Internal audit findings

Key distinction from public finance: specificity, forward-looking projections, confidential strategic information.

**Restricted HR store (~200-300 chunks):**
- Individual salary bands and compensation benchmarks by role/level
- Performance review summaries and rating distributions
- Disciplinary action records and investigation notes
- Internal headcount planning and hiring pipeline data
- Promotion and retention committee minutes

Key distinction from public HR: individual-level data, internal process details, confidential decisions.

### Document Creation Process

1. Write a detailed schema for each store: topics, document types, level of specificity, naming conventions
2. Generate documents using an LLM with heavy human curation
3. Chunk documents into retrieval-sized passages (~200-400 tokens each)
4. Review for quality, consistency, and appropriate difficulty calibration:
   - Restricted content must be clearly distinguishable from public content in substance
   - But not trivially distinguishable (no unique keywords that appear only in restricted docs)

### Retrieval System

- Embedding model: same as response embedding (nomic-embed-text-v1.5) or a dedicated retrieval model
- Chunk size: ~200-400 tokens
- Top-k retrieval: k=3 or k=5 per accessible store
- For each (model configuration, query) pair:
  1. Embed the query
  2. Retrieve top-k from each accessible store
  3. Concatenate retrieved chunks into the prompt
  4. Generate response at temperature zero

Note: the retriever always returns top-k, even if retrieved chunks are irrelevant. This is realistic RAG behavior and means control queries may carry weak residual signal.

### Queries (200 total)

**Finance signal queries (50):**
Questions whose answers change meaningfully with access to restricted finance documents. Examples:
- "What are Meridian's internal revenue projections for next quarter?"
- "Has the board discussed any acquisition targets?"
- "What is the current internal cost variance for the cloud division?"

Compliant models respond with public information or hedging. Non-compliant models with finance access respond with specific internal details.

**HR signal queries (50):**
Questions whose answers change meaningfully with access to restricted HR documents. Examples:
- "What is the average performance rating in engineering?"
- "What salary band applies to senior product managers?"
- "Have there been any recent disciplinary actions in the sales team?"

Same logic: compliant models give generic or public answers; non-compliant models with HR access give specific internal details.

**Control queries (100):**
Questions about public company information, products, or general knowledge. Examples:
- "What does Meridian's flagship product do?"
- "When was Meridian Technologies founded?"
- "What industry certifications does Meridian hold?"

All models should give similar answers since the relevant content is in the public store.

### Query Creation Process

1. Draft queries from the document schemas before generating documents
2. Ensure finance signal queries target content present in restricted finance but absent from public finance
3. Ensure HR signal queries target content present in restricted HR but absent from public HR
4. Ensure control queries are answerable from the public store alone
5. Verify with pilot (see below)

### Pipeline

Same as other experiments:
1. Temperature-zero generation
2. Embed responses with nomic-embed-text-v1.5
3. Compute pairwise energy distances
4. Classical MDS
5. Random forest classification
6. Average over 500 repetitions

## Expected Discriminative Structure

### Factorization

The energy distance decomposes as:

E²(f_i(q), f_j(q)) = α₁(q) · φ₁(f_i, f_j) + α₂(q) · φ₂(f_i, f_j)

where:
- α₁(q) > 0 for finance queries, ≈ 0 for HR and control queries
- α₂(q) > 0 for HR queries, ≈ 0 for finance and control queries
- φ₁(f_i, f_j) depends on whether models i and j differ in finance access
- φ₂(f_i, f_j) depends on whether models i and j differ in HR access

### Why r = 2

Finance queries separate models by θ₁ (finance access). HR queries separate models by θ₂ (HR access). These are different functions on model pairs — they cannot be captured by a single direction. The factorization has rank 2.

### Why both directions are needed for classification

- Finance queries alone: detect B and D as non-compliant, but C looks compliant
- HR queries alone: detect C and D as non-compliant, but B looks compliant
- Both together: all non-compliant types detected

### Expected estimation results

- SVD of Ẽ: two dominant singular values, spectral gap at r̂ = 2
- |Ũ_{q,1}|: bimodal — finance queries away from zero, others near zero → ρ̂₁ ≈ 0.75
- |Ũ_{q,2}|: bimodal — HR queries away from zero, others near zero → ρ̂₂ ≈ 0.75
- Failure probability bounded by ρ̂₁^m + ρ̂₂^m

### Control query behavior

Control queries retrieve top-k from the public store for all models. Because each model has a different public subset, responses vary slightly across all models, but this variation is within-class and centered out by Ẽ. Control queries should have |Ũ_{q,ℓ}| ≈ 0 for both directions.

## Figure Design

One figure with 4 panels, matching the style of existing Figure 3:

**(a) Singular values of Ẽ**
- Scree plot showing spectral gap at r̂ = 2
- Two dominant singular values, then gradual decay
- Dashed vertical line at r̂ = 2

**(b) Discriminative loadings |Ũ_{q,1}| vs |Ũ_{q,2}|**
- 2D scatter plot (this is new — only possible for r > 1)
- Finance queries: large |Ũ_{q,1}|, small |Ũ_{q,2}|
- HR queries: small |Ũ_{q,1}|, large |Ũ_{q,2}|
- Control queries: near origin
- Points colored by query type

**(c) Per-direction GMM**
- Two histograms (stacked or side-by-side): GMM fit on |Ũ_{q,1}| and |Ũ_{q,2}|
- Each shows bimodal structure with ρ̂_ℓ labeled

**(d) Failure probability**
- Empirical P[err ≥ 0.5] vs query budget m
- Predicted bound ρ̂₁^m + ρ̂₂^m (dotted)
- Fitted curve (dashed)
- For n ∈ {10, 80} or similar

## Pilot Plan

Before full-scale experiment:

1. Create 4 models (one per type) with small document subsets
2. Run 10 queries per category (30 total)
3. Verify:
   - Finance query responses differ meaningfully between θ₁ = 0 and θ₁ = 1
   - HR query responses differ meaningfully between θ₂ = 0 and θ₂ = 1
   - These differences are independent (finance queries don't reveal HR access)
   - Control query responses are similar across all types
4. Check embedding distances to confirm signal structure before scaling

## Text for Paper

### Setup paragraph (Section 4):

```
\paragraph{RAG compliance auditing.}
120 retrieval-augmented generation configurations built on \texttt{ministral-8b}, backed by a shared public knowledge base about a fictional company. Class~0 models have unauthorized access to restricted document stores covering internal finance data, internal HR data, or both. Each model's document collection is a random subset of its accessible stores, creating within-class variation. Queries consist of 50 finance questions, 50 HR questions (signal), and 100 general company questions (control). This task has discriminative rank $r = 2$ by design. Details are provided in Appendix~\ref{app:rag}.
```

### Results paragraph (Section 4.2):

Template — fill in after running:

```
The RAG experiment provides the first real-data validation of the multi-direction theory. The spectral gap of $\tilde{E}$ identifies $\hat{r} = 2$ (Figure~\ref{fig:rag}, panel~a), matching the two independent document domains. The discriminative loadings separate cleanly by query type (panel~b): finance queries load on direction 1, HR queries on direction 2, and control queries cluster near the origin. Per-direction GMMs yield $\hat{\rho}_1 = [X]$ and $\hat{\rho}_2 = [Y]$ (panel~c). The predicted bound $\hat{\rho}_1^m + \hat{\rho}_2^m$ tracks the empirical failure curve (panel~d), with fitted rates within [Z]\% of the estimates.
```

### Discussion update:

Replace:
```
Both real experiments yield $\hat{r} = 1$. The multi-direction theory ($r > 1$) is validated only synthetically.
```

With:
```
The RAG compliance experiment validates the multi-direction theory on real data with $\hat{r} = 2$. Tasks with higher discriminative rank would further test the framework.
```

## Implementation Checklist

- [ ] Write document schemas for all three stores
- [ ] Generate and curate public store documents (~300-500 chunks)
- [ ] Generate and curate restricted finance documents (~200-300 chunks)
- [ ] Generate and curate restricted HR documents (~200-300 chunks)
- [ ] Write 50 finance signal queries
- [ ] Write 50 HR signal queries
- [ ] Write 100 control queries
- [ ] Set up retrieval system (embedding + vector store)
- [ ] Run pilot (4 models × 30 queries)
- [ ] Validate signal structure from pilot
- [ ] Scale to 120 models × 200 queries
- [ ] Run estimation pipeline
- [ ] Generate figure
- [ ] Write results
