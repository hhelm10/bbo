# On the Orthogonality of Queries in the Black-Box Setting

Framework for understanding **query complexity** in classifying black-box generative models (e.g., LLMs) using multidimensional scaling (MDS) embeddings of pairwise energy distances between response distributions.

Given *n* black-box models that respond to queries, the framework answers: **how many queries *m* are needed to reliably classify models by behavioral type?**

## Installation

```bash
pip install -e .
```

Requires Python >= 3.9. For LLM experiments, set API keys:
```bash
export MISTRAL_API_KEY=...
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
```

---

## Theoretical Framework

### Problem Setup

We observe *n* black-box models *f_1, ..., f_n*, each with an unknown binary label *y_i in {0, 1}*. We can query each model on *m* queries *q_1, ..., q_m* and observe embedded responses *g(f_i(q_k))* in R^p. The goal is to classify the models using as few queries as possible.

### Pipeline

```
queries q_1,...,q_m  -->  responses g(f_i(q_k))  -->  energy distances D_m(f_i,f_j)
    -->  MDS embedding X_i in R^d  -->  classifier  -->  labels y_hat_i
```

1. **Energy distance**: D_m(f_i, f_j) = sqrt( sum_k ||g(f_i(q_k)) - g(f_j(q_k))||^2 )
2. **Classical MDS**: Embed the n x n distance matrix into R^d
3. **Classification**: Random forest on MDS coordinates (70/30 train/test split)

### Key Theoretical Quantities

| Quantity | Symbol | Definition |
|----------|--------|------------|
| Discriminative rank | *r* | Number of latent dimensions carrying class-relevant signal |
| Zero-set probability | *rho_l* | Fraction of queries with zero signal in direction *l* |
| Query complexity bound | *m** | Smallest *m* such that sum_l rho_l^m <= epsilon |
| Bayes rate | *L** | Irreducible classification error (= label noise eta) |

### Main Results

**Theorem 1 (Query Complexity):** P[error >= 0.5] <= sum_l rho_l^m

The probability that the classifier fails to beat chance decays exponentially in *m*, with rate governed by the zero-set probabilities rho_l.

**Theorem 2 (Bayes Convergence):** Mean error --> L* as m, n --> infinity

The mean classification error converges to the Bayes rate, with excess error decaying at rate governed by the same rho_l quantities.

### Estimation from Data

Given precomputed responses, the framework estimates (r_hat, rho_hat) without knowing ground truth:

1. **Compute between-class centered energy tensor** E_disc from pairwise distances
2. **SVD of E_disc**: largest successive ratio in the singular value spectrum gives r_hat
3. **GMM on loadings**: for each direction l = 1..r_hat, fit a 2-component Gaussian mixture to |U_{q,l}|; the weight of the near-zero component estimates rho_hat_l

---

## Synthetic Generative Model

### Bernoulli-Weight Discriminative Field

The synthetic model generates a controlled problem with known ground truth (r, rho, L*).

**Field construction** (M queries, r directions):

```
alpha_l(q) = xi_{q,l} * w_{q,l}

where:
  xi_{q,l} ~ Bernoulli(p)      # activation: does query q probe direction l?
  w_{q,l}  ~ Uniform(0, 1)     # intensity: how strongly?
```

The parameter *p* (signal_prob) controls sparsity: rho_l = 1 - p for all l.

**Model generation** (n models):

For each model f_i:
1. Draw latent type: theta_i ~ Uniform({0,1}^r)
2. Compute label: y_i = parity(theta_i) XOR Bernoulli(eta)
3. Compute sign vector: s_l(i) = 1 - 2*theta_{i,l} in {+1, -1}
4. Compute all M embedded responses:

```
g(f_i(q)) = sum_{l=1}^{r} sqrt(alpha_l(q)) * (1/2) * s_l(i) * v_l
```

where v_1, ..., v_r are orthonormal direction vectors in R^{p_embed} (from QR decomposition).

**Key properties:**
- Parity labels ensure all *r* dimensions must be activated to beat chance
- The factorization ||g(f_i(q)) - g(f_j(q))||^2 = sum_l alpha_l(q) * 1[theta_{i,l} != theta_{j,l}] holds by orthonormality
- When eta > 0, the Bayes rate is L* = eta (irreducible label noise)

---

## Repository Structure

```
bbo/
  api/                        # LLM and embedding API clients
    clients.py                #   Chat completion (Mistral, OpenAI, Anthropic)
    embeddings.py             #   Text embedding (nomic, OpenAI, Google, Voyage)

  models/                     # Model abstractions
    base.py                   #   Abstract Model base class
    llm.py                    #   BenchmarkModel for precomputed responses
    synthetic.py              #   SyntheticModel, SyntheticProblem, make_problem()

  queries/                    # Query sampling
    query_set.py              #   sample_queries(M, m, distribution, rng)
    distributions.py          #   UniformDistribution, SubsetDistribution

  distances/                  # Distance computation
    energy.py                 #   pairwise_energy_distances_t0(), per_query_energy_tensor()

  embedding/                  # Dimensionality reduction
    mds.py                    #   ClassicalMDS with auto-dimension selection

  classification/             # Evaluation
    evaluate.py               #   single_trial(), classify_and_evaluate(), make_classifier()

  estimation/                 # Parameter estimation
    rank_rho.py               #   compute_E_disc(), estimate_discriminative_rank(),
                              #   estimate_rho(), predict_mstar()

  plotting/                   # Figure generation
    style.py                  #   set_paper_style(), PALETTE
    synthetic_plots.py        #   plot_figure_combined() [2x4 synthetic grid]
    real_plots.py             #   plot_real_data_3x3() [3x3 real data grid]
    motivating_plots.py       #   plot_figure1_motivating() [3-col motivating]
    rag_plots.py              #   plot_rag_figure() [3-panel RAG standalone]
    system_prompt_plots.py    #   plot_system_prompt_figures()
    estimation_panels.py      #   plot_estimation_panels() [shared scree/GMM/P[fail]]

  experiments/                # Experiment implementations
    config.py                 #   All experiment config dataclasses
    runner.py                 #   Generic parallel sweep runner

    synthetic/                # Controlled experiments
      exp1_error_vs_m_rank.py #   P[err>=0.5] vs m, varying r
      exp2_error_vs_m_rho.py  #   P[err>=0.5] vs m, varying rho
      exp3_query_distribution.py  # Effect of query sampling distribution
      exp4_error_vs_n.py      #   P[err>=0.5] vs n (sample complexity)
      exp5_bayes_convergence.py   # Bayes rate convergence
      exp_e_error_vs_m_eta.py #   Mean error vs m, varying label noise eta
      exp_f_error_vs_n_eta.py #   Mean error vs n, varying eta
      exp_g_error_vs_m_rank_eta.py  # Mean error vs m, varying r with noise

    motivating/               # Fine-tuned LoRA adapters experiment
      config.py               #   MotivatingConfig
      prepare_data.py         #   Generate queries and training data
      train_adapters.py       #   Fine-tune LoRA adapters
      generate_responses.py   #   Query adapters and collect responses
      embed_responses.py      #   Embed responses with nomic
      run_classification.py   #   MDS + concat classification sweep

    system_prompt/            # System prompt variation experiment
      config.py               #   SystemPromptConfig
      prepare_data.py         #   Generate personas and queries
      generate_responses.py   #   Query models via Mistral API
      embed_responses.py      #   Embed with multiple embedding models
      run_classification.py   #   Classification across base/embed models

    rag/                      # RAG compliance auditing experiment
      config.py               #   RAGConfig
      document_store.py       #   LLM-generated document stores
      system_assignment.py    #   Per-system chunk subset assignments
      retrieval.py            #   Query generation + cosine retrieval
      generate_responses.py   #   RAG prompt construction + API calls
      embed_responses.py      #   Response embedding pipeline
      run_classification.py   #   MDS + RF classification sweep

scripts/                      # CLI entry points
  run_synthetic.py            #   Run synthetic experiments + generate figures
  run_motivating.py           #   Full motivating pipeline (prepare/train/generate/embed/classify/plot)
  run_system_prompt.py        #   System prompt experiment pipeline
  run_rag.py                  #   RAG experiment pipeline
  recompute_failure_probs.py  #   Recompute P[err>=0.5] with parallel reps
  plot_system_prompt_figure.py  # System prompt figure generation
  compute_rank_rho.py         #   Estimate r_hat, rho_hat from real data
```

---

## Core Functions Reference

### Distance and Embedding

| Function | Module | Description |
|----------|--------|-------------|
| `pairwise_energy_distances_t0(responses, query_indices)` | `distances.energy` | Compute n x n energy distance matrix from (n, M, p) response tensor |
| `per_query_energy_tensor(responses)` | `distances.energy` | Compute (M, n_pairs) per-query squared distance tensor for SVD |
| `ClassicalMDS(n_components).fit_transform(D)` | `embedding.mds` | Classical MDS embedding with auto-dimension selection |
| `select_dimension(eigenvalues, n_elbows)` | `embedding.mds` | Zhu-Ghodsi profile likelihood dimension selection |

### Estimation

| Function | Module | Description |
|----------|--------|-------------|
| `compute_E_disc(E, pairs, labels)` | `estimation.rank_rho` | Between-class centered dissimilarity matrix |
| `estimate_discriminative_rank(E_disc)` | `estimation.rank_rho` | Estimate r_hat via largest successive ratio in SVD spectrum |
| `estimate_rho(U, r_hat)` | `estimation.rank_rho` | Per-direction rho_hat via 2-component GMM on \|U_{q,l}\| |
| `predict_mstar(rho_hats, epsilon)` | `estimation.rank_rho` | Query budget m* from bound sum rho_l^m <= epsilon |

### Classification

| Function | Module | Description |
|----------|--------|-------------|
| `single_trial(responses, labels, query_idx, ...)` | `classification.evaluate` | Full pipeline: distance -> MDS -> classify -> error rate |
| `classify_and_evaluate(X, y, classifier_name)` | `classification.evaluate` | Train/test classification on MDS coordinates |
| `make_classifier(name)` | `classification.evaluate` | Factory for sklearn classifiers (rf, knn, lda, svm) |
| `sample_queries(M, m, distribution, rng)` | `queries.query_set` | Sample m query indices from distribution over {0,...,M-1} |

### Synthetic Model

| Function | Module | Description |
|----------|--------|-------------|
| `make_problem(M, r, signal_prob, p_embed)` | `models.synthetic` | Create Bernoulli-Weight synthetic problem |
| `problem.generate_models(n, eta)` | `models.synthetic` | Generate n models with parity labels and optional noise |
| `get_all_responses(models)` | `models.synthetic` | Stack responses into (n, M, p) tensor |
| `get_labels(models)` | `models.synthetic` | Extract label vector |

---

## Experiments

### Synthetic Experiments (Figure 2)

Run all synthetic experiments:
```bash
python scripts/run_synthetic.py
```

The 2x4 combined figure validates both theorems:
- **Row 1**: P[err >= 0.5] vs m/n for varying r, rho, n, eta (Theorem 1)
- **Row 2**: |Mean Error - L*| for the same four experiments (Theorem 2)

### Real Data Experiments

Three real-world applications validate the theory on LLM behavioral classification:

**Motivating Example** (Figure 1): Fine-tuned LoRA adapters with known sensitive-topic bias.
```bash
python scripts/run_motivating.py --step all
```

**System Prompt Experiment**: Detect covert persuasion bias injected via system prompts.
```bash
python scripts/run_system_prompt.py --step all
```

**RAG Compliance Auditing**: Detect unauthorized document store connections in RAG chatbots (r_hat = 2, validating multi-direction theory).
```bash
python scripts/run_rag.py --step all
```

### 3x3 Real Data Figure (Figure 3)

Unified view across all three datasets:
```python
from bbo.plotting.real_plots import plot_real_data_3x3

plot_real_data_3x3(
    motivating_npz="results/motivating/motivating_responses.npz",
    motivating_fail_csv="results/motivating/failure_probs.csv",
    system_prompt_npz="results/system_prompt/embeddings/mistral-small__nomic-embed-text-v1.5.npz",
    system_prompt_fail_csv="results/system_prompt/failure_probs.csv",
    rag_npz="results/rag/embeddings/ministral-8b__nomic-embed-text-v1.5.npz",
    rag_classification_csv="results/rag/classification_ministral-8b__nomic-embed-text-v1.5.csv",
)
```

| Row | Col 1: Scree plot | Col 2: GMM on \|U_{q,l}\| | Col 3: Classification |
|-----|-------------------|---------------------------|----------------------|
| Motivating | r_hat = 1 | rho_hat = 0.59 | P[err >= 0.5] (Thm 1) |
| System Prompt | r_hat = 1 | rho_hat = 0.80 | P[err >= 0.5] (Thm 1) |
| RAG | r_hat = 2 | rho_hat = [0.45, 0.52] | Mean error (Thm 2) |

---

## Key Results

| Dataset | r_hat | rho_hat | rho_fit (P[fail]) | m* (95%) |
|---------|-------|---------|-------------------|----------|
| Motivating | 1 | 0.59 | 0.54 | 6 |
| System Prompt | 1 | 0.80 | 0.83 | 14 |
| RAG | 2 | [0.45, 0.52] | 0.60 (mean err) | 6 |

The estimated rho_hat matches the fitted rho from P[err >= 0.5] curves, confirming the theory.
