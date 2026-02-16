# Computational Model for Synthetic Experiments

## Bernoulli-Weight Model

We define a generative model for the discriminative structure of a classification problem over black-box models, parameterized by the number of discriminative dimensions $r$, the activation probability $p$, and the number of queries $M$.

### Discriminative field

The $M \times r$ weight matrix is generated independently:

$$\alpha_\ell(q) = \xi_{q\ell} \cdot w_{q\ell}$$

where $\xi_{q\ell} \sim \text{Bernoulli}(p)$ controls whether query $q$ activates dimension $\ell$, and $w_{q\ell} \sim \text{Uniform}(0,1)$ controls the signal intensity. The activation probability $p \in (0, 1]$ is the single tuning parameter.

### Models

Each model $f$ has a latent type vector $\theta_f \in \{0, 1\}^r$. The class label is $y = \theta_{f,1}$ (or any fixed function of $\theta_f$). The model-pair kernel is:

$$\phi_\ell(f, f') = \mathbf{1}[\theta_{f,\ell} \neq \theta_{f',\ell}] \cdot c_\ell$$

where $c_\ell > 0$ is a fixed scale. The squared energy distance at query $q$ between models $f, f'$ is then:

$$\mathcal{E}^2(P_f(q), P_{f'}(q)) = \sum_{\ell=1}^r \alpha_\ell(q) \, \phi_\ell(f, f')$$

At temperature zero, this is realized by having the response $f(q)$ depend on $\theta_f$ along the activated dimensions: if $\alpha_\ell(q) > 0$ and $\theta_{f,\ell} = 0$, the response reflects token $a_\ell$; if $\theta_{f,\ell} = 1$, the response reflects token $b_\ell$. If $\alpha_\ell(q) = 0$, query $q$ reveals nothing about dimension $\ell$.

### Properties

**Zero sets.** Since $\alpha_\ell(q) = 0$ iff $\xi_{q\ell} = 0$, each zero set has mass $\Pi_Q(\mathcal{Z}_\ell) = 1 - p$ under uniform $\Pi_Q$.

**Active dimensions per query.** $\text{Binomial}(r, p)$ with expectation $pr$.

**Orthogonal queries.** A query is orthogonal to all pairs iff $\xi_{q\ell} = 0$ for all $\ell$, which occurs with probability $(1-p)^r$.

**Query complexity.** From the corollary, $m^* = \lceil p^{-1} \log(2r/\epsilon) \rceil$.

### Spectrum of regimes

The single parameter $p$ interpolates between sparse and dense discriminative structure:

| $p$ | Regime | Active dims per query | $\rho$ | $m^*$ | Character |
|-----|--------|-----------------------|--------|-------|-----------|
| $1/r$ | Sparse | $\sim 1$ | $1 - 1/r$ | $\sim r \log r$ | Each query probes ~1 dimension; coupon-collector scaling |
| $0.3$ | Moderate | $\sim 0.3r$ | $0.7$ | $\sim 3 \log r$ | Each query probes several dimensions; realistic for LLMs |
| $0.5$ | Overlap | $\sim r/2$ | $0.5$ | $\sim 2 \log r$ | Substantial redundancy across queries |
| $1$ | Dense | $r$ | $0$ | $1$ | Every query probes every dimension; single query suffices |

### Use in experiments

**Exp 1 (Error vs $m$ for varying $r$).** Fix $p = 0.3$, vary $r \in \{2, 5, 10, 25, 50, 100\}$. All curves should have slope $\log(1 - p) = \log 0.7$ on a log scale, with intercept $\log r$.

**Exp 2 (Error vs $m$ for varying $\rho$).** Fix $r = 5$, vary $p \in \{0.1, 0.3, 0.5, 0.8\}$. Slope changes as $\log(1-p)$.

**Exp 3 (Query distribution).** Fix $r = 5$, $p = 0.3$. Compare uniform $\Pi_Q$, $\Pi_Q$ concentrated on queries with many active dimensions (high $\sum_\ell \alpha_\ell(q)$), and $\Pi_Q$ concentrated on orthogonal queries.
