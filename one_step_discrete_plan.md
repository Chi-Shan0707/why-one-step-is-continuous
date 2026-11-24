# One-Step Discrete Generation: A Small Mechanism Study

*Working plan. Scope: a gaming laptop, toy data, exact metrics.*

---

## 1. Motivation

Every successful one-step generator I know of gets its randomness from **continuous** noise. The Gaussian methods do this directly (consistency models, MeanFlow, flow maps). The discrete ones do it too, whenever they work well:

- **FMLM** and **Categorical Flow Maps** run a continuous flow over one-hot vectors or the simplex.
- **Duo** treats uniform-state diffusion as the argmax of a Gaussian diffusion.
- **Coupling Models** decode tokens from a latent $z \sim \mathcal N(0,I)$.

Discrete one-step methods that stay discrete (Di[M]O, ReDi) do exist. They still borrow continuity for training, through soft embeddings or straight-through estimators.

So I want to know two things:

> **What exactly is missing from a discrete source of randomness?** And how much of it is information theory (a hard limit), and how much is geometry and optimization (a soft limit)?

The aim is to test a few concrete mathematical hypotheses with small experiments that can be checked exactly. Benchmark performance is not the aim.

---

## 2. Setting

A one-step generator takes a source sample $Z$, runs one forward pass, and samples every position independently:

$$X_i \sim p_\theta(\cdot \mid Z), \qquad p_\theta(x) = \mathbb E_Z \Big[\prod_{i=1}^L p_\theta(x_i \mid Z)\Big].$$

This is a **mixture of product distributions** indexed by $Z$. Any correlation between positions has to be carried by $Z$.

---

## 3. Hypotheses

| | Statement | Status |
|---|---|---|
| **H1** (rank) | If $Z$ takes $k$ values, $p_\theta$ can match $p$ iff the nonnegative tensor rank of $p$ is at most $k$. A deterministic source (all-mask) gives only product distributions, and the best one has $\mathrm{KL}(p\,\|\,p_\theta) = TC(p)$. | Theorem. Sanity check. |
| **H2** (self-made randomness) | Multi-step samplers work because tokens committed in earlier steps act as shared randomness. With $k$ steps, the error falls from $\approx TC(p)$ at $k=1$ to $\approx 0$ at $k=L$. | Known. Reproduce. |
| **H3** (geometry) | Continuous one-step maps carry correlation through **shared decision boundaries**. Positions that are correlated flip together when $z$ crosses a boundary. | Interpretation. Test it. |
| **H4** (structure) | Suppose $H(Z)$ is above the common information of $p$. Then discrete sources **with geometry** (quantized Gaussian, Rademacher) perform about as well as Gaussian. Discrete sources **without geometry** (shuffled levels, categorical codes) are clearly worse in optimization and in generalization. | **Conjecture. The main target.** |

**H4 is falsified if** structureless discrete sources with enough bits match Gaussian under the same objective. The obstacle would then lie elsewhere, for example in gradients or in the training objective. That would be a useful result in its own right.

---

## 4. Shared Harness

Everything runs in one small codebase. Only the variable under study changes between runs.

**Data.** Use $V=4$, $L=6$, which gives $4^6 = 4096$ states. That is small enough to enumerate, so every metric is exact.

| Family | Why |
|---|---|
| Mixture of $r$ random products, $r \in \{1,2,4,16,64\}$ | Nonnegative rank $\le r$ is known, and the common information is $\lesssim \log_2 r$ bits. This gives a controllable threshold. |
| Copy: $x_1 = \dots = x_L$ uniform | Extreme case, with nonnegative rank $=V$. |
| First-order Markov chain | Local correlation, closer in spirit to language. |

**Metrics.** These are computed exactly by enumeration:

- $\mathrm{KL}(p\,\|\,p_\theta)$ and TV;
- the model's own $TC(p_\theta)$;
- *generalization*: train on $n \in \{500, 5000\}$ samples and report the exact KL to the true $p$.

**Model.** An MLP with input $z\in\mathbb R^{d_z}$ and output $L\times V$ logits: 3 layers, width 256.

**Unified objective.** This works for every source type:
$$\mathcal L = -\mathbb E_{x\sim \text{data}} \log \frac1N\sum_{n=1}^N \prod_i p_\theta(x_i\mid z_n), \quad z_n \sim P_Z.$$
Compute it with log-sum-exp and use $N=256$. The estimate has a Jensen bias, so report the exact KL with a large $N$ at evaluation. For a source with finitely many values $K$, enumerate all of them when $K \le 4096$.

Flow matching and MeanFlow serve only as reference methods for the continuous case. Their ODEs degenerate when the noise is atomic, so they cannot be used to compare noise types.

---

## 5. Experiments

### E1: Rank threshold (CPU, minutes)
Fit a $k$-component latent-class model, a mixture of products, with EM. Do this for the mixture-of-$r$ data while sweeping $k$.
**Expect:** KL equals $TC(p)$ at $k=1$, falls monotonically, and is $\approx 0$ for $k \ge r$. This checks H1 and validates the harness.

### E2: Steps as shared randomness (CPU/GPU, < 30 min)
Train a small masked diffusion model on the Markov-chain data. Sample with $k \in \{1,2,3,6\}$ steps, unmasking $L/k$ tokens per step.
**Expect:** KL $\approx TC(p)$ at $k=1$ and $\approx 0$ at $k=6$. This checks H2.

### E3: Source sweep (main experiment, GPU helps, ~2–4 h total)
Keep the generator and the objective fixed. Change only $P_Z$:

| Source | Discrete? | Geometry? |
|---|---|---|
| Gaussian $\mathcal N(0,1)$ | no | yes |
| Uniform $U[-\sqrt3,\sqrt3]$ | no | yes |
| Quantized Gaussian, $2/4/16/256$ levels, standardized | yes | yes (ordered) |
| Rademacher $\{\pm1\}$ | yes | yes (hypercube) |
| Shuffled quantized: levels randomly permuted per dimension | yes | **no order** |
| Categorical: $K$ symbols with fixed random embeddings | yes | **none** |

Sweep the total source entropy $H(Z) = d_z \log_2(\text{levels})$ from below $\log_2 r$ to far above it.
**Plot:** final KL against $H(Z)$, one curve per source. Do this for the training distribution and for the finite-sample generalization setting.
**Expect:** every source fails below the threshold, which is the information limit. Above the threshold, the sources split by geometry if H4 holds.

### E4: Coupling quality (GPU, ~1–2 h)
Train $G$ by **supervised regression on fixed $(z, x)$ pairs**, then test it on held-out $z$:
- (a) Gaussian $z$ with the ODE coupling from a one-hot flow-matching model;
- (b) categorical $z$ with a random coupling;
- (c) categorical $z$ with a Hamming-OT-style coupling;
- (d) Gaussian $z$ with a learned coupling (Coupling Models Stage A, simplified).

**Expect:** (a) ≈ (d) > (c) > (b) on held-out accuracy and on distributional KL. This isolates the coupling's geometry from the amount of information.

### E5 (optional): Probe a real model
Load an FMLM LM1B checkpoint and run inference only. Interpolate between pairs $z \to z'$ and log where each position flips.
**Test for H3:** flips cluster across positions, and the clustering is stronger for positions that are semantically linked. Compare against a shuffled-position null.

---

## 6. Laptop Strategy

**Assumed hardware:** a single consumer GPU with about 8 GB, or CPU only. Every experiment except E5 fits on CPU; the GPU only speeds up the sweeps.

**Principles**
- Use tiny data with exact metrics. Never use FID or perplexity here.
- Change one variable per sweep and use 3 seeds per configuration.
- Every run writes a single JSON with its config, metrics, and wall time. Plots are generated from those JSONs only.
- Cap each run at about 5 minutes. A run that needs more is a design smell.

**Layout**
```
onestep-discrete/
  data.py      # families + exact p(x), TC, enumeration
  sources.py   # all P_Z variants
  model.py     # MLP -> L×V logits
  losses.py    # MC marginal likelihood, regression loss
  exact.py     # exact KL / TV / TC of p_theta
  run.py       # one config -> one JSON
  sweeps/      # E1–E4 sweep scripts
  plots.py
  results/     # JSONs (git-ignored)
```

**Reuse**, ranked by how well each repo suits mechanism work:
1. `tunno-ng/onestep_eval` is a minimal skeleton for FM and MeanFlow on low-dimensional data. Borrow its structure. It is community code with no discrete support, so test it before trusting it.
2. `pengzhangzhi/Coupling-Models` is the reference for "Gaussian latent → factorized decoder". Port a simplified Stage A/B for E4(d).
3. `sony/di4c` is a reference for mixture parameterizations (E1). Its dependency stack is heavy.
4. `david3684/flm` provides the checkpoints for E5. Do not train it locally. Replace flash-attn with PyTorch SDPA if needed.
5. MeanFlow (unofficial) runs on continuous images only and adds little here.

**Milestones and decision points**
1. **Harness + E1.** Pass if the KL at $k=1$ matches $TC(p)$ to 1e-3. If it does not, fix the harness before going further.
2. **E2.** This is a reproduction only; spend at most half a day on it.
3. **E3.** Main result.
   - If the sources separate by geometry, H4 is supported. Go on to E4 to locate the cause in the coupling.
   - If they do not separate, H4 is falsified. Pivot: compare the training *objectives* (likelihood, STE, REINFORCE) on the same structureless source.
4. **E4, then optionally E5.**
5. **Write-up:** one figure per hypothesis and a short blog post or workshop note.

---

## 7. References

- Hayakawa et al., *Distillation of Discrete Diffusion through Dimensional Correlations* (Di4C), ICML 2025.
- Yoo et al., *ReDi: Rectified Discrete Flow*, NeurIPS 2025.
- Yoo et al., *Infinite Mask Diffusion for Few-Step Distillation* (IMDM), 2026.
- Sahoo et al., *The Diffusion Duality* (Duo), ICML 2025.
- Zhu et al., *Di[M]O* (ICCV 2025) and *Soft-Di[M]O* (ICLR 2026).
- Lee et al., *Flow Map Language Models* (FMLM), 2026.
- Roos et al., *Categorical Flow Maps*, ICML 2026.
- Peng et al., *Coupling Models for One-Step Discrete Generation*, 2026.
- Wyner, *The Common Information of Two Dependent Random Variables*, 1975.
