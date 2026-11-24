root agent idle; waiting for 1 background task(s) to finish
I'm running a quick read-only diagnostic check on the code's numerical behaviors and will analyze the output as soon as it finishes.
Waiting for the CPU diagnostic check to complete before finalizing the review.
Waiting for the verification checks to complete.
### Findings

#### 1. `oracle_cond` produces unnormalized conditionals (summing to $V$) whenever $p(x_U) = 0$
- **Location:** [e2_steps.py:84](file:///mnt/d/CS/GenerativeModels/One-Step-Should-be-Continuous/onestep-discrete/sweeps/e2_steps.py#L84)
- **Failure:** `(marg(keep + [i]).clamp_min(1e-300) / den.clamp_min(1e-300)).log()` evaluates `log(1e-300 / 1e-300) = 0.0` whenever `den == 0` (which always implies `marg(keep + [i]) == 0`). That sets $q(x_i = v \mid x_U) = 1.0$ for **all** $V=4$ symbols ($\sum_v q(v \mid x_U) = 4$). On `copy` (which has structural zeros), multi-token unmasking produces inconsistent prefixes with $p(x_U)=0$, so `k_step_logprob(orc, k).exp()` is not a valid distribution: it sums to **`60.0625`** at $k=2$ and **`192.25`** at $k=3$ (yielding `tv = 30.47` and `96.375` instead of $\le 1$). `kl_oracle` at [e2_steps.py:124](file:///mnt/d/CS/GenerativeModels/One-Step-Should-be-Continuous/onestep-discrete/sweeps/e2_steps.py#L124) escapes corruption only because `D.kl` sums strictly over $p(x) > 0$, where every subset marginal $p(x_U) \ge p(x) > 0$; any metric touching $p(x)=0$ (TV, TC, reverse KL, sampling) is broken. Fall back to `-math.log(D.V)` where `den == 0` (and `-inf` where `den > 0, num == 0`).
- **Confidence:** High

#### 2. `categorical` embeddings are not standardized across the $K$ atoms
- **Location:** [sources.py:85-86](file:///mnt/d/CS/GenerativeModels/One-Step-Should-be-Continuous/onestep-discrete/sources.py#L85-L86) (cf. [sources.py:1](file:///mnt/d/CS/GenerativeModels/One-Step-Should-be-Continuous/onestep-discrete/sources.py#L1))
- **Failure:** `torch.randn(K, CAT_DIM, generator=g)` is never centered or variance-scaled over the $K$ atoms, unlike `quantized`, `rademacher`, `shuffled`, and `onehot` (all exact mean `0`, variance `1` per coordinate). Under the uniform distribution over the $K$ atoms, the actual mean vector has expected norm $\sqrt{32/K}$ (for `seed=0`: `4.02` at $K=1$, `3.74` at $K=2$, `2.62` at $K=4$, `1.96` at $K=8$, `1.29` at $K=16$) and expected per-coordinate population variance $(K-1)/K$ (`0.5` at $K=2$, `0.75` at $K=4$) with wide per-coordinate spread (e.g., `0.000`–`3.825` at $K=2$, `0.080`–`3.681` at $K=4$, `0.355`–`1.646` at $K=16$). At $K=1$, the single atom sits at a random nonzero vector instead of `0`. This systematically distorts input mean/scale for `categorical` right in the threshold regime ($K \le 16$).
- **Confidence:** High

#### 3. Training with `enumerate=True` vs `enumerate=False` compares two different objectives
- **Location:** [run.py:36](file:///mnt/d/CS/GenerativeModels/One-Step-Should-be-Continuous/onestep-discrete/run.py#L36), [run.py:60-62](file:///mnt/d/CS/GenerativeModels/One-Step-Should-be-Continuous/onestep-discrete/run.py#L60-L62), [run.py:75](file:///mnt/d/CS/GenerativeModels/One-Step-Should-be-Continuous/onestep-discrete/run.py#L75), [losses.py:9-16](file:///mnt/d/CS/GenerativeModels/One-Step-Should-be-Continuous/onestep-discrete/losses.py#L9-L16)
- **Failure:** If `enumerate=True` is enabled during training for finite sources ($K \le 4096$) while continuous sources (and finite sources with $K > 4096$) use `enumerate=False` (`N=256`), finite sources optimize the **exact** marginal NLL (`KL(p || p_theta) + H(p)`) with deterministic gradients and batch size $K$, whereas MC-trained sources optimize a biased $N$-sample IWAE upper bound with stochastic gradients. At `N=256`, even at the exact ground-truth mixture, `mc_nll` has a mean Jensen bias of **`0.029` nats** on `mix16` and **`0.130` nats** on `mix64` (>11% of `TC(mix64)=1.15`, where any given component is missed completely with prob $(63/64)^{256} \approx 1.8\%$ per step), pushing the minimizer toward blurred/higher-entropy components. Keep `enumerate=False` with identical `N` across all sources during training (and enumerate only at evaluation).
- **Confidence:** High

#### 4. `rademacher` collides with `quantized(m=2)` on output filename / cache
- **Location:** [sources.py:74-77](file:///mnt/d/CS/GenerativeModels/One-Step-Should-be-Continuous/onestep-discrete/sources.py#L74-L77), [run.py:40-49](file:///mnt/d/CS/GenerativeModels/One-Step-Should-be-Continuous/onestep-discrete/run.py#L40-L49)
- **Failure:** `make_source("rademacher", d=d)` rewrites `kind, m = "quantized", 2` and returns `Source.name == f"quantized_d{d}_m2"`. Because `run.py` caches and writes results by `Config.name()` (which uses `src.name`), running both `rademacher` and `quantized(m=2)` in the same `exp` maps to the exact same JSON path: without `--force`, the second silently returns the first's cached JSON (with the wrong `config["source"]`); with `--force`, it overwrites the first.
- **Confidence:** High

#### 5. `gaussian_levels(1)` and `onehot(m=1)` silently produce `NaN` tables
- **Location:** [sources.py:22-28](file:///mnt/d/CS/GenerativeModels/One-Step-Should-be-Continuous/onestep-discrete/sources.py#L22-L28), [sources.py:82-84](file:///mnt/d/CS/GenerativeModels/One-Step-Should-be-Continuous/onestep-discrete/sources.py#L82-L84)
- **Failure:** At `m=1`, both divide `0.0 / 0.0` in PyTorch tensor arithmetic and silently return `tensor([nan])` / `tensor([[nan]])` without raising, poisoning any run that sweeps `m=1` ($H=0$ bits).
- **Confidence:** High

#### 6. Unseparated RNG seeds couple model init and `uniform` step-1 noise on CPU (exact `corr = 1.0`)
- **Location:** [run.py:51-54](file:///mnt/d/CS/GenerativeModels/One-Step-Should-be-Continuous/onestep-discrete/run.py#L51-L54) (and [run.py:64](file:///mnt/d/CS/GenerativeModels/One-Step-Should-be-Continuous/onestep-discrete/run.py#L64))
- **Failure:** `torch.manual_seed(cfg.seed)` (used for `MLP` init), `gen = torch.Generator(DEVICE).manual_seed(cfg.seed)` (used for `src.sample`), and `D.empirical(..., seed=cfg.seed)` reuse the exact same raw seed without domain offsets (unlike `make_source`, which uses `1000 + seed`). On CPU, `gen` and the default CPU RNG start in the identical MT19937 error: interrupted
