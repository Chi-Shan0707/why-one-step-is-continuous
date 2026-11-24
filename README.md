# why-one-step-is-continuous

**Why does every good one-step generator, including the discrete ones, get its randomness from
continuous noise?** This repo answers that with a small mechanism study: 6 positions × 4 tokens =
4,096 states, so every distribution is enumerated and the metrics are exact. The exceptions are
continuous noise and discrete sources with more than 2¹⁶ atoms, which use large-sample estimates
(bias ≤ 3.4%).

**Short answer:** bits are not the problem, and neither is representation. With enough entropy,
discrete noise can represent the data almost exactly. On an equal-weight mixture, well-separated
discrete atoms even fit *better* than Gaussian noise. What discrete noise lacks is what one-step
*training* relies on:

| what one-step training needs | continuous noise | discrete noise | evidence |
|---|---|---|---|
| **enough bits**: ≥ rank(p) atoms (log₂K ≥ log₂ rank) | ✓ | ✓ for uniform tokens (✗ for masks: KL = TC exactly) | E1, E2, E3 |
| **mass that can move between modes** | ✓ mass slides with decision boundaries | ✗ allocation flat from step 1k to 16k; for K ≥ 64 total KL 12×–10⁴× above what the atoms could achieve | E8 |
| **a deterministic noise→data transport** to distill from | ✓ marginalizing velocities gives an ODE that cannot branch | ✗ marginalizing jump rates gives a branching CTMC, so teacher couplings are stochastic (I(Z;X) = 0 for the reversed chain, ~1.3 of 8.6 bits for an exact DFM teacher); deterministic ones must be constructed | E7, E7b, E6 |
| **geometry aligned with that transport**, to generalize to unseen z | ✓ | only if aligned: order or Hamming structure (ordered 64-level ≈ continuous, Hamming-OT tokens ≈ Gaussian OT; shuffled or one-hot = z-independent baseline) | E4 |

![mass allocation](onestep-discrete/figures/e8_mass.png)

*E8. Left: for K ≥ 64, total KL of finite sources is far above what their K atoms could achieve
(dashed, an upper bound on the optimum); the best Gaussian is better. Right: each KL is split exactly into mass-allocation error and leakage. Atomic
sources fix their mode weights in the first 1k steps and never correct them. A Gaussian source drives
allocation error to ~2e-4 (half of that is evaluation noise); its remaining error is boundary
leakage, which keeps shrinking.*

![teacher coupling](onestep-discrete/figures/e7_teacher_coupling.png)

*E7, computed exactly. The best one-step student distilled from a discrete teacher is limited by
the information I(Z;X) in the teacher's coupling, not by the entropy of the noise. The reversed
noising chain from its 12-bit prior carries I = 0, so the student is a product (KL = TC), as with
no noise at all. An exact Discrete Flow Matching teacher (stars) keeps ~1.3 of ~8.6 bits on
mix16 and Markov (0.6 of 2 on copy), and the student is still far from the data. A deterministic
(ODE-style) coupling would give KL → 0.*

The full write-up, with every figure, table, caveat, and the hypothesis-by-hypothesis verdicts, is
in **[REPORT.md](REPORT.md)**. The original study plan is
[one_step_discrete_plan.md](one_step_discrete_plan.md).

## Verdicts on the plan's hypotheses

* **H1 (rank)**: confirmed. KL = TC exactly at k = 1, and ≤ 3·10⁻⁶ for k > r. At k = r, EM
  sometimes stalls in local optima (mix64: 0.003).
* **H2 (steps as shared randomness)**: confirmed. Exact k-step laws: the oracle goes from TC at
  k = 1 to 0 at k = L, and the trained MDM tracks it.
* **H3 (shared decision boundaries)**: supported on Markov data. Adjacent positions co-flip
  6.7–8.3× more than chance, with ρ = 0.65–0.85 against mutual information. It also holds for
  hypercube sources, so it is not unique to continuous noise.
* **H4 (geometry ⇒ discrete as good as Gaussian; structureless ⇒ worse)**: **falsified as
  stated**, both for fitting (structureless atoms fit *best* on equal weights) and for
  generalization (the hypercube generalizes as badly as one-hot). Geometry *does* matter in a
  different place: for learning a coupling. There, unordered sources fall to a z-independent
  baseline and ordered discrete matches continuous.

## Experiments

| | question | script | time (RTX 5060 laptop) |
|---|---|---|---|
| E1 | rank threshold (EM) | `sweeps/e1_rank.py` | 15 min |
| E2 | k-step masked diffusion, exact laws | `sweeps/e2_steps.py` | 5 min |
| E3 | source sweep, population and n = 500 | `sweeps/e3_sources.py`, `sweeps/e3_budget.py` | ~3 h |
| E4 | coupling learnability (ODE / OT / VAE / KR × source) | `sweeps/e4_coupling.py` | 40 min |
| E5 | do correlated positions flip together? (toy) | `sweeps/e5_boundaries.py` | 10 min |
| E6 | MC marginal likelihood vs samples per step | `sweeps/e6_nscaling.py` | 25 min |
| E7 | information in a discrete teacher's coupling (exact) | `sweeps/e7_teacher_coupling.py` | 1 min |
| E7b | exact Discrete Flow Matching teacher coupling | `sweeps/e7b_dfm.py` | 15 min |
| E8 | can mass move between modes? | `sweeps/e8_mass.py` | 40 min |

```bash
cd onestep-discrete
python sweeps/e7_teacher_coupling.py     # fastest result: exact, about 1 minute
python run.py --data mix16 --source gaussian --d 4   # one config -> one JSON in results/
python plots.py                           # every figure, from results/*.json only
```

Requirements: Python ≥ 3.10, PyTorch ≥ 2.0 (CUDA optional), SciPy, Matplotlib (`pip install -r onestep-discrete/requirements.txt`).

## Layout

```
onestep-discrete/
  data.py      data families, exact p(x), TC, KL
  sources.py   every P_Z: gaussian, uniform, quantized, rademacher, shuffled, onehot, categorical
  model.py     MLP z -> L x V logits
  losses.py    MC marginal likelihood, regression CE
  exact.py     exact / large-N marginal of a one-step generator; KL = allocation + leakage
  run.py       one config -> one JSON (resumable)
  sweeps/      E1-E8 (+E3b, E7b)
  plots.py     figures from JSONs only
  results/     JSONs of every run reported here
  figures/
REPORT.md      the write-up
```

## Method notes

* One generator (MLP, 3 layers, width 256) and, within each experiment, one objective for every
  source: −Σₓ w(x) log (1/N) Σₙ ∏ᵢ p_θ(xᵢ | zₙ) (N = 256 in E3). Only P_Z changes. E4 uses
  supervised regression on fixed pairs instead.
* Every reported KL is exact by enumeration. The exceptions are sources with more than 2¹⁶ atoms
  and continuous sources, which use a 2¹⁶-sample MC estimate (median relative bias 0.2%, max
  3.4%, checked on 151 runs).
* The code and the report were checked three times by an independent agent, including a full
  fact-check of every number. The verified findings were fixed and the affected experiments
  re-run (the DFM teacher, E7b, was added in response). The git history is the research log.
