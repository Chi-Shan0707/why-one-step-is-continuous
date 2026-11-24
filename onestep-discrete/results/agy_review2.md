Correctness bugs found:

1. sweeps/e4_coupling.py:180-185
   Failure: Non-exchangeable train/test split after Hamming OT. make_pairs() runs linear_sum_assignment(ham) and slices z[:n_tr], x[:n_tr], z[n_tr:], x[n_tr:] without randomly permuting the matched pairs. Because SciPy's Jonker-Volgenant solver processes rows 0..M-1 in order on heavily tied integer Hamming costs {0..6}, early rows (train) systematically get lower-cost, higher-entropy matches while late rows (test) get leftover duplicate modes and higher-cost matches (verified over 5 seeds on mix16: cost_tr ~ 1.34 vs cost_te ~ 1.62, H(x_tr) ~ 5.92 vs H(x_te) ~ 4.92, TV(x_te, p) ~ 0.395 vs 0.344 for random, top-5 mode frequencies ~2x higher in x_te). This causes the permutation-invariant onehot/ot cell to spuriously beat onehot/random across all 3 seeds on both mix16 and markov.
   Confidence: High

2. sweeps/e4_coupling.py:198-202
   Failure: Deployed-law evaluation conflates 80% training-set memorization with generalization for code/onehot. For src in ("code", "onehot"), z_dep = source_input(src, torch.arange(M)) averages over all M = 4096 symbols, 80% of which are the exact training inputs z_tr (memorized to train_seq_acc = 1.0), whereas gauss/unif/quant/shuf evaluate z_dep on 2**16 fresh draws with ~0% training overlap. On mix16, onehot/random has held-out-only KL = 6.49, TV = 0.87, TC = 0.0001, yet reports KL = 0.60-0.76, TV = 0.27, TC = 1.79-1.89 (lower TV than every continuous method); code/ot reports KL = 0.27 (0.65 held-out only), inverting cross-source rankings against continuous/KR baselines.
   Confidence: High

3. sweeps/e4_coupling.py:151
   Failure: kr_decode() uses torch.searchsorted(cdf, u[:, i:i+1].double()) with default right=False (side='left') and .clamp_max(D.V - 1). On distributions with zero-probability entries (e.g., copy, code<r>), u == 0.0 (possible from torch.rand on [0, 1)) selects a leading zero-probability token, making cond.sum() == 0 -> NaN at subsequent positions and corrupting the output (verified: copy at u=[0.3, 0.0, ...] outputs impossible [1, 0, 3, 3, 3, 3]; code16 at u=0 outputs impossible [0, 0, 0, 3, 3, 3]). Likewise, if trailing conditional entries are zero and u >= cdf[-1] (e.g., u=1.0 or float rounding with cdf[-1] < 1), clamp_max(D.V - 1) selects a zero-probability token.
   Confidence: High

4. data.py:55-60
   Failure: codebook(r) claims "needs exactly log2 r bits of shared randomness", but samples r uniformly random states from 4^6 = 4096 without enforcing minimum Hamming distance >= 2. Random Hamming-1 collisions create combinatorial rectangles inside supp(p) coverable by single product components, strictly reducing nonnegative rank below r and required I(Z; X) below log2 r. Verified on seed=0: code16/64/256/1024 have 1 / 9 / 142 / 2,373 Hamming-1 pairs (1015/1024 codewords in code1024 have a Hamming-1 neighbor), yielding exact KL=0 rank <= 15 / 59 / 188 / 518 and exact-fit I(Z; X) <= 3.88 / 5.84 / 7.45 / 8.89 bits (claimed 4 / 6 / 8 / 10 bits); a distance-2 code (e.g., parity check sum(x) % 4 == 0, size <= 1024) is required for the claim to hold.
   Confidence: High

5. sweeps/e6_nscaling.py:3,18-22
   Failure: Inherits the data.codebook(r) failure: sweeps r in [16, 64, 256, 1024] assuming code<r> requires log2 r bits (2^I = r) to test the N >~ 2^I threshold, whereas code256 and code1024 have exact-fit rank <= 188 and <= 518 (I <= 7.45 and <= 8.89 bits, with TC(code1024) = 1.38 nats < TC(code16) = 4.81 nats), invalidating the N = r reference threshold at large r.
   Confidence: High

6. sweeps/e5_boundaries.py:60
   Failure: lift() divides joint by (rate[:, None] * rate[None, :]).clamp_min(1e-12), turning mathematically undefined 0/0 entries (whenever a position has zero flips, rate[i] == 0) into 0.0 instead of NaN, which silently biases spearman_lift_mi, mean_lift_adjacent, and mean_lift_distant.
   Confidence: High

7. sweeps/e5_boundaries.py:65-67
   Failure: spearman() computes ranks with a.argsort().argsort(), which breaks ties arbitrarily instead of using average ranks. On tied vectors (e.g., constant pairwise MI in copy, or tied zero lifts), it produces spurious correlations (verified: returns 1.0 instead of NaN on copy MI; returns 0.20 instead of 0.7746 on [1,1,1,2] vs [3,2,1,4]).
   Confidence: High

8. plots.py:109-110
   Failure: fig_e3() clamps y and lo to FLOOR = 1e-4 (lines 107-108) but does NOT clamp hi = [max(disc[key][h]) for h in H] (line 109). Whenever all seeds at any (series, H) point have kl < 1e-4, hi - y < 0 and ax.errorbar raises ValueError: 'yerr' must not contain negative values (verified; results/e3 already has 2/3 seeds of onehot_m16 at H=4 below 1e-4). Similarly, line 114 (best) is unclamped against FLOOR.
   Confidence: High

9. plots.py:100
   Failure: fig_e3() does not guard against an empty panels list (unlike fig_e4/fig_e6/fig_e7), so plt.subplots(1, 0, ...) raises ValueError if no results/e3/*.json files exist.
   Confidence: High

10. plots.py:166,178
    Failure: fig_e6() places the legend only on axes[0][0] (rv = 16), labeling the categorical curve "categorical K=64" for the entire 4-panel figure even though panels 2-4 actually plot K = 256, 1024, 4096.
    Confidence: High

11. plots.py:167-170,173
    Failure: fig_e6() plots raw r["final"]["kl"] for the sampled curves on a log y-axis without clamping to FLOOR, whereas the enumerated reference line (enum, line 173) IS clamped with max(enum[0], FLOOR). Any sampled run with 0 < kl < 1e-4 renders below the clamped exact-enumeration line (falsely inverting the comparison), and kl <= 0 drops off the log axis; if multiple seeds exist per (src, N), pts = sorted(...) also connects seeds into zigzagging lines instead of averaging.
    Confidence: High

12. plots.py:122,177
    Failure: Both fig_e3() and fig_e6() label the y-axis "exact KL", although gaussian, uniform, and H = 32 discrete runs (K > 2^16) in fig_e3, and Gaussian d=8 in fig_e6, are 65,536-sample Monte Carlo estimates (exact_eval=False), not exact KL.
    Confidence: High

13. plots.py:175,179
    Failure: fig_e6() panel titles claim "code{rv}: needs {int(math.log2(rv))} bits" and suptitle claims "N >~ 2^I (vertical line: N = r)", which is false for data.codebook(rv) at large rv (e.g., code1024 needs <= 8.89 bits / rank <= 518).
    Confidence: High

14. plots.py:51
    Failure: fig_e1() title claims "E1: KL hits 0 at k = r (dotted)", whereas results/e1/mix64.json has kl_best = 0.00328 at k = 64 (above FLOOR = 1e-4, not hitting 0 at k = r due to EM local minima); also FLOOR = 1e-4 clipping is unannotated on fig_e1, fig_e2, fig_e3, and fig_e7 (and fig_e2:65 clamps to FLOOR before averaging across seeds rather than after).
    Confidence: Medium

Verified clean (no correctness bugs found):
- sweeps/e7_teacher_coupling.py: joint law, Bayes-optimal student law p_G (sums to 1.0), I(Z;X), and H(Z) match an independent zero-safe linear-space reference to <= 1.24e-14 across copy, markov, mix16, and code16 for all 20 (kernel, alpha) pairs.
- sweeps/e4_coupling.py: source_input("shuf", z) indexing matches SHUF[i, z[k, i]] exactly (diff 0.0); OT assignment direction x = x[assign(cost)] correctly minimizes pair cost; fm_loss/ode_decode formulas and VAE ELBO/KL match exact references (mix16 gauss/ode target_tv = 0.357 is MLP velocity-fit error, not an ODE/loss bug: exact velocity with the same 100-step Euler gives TV = 0.181).
- sweeps/e5_boundaries.py: pairwise_mi() matches H(X_i) + H(X_j) - H(X_i, X_j) to 8.2e-16; flips() matches gaussian/rademacher inputs (gaussian_levels(2) == [-1, 1]) and yields lift ~ 1 (gaussian) / 0 (rademacher) on independent synthetic generators.
