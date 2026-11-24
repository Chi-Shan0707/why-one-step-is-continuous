1.
Location: onestep-discrete/sweeps/e7b_dfm.py (lines 60-62); REPORT.md (lines 221-222, 230, 328)
Claim: e7b_dfm.py logs max_tv_marginal_vs_pt as a marginal check throughout integration; REPORT.md (lines 221-222) claims "the marginal matches p_t to <= 7e-4 TV throughout", and REPORT.md (line 328) claims "E7b is deterministic up to Euler error <= 7e-4 TV".
Truth: The DFM marginal CTMC rates, forward Kolmogorov equation, inflow/outflow tensor indexing, and s = -log(1 - t) time change are correct (verified to 3.1e-13 TV against scipy DOP853 on V = 3, L = 3), but the marginal check and reported error bound are wrong in three ways:
  (a) Even in the saved JSONs, onestep-discrete/results/e7/markov_dfm.json has max_tv_marginal_vs_pt = 0.0007121 (7.12e-4), which already exceeds 7e-4.
  (b) e7b_dfm.py line 60 checks only `if k % 500 == 0:`, evaluating 5 of 2,400 steps (k = 0, 500, 1000, 1500, 2000, i.e. t = 0.005, 0.918, 0.993, 0.9994, 0.99995) and skipping t in (0.005, 0.918) where the Euler marginal TV error peaks. Checking across the full trajectory gives peak marginal TV against p_t of 7.10e-4 on mix16 (at k = 190, t = 0.615; logged 5.38e-4), 8.88e-4 on markov (at k = 170, t = 0.575; logged 7.12e-4), and 1.80e-3 on copy (at k = 240, t = 0.700; logged 5.54e-4, understated by 3.25x).
  (c) REPORT.md line 328 conflates marginal TV with Euler error of the joint coupling and student metrics. On copy, refining ds from 0.005 to 0.001 shifts student_kl from 2.1482 to 2.1366 (Richardson limit ~2.134, ~0.015 nats Euler error) and I_bits from 0.6414 to 0.6453 (limit ~0.646, ~0.005 bits), affecting the last reported decimal digit for copy in REPORT.md line 230 (2.15 and 0.64 vs true CTMC ~2.13 and 0.65).
Confidence: High

2.
Location: README.md (lines 16, 22); REPORT.md (lines 30, 295-296)
Claim: In E8, finite sources' total KL is "12x-10^4x above what the atoms allow" (README.md:16), "far above what their K atoms could achieve (dashed)" (README.md:22), "12x to more than 10^4x above what the atoms could achieve" (REPORT.md:30), and "Their total KL is 12x (K = 64) to more than 10^4x (K = 4096) above what their atoms could achieve" (REPORT.md:295-296).
Truth: E8 also tests K = 16 (categorical K = 16 and Rademacher d = 4), where onestep-discrete/results/e8/floors.json gives 0.4137. Mean total KL is 0.4539 for categorical K = 16 (1.10x, not >= 12x) and 0.6586 for Rademacher d = 4 (1.59x). Furthermore, two of three categorical K = 16 seeds (s0: 0.3875, s2: 0.3786; s1: 0.5955) beat 0.4137, showing that README.md's phrasing "what the atoms allow" overstates achievable(K) (which REPORT.md:284-285 correctly defines as an upper bound from sharp single-codeword assignment, not a hard lower bound). The 12x to >10^4x range (12.1x to 35,570x) applies only to K >= 64.
Confidence: High

3.
Location: REPORT.md (lines 30-31, 292, 299-300)
Claim: The E8 table (line 292) gives Rademacher d = 12 total KL as 0.025; lines 30-31 state "on unequal weights every finite source but one fits worse than Gaussian"; lines 299-300 state "On total KL it beats every finite source except Rademacher d = 12, which ties it (0.025)."
Truth: In onestep-discrete/results/e8/zipf16_pop_quantized_d12_m2_Nenum_s{0,1,2}.json, the three seeds are 0.012578, 0.032567, and 0.028207, whose mean is 0.024451, rounding to 0.024 rather than 0.025. Also, the Gaussian comparison depends on d: Gaussian d = 2 (0.0178) beats every finite source (including Rademacher d = 12, which is 38% worse), whereas Gaussian d = 4 (0.0270) and d = 8 (0.0266) are beaten by Rademacher d = 12 (0.0245). It approximates a tie only against the unstated pooled mean across Gaussian d = 2, 4, 8 (0.0238).
Confidence: High

4.
Location: REPORT.md (lines 236-237)
Claim: "On copy the masking kernel is 1.7-2.6x better than the uniform kernel at equal I, because one unmasked token pins the whole sequence."
Truth: On copy, the masking kernel's exact student KL at mutual information I (bits) is KL_mask(I) = -log(I/2 + (1 - I/2) * 4^-5). Comparing exact uniform student_law against exact KL_mask(I) at identical I across all uniform rows in onestep-discrete/results/e7/copy.json (alpha = 0.02..0.9, I = 0.005..1.998) yields KL_unif / KL_mask = 1.21-1.73x (and 1.61-1.74x over a dense sweep I in [0.1, 1.98]; even at alpha = 0.999, 2 - I = 9.6e-10, the ratio is 1.91x). The 2.6x figure is an artifact of log-linearly interpolating between coarse mask rows in copy.json (alpha = 0.3 and 0.5) at I = 1.9067, which underestimates KL_mask as 0.0296 instead of 0.0477 (0.0774 / 0.0296 = 2.62 vs true 1.62).
Confidence: High

5.
Location: README.md (lines 17, 33); REPORT.md (lines 43-44, 322)
Claim: README.md line 33 (caption for e7_teacher_coupling.png, which plots stars for copy, markov, and mix16) states "An exact Discrete Flow Matching teacher (stars) keeps ~1.3 bits and the student is still far from the data." README.md line 17, REPORT.md lines 43-44, and REPORT.md line 322 cite ~1.3 of 8.6 bits (or I = 1.25-1.30 bits of H(X) ~ 8.6, student KL 1.07-1.41 vs TC 1.75-2.01) without specifying that this excludes copy.
Truth: Per onestep-discrete/results/e7/copy_dfm.json, the copy DFM star is at I = 0.641 bits out of H(X) = 2.000 bits (32% of H(X), student KL 2.148 vs TC 6.931, recovering 69% of TC), not ~1.3 bits. Only markov (1.250 of 8.559 bits) and mix16 (1.304 of 8.704 bits) lie in the 1.25-1.30 of ~8.6 range.
Confidence: High

6.
Location: REPORT.md (lines 235-236)
Claim: "On mix16 and Markov the student's error is essentially a function of the coupling's I(Z;X), and the DFM points fall on the forward-joint curves."
Truth: At DFM's exact I(Z; X), DFM student KL is 10-14% higher than both forward-joint curves rather than on them: on mix16 (I = 1.3038 bits), DFM KL is 1.4147 vs uniform forward-joint 1.2819 (+10.4%) and mask forward-joint 1.2726 (+11.2%); on markov (I = 1.2498 bits), DFM KL is 1.0693 vs uniform 0.9360 (+14.2%) and mask 0.9589 (+11.5%).
Confidence: High

7.
Location: README.md (lines 5-6, 97-98); embedded E3 figure via onestep-discrete/plots.py (line 132) in REPORT.md (line 126)
Claim: README.md lines 5-6 state "The one exception is continuous noise, which uses large-sample estimates with < 4% bias." README.md lines 97-98 state "One generator (MLP, 3 layers, width 256) and one objective for every source: ... with N = 256 fresh samples per step. Only P_Z changes." The E3 figure footnote (plots.py:132, rendered in REPORT.md:126) states "bias < 1%".
Truth: Continuous noise is not the only exception: exact.py line 32 also uses 2^16-sample MC evaluation for finite sources with K > 2^16, and 39 of the 151 MC-evaluated runs in results/ (exact_eval: false) are discrete (H = 32 configs in E3 and E3b), as README.md lines 99-100 itself notes. In addition, N = 256 describes E3, not all experiments: E8 uses the enumerated objective for finite sources and N = 1024 for Gaussian, E3b uses N = 256 and 1024, E6 sweeps N in {16..4096} plus enumerated, and E4 uses supervised regression CE (plus width 512 for the FM velocity net). Finally, within E3 itself, the quarter-N implied bias (kl_quarter - kl) / (3 * kl) reaches 2.07% (onestep-discrete/results/e3/mix16_pop_onehot_d8_m16_N256_s2.json), exceeding the "< 1%" printed in the E3 figure footnote.
Confidence: High

8.
Location: README.md (line 17); REPORT.md (lines 36-40, 52-54, 238-241, 321-322)
Claim: Because a deterministic map on a finite set cannot follow a continuous marginal path p_t, "Every diffusion- or flow-style discrete teacher must inject fresh randomness, and its coupling is stochastic" (REPORT.md:38-40; similarly README.md:17 and REPORT.md:238-241), and "the training coupling should come from a deterministic transport, which only continuous-state dynamics provide" (REPORT.md:53-54).
Truth: Stochastic intermediate dynamics (random jump times in a CTMC) do not imply a stochastic endpoint coupling (X_0, X_1). Counterexample on copy: start X_0 ~ Uniform(V^L), keep position 0 fixed (rate 0), and for i = 1..5 jump x_i -> x_0 at rate 1 / (1 - t) whenever x_i != x_0. This finite-state CTMC realizes the smooth continuous marginal path p_t(x) = (1/V) prod_{i>=1} [t 1[x_i = x_0] + (1 - t)/V] from Uniform(V^L) to copy, yet yields the 100% deterministic endpoint coupling X_1 = (X_{0,0}, ..., X_{0,0}) a.s., with I(X_0; X_1) = H(X_1) = 2 bits and student KL = 0. Even for non-quantized targets mix16 and markov, monotone couplings from Uniform(4096) retain 8.43 / 8.70 (96.8%) and 8.28 / 8.56 (96.7%) of H(X). What limits E7b's DFM teacher to 1.25-1.30 bits (or 0.64 on copy) is specifically the independent-coupling (p_0 x p_1) uniform mixture path, not finiteness of the state space alone; coupled or rectified discrete flows (such as OT-DFM or ReDi) are not tested.
Confidence: High

9.
Location: README.md (line 15); REPORT.md (lines 17, 59)
Claim: "H(Z) >= log2 rank(p)" (README.md:15, REPORT.md:59) and "Information theory gives a hard limit, and it is real but cheap. Z must carry at least log2(rank) bits. Below that every source fails identically." (REPORT.md:17).
Truth: Nonnegative tensor rank lower-bounds support size |supp(Z)| >= rank_+(p), i.e. Hartley entropy log2 |supp(Z)| >= log2 rank_+(p), not Shannon entropy H(Z) in general. They coincide only for uniform finite sources (the ones tested in E3). Counterexample from this repo: zipf16 has rank_+(p) = 16 (log2 rank = 4.0 bits), yet a non-uniform discrete source Z ~ Zipf(16) achieves KL = 0 with H(Z) = H(zipf16) = 3.403 bits < 4.0. The general information-theoretic lower bound on H(Z) is multi-partite Wyner common information C_W(p) <= log2 rank_+(p).
Confidence: High

10.
Location: README.md (line 18)
Claim: In the summary table row "geometry aligned with that transport, to generalize to unseen z", discrete noise works "only if ordered (ordered 64-level ~ continuous; shuffled = z-independent baseline) | E4".
Truth: E4 also tests discrete per-position token codes with Hamming-OT (code, which has Hamming/factorial geometry rather than 1-D order) and obtains 0.853 / 0.837 held-out token accuracy (markov / mix16), outperforming continuous Gaussian OT (0.823 / 0.775). Saying "only if ordered" over-narrows aligned geometry to 1-D order (whereas REPORT.md lines 188 and 319 state it accurately by including Hamming-OT tokens).
Confidence: High
