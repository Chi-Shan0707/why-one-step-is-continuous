1. NUMERIC, ROUNDING, AND COUNT ERRORS

Location: REPORT.md line 164
What is claimed: "Five candidates:"
What is true: The table directly beneath (lines 166-173) lists six numbered candidates (#1 through #6), with subsections (3), (4), (5), and (6) following.
Confidence: High

Location: REPORT.md lines 218, 286 (and onestep-discrete/plots.py line 166)
What is claimed: E4 trains G on "3,277 fixed (z, x) pairs; test on 819 held-out z" (line 286: "3,277 pairs").
What is true: In onestep-discrete/sweeps/e4_coupling.py line 187, n_tr = int(FRAC_TRAIN * M) = int(0.8 * 4096) = 3276, leaving 4096 - 3276 = 820 held-out test pairs. Verified against raw E4 JSONs (e.g., markov_onehot_ot_s0.json has test_seq_acc = 0.001219512... = 1/820 and test_tok_acc = 1869 / (820 * 6)).
Confidence: High

Location: REPORT.md line 229 (E4 table, Gaussian/uniform x VAE, markov)
What is claimed: Held-out token accuracy "0.72 / 0.69" (markov / mix16).
What is true: Mean of the 3 seeds in results/e4/markov_gauss_vae_s{0,1,2}.json (0.7254, 0.7120, 0.7063) is 0.7146, which rounds to 0.71, not 0.72. (mix16 mean is 0.6925 -> 0.69.)
Confidence: High

Location: REPORT.md line 253 (E8 table, categorical K = 1024, leakage @16k)
What is claimed: Leakage at 16k for categorical K = 64 / 256 / 1024 / 4096 is "-> 0.000 / 0.003 / 0.012 / 0.008".
What is true: Mean leak_nats at step 16k across results/e8/zipf16_pop_categorical_K1024_Nenum_s{0,1,2}.json (0.00981, 0.00883, 0.01575) is 0.011464, which rounds to 0.011, not 0.012.
Confidence: High

Location: REPORT.md lines 257-258 (E8 ratio for K = 64)
What is claimed: Finite sources are "at 13x (K = 64) to more than 10^4x (K = 4096) above what their atoms could represent."
What is true: Using the unrounded JSON values, categorical K = 64 has mean kl_alloc = 0.064778 and floors["64"] = 0.0053558, giving 12.09x (12x), not 13x. The 13x comes from dividing the rounded table entries 0.065 / 0.005. (For Rademacher d = 6, K = 64, the ratio is 21.97x; for K = 4096, ratios are 2.72e4x categorical and 2.16e4x Rademacher.)
Confidence: High

Location: REPORT.md lines 121-123 (E3 population numbers above 4 bits)
What is claimed: "Structureless and well-separated atoms are best: one-hot levels 0.0004-0.010, categorical K = 256 0.003, Rademacher 0.008-0.014. Ordered dense sources are worse: quantized m = 16 0.05, Gaussian 0.055, uniform 0.045."
What is true: In results/e3/mix16_pop_*.json (3-seed means):
- Rademacher for H in [4, 16] includes d = 16 (H = 16) at 0.0202, outside 0.008-0.014 (d = 4..12 are 0.0077-0.0140; d = 32 is 0.0625).
- quantized m = 16 for H >= 4 includes d = 1 (H = 4) at 0.1492 (cited later on line 132), and d = 2, 3, 4, 8 are 0.0564, 0.0537, 0.0525, 0.0588 (two of which round to 0.06, not 0.05). Also quantized m = 256 is 0.2282 at d = 1 (H = 8) and 0.0662 at d = 2 (H = 16).
- Gaussian 0.055 and uniform 0.045 are only the best dimension (d = 4: 0.0555 and 0.0448), not stated as "best d"; across d = 1..32 Gaussian spans 0.0555-0.2223 and uniform spans 0.0448-0.2149.
- categorical K = 256 (0.0032) is only the best K; across H in [4, 16], categorical ranges up to 0.0341 (K = 16) and 0.0608 (K = 65536, worse than best-d Gaussian and uniform).
Confidence: High

Location: REPORT.md line 138 (E3 generalization n = 500, Rademacher range)
What is claimed: At equal H between 8 and 16 bits, "Rademacher 0.80-0.88."
What is true: 3-seed means for Rademacher at H = 8, 12, 16 in results/e3/mix16_500_quantized_d{8,12,16}_m2_N256_s*.json are 0.8847, 0.8051, 0.8218. 0.805121 rounds to 0.81, not 0.80 (range 0.81-0.88).
Confidence: High

Location: REPORT.md lines 116, 125-127, 290 (E3b single-seed omission and rounding)
What is claimed: Under the section heading "mean of 3 seeds" (line 116): "At 16k steps and N = 1024: ... vs Rademacher d = 4 0.0001 ... (E3b)." Limitations (line 290) notes single-seed only for E6.
What is true: Every run in results/e3b_steps16000/ and results/e3b_steps4000/ (and sweeps/e3_budget.py) uses only 1 seed (s0), not 3 seeds. Also Rademacher d = 4 at 16k steps, N = 1024 (mix16_pop_quantized_d4_m2_N1024_s0.json) has final KL 0.00005005 (5.0e-5, only 0.0001 by 4dp half-up rounding).
Confidence: High

Location: REPORT.md line 118 (E3 below 4 bits)
What is claimed: "Below 4 bits every source fails identically (1.74 / 1.22 / 0.67 nats at H = 1 / 2 / 3)."
What is true: Means are close but not identical: H = 1 has categorical_K2 = 1.7385 vs rademacher d = 1 = 1.7479 (rounds to 1.75; seed 0 is 1.7665); H = 2 has 1.2208-1.2244 for rademacher/quantized/categorical vs 1.2408 for onehot_d1_m4 (seed 1 is 1.2807); H = 3 has categorical_K8 = 0.6718 vs rademacher d = 3 = 0.6793 (rounds to 0.68).
Confidence: High


2. MC BIAS AND "EXACT" CLAIMS

Location: README.md lines 4-5, 89; REPORT.md lines 3-4, 70-71; onestep-discrete/plots.py line 132
What is claimed:
- README.md lines 4-5: "every number is exact" (contradicted by lines 88-89); line 89 (and plots.py line 132): 2^16-sample MC estimate has "(bias < 1%, checked)".
- REPORT.md lines 3-4: "every KL below is computed by enumeration" (qualified later on lines 69-70); lines 70-71 cite a single check: "0.2029 vs 0.2041 at a quarter of the samples."
What is true:
- The cited 0.2029 vs 0.2041 (0.63% difference) is a single high-KL 1D Gaussian run (mix16_pop_gaussian_d1_N256_s0.json).
- Across all 151 MC runs with kl_quarter_n_eval in results/**/*.json, 56/151 have |kl_quarter - kl| / kl > 1%, 34/151 exceed 2%, and 7/151 exceed 5%. Even after dividing by 3 for the O(1/N) Richardson bias estimate bias(N) ~= (kl(N/4) - kl(N)) / 3, 16/151 runs exceed 1% relative bias (including 7/108 in E3 alone), reaching:
  - e5/markov_pop_gaussian_d4_N256_s2.json: 0.01550 vs 0.01707 (+10.16% diff, ~3.39% implied bias)
  - e8/zipf16_pop_gaussian_d2_N1024_s0.json: 0.01711 vs 0.01839 (+7.49% diff, ~2.50% implied bias)
  - e3b_steps16000/mix16_pop_gaussian_d4_N256_s0.json: 0.02059 vs 0.02190 (+6.35% diff, ~2.12% implied bias)
  - e3/mix16_pop_onehot_d8_m16_N256_s2.json: 0.02882 vs 0.03061 (+6.21% diff, ~2.07% implied bias)
  - e3b_steps16000/mix16_pop_gaussian_d4_N1024_s0.json: 0.01160 vs 0.01221 (+5.32% diff, ~1.77% implied bias)
- Specifically for E8 Gaussian kl_alloc (REPORT.md lines 27, 255; README.md line 22: "0.0002 / 0.0003 / 0.0003"), multinomial sampling variance across 16 codewords at N_eval = 2^16 produces an expected positive KL bias of (16 - 1) / (2 * 65536) = 0.000114 nats (verified by simulation: 0.000113 +/- 0.000042). Roughly half of the reported 0.00021-0.00027 nats is at the MC sampling noise floor.
Confidence: High


3. QUALITATIVE AND INTERPRETIVE CLAIMS VS RAW JSONS / PLAN

Location: README.md line 36; REPORT.md line 88 (E1)
What is claimed:
- README.md line 36: "H1 (rank): confirmed exactly. KL = TC at k = 1 and 0 at k = r."
- REPORT.md line 88: "KL falls monotonically and reaches 0 at k = r for r = 2, 4, 16."
What is true:
- In results/e1/mix64.json, k = r = 64 does not reach 0: kl_best = 0.003281 across all 10 restarts (range 0.00328-0.01156). REPORT.md notes this on lines 88-89, while README.md omits it and says "0 at k = r".
- Strictly, kl_best is not monotone for k > r due to finite EM iterations (1000) on overparameterized mixtures: e.g., mix16 rises from 1.11e-9 (k = 16) to 1.15e-6 (k = 128), and mix64 rises from 1.39e-6 (k = 96) to 2.73e-6 (k = 128).
Confidence: High

Location: README.md line 37 (E2)
What is claimed: "H2 (steps as shared randomness): confirmed exactly."
What is true: Only the oracle (exact conditionals) hits KL = 0 at k = 6. The trained MDM on mix16 (results/e2/mix16_s{0,1,2}.json) has mean KL 0.421 vs oracle 0.364 at k = 2, 0.171 vs 0.123 at k = 3, and 0.0343 (not 0) at k = 6. (REPORT.md line 94 appropriately drops "exactly" and writes "Confirmed (E2)".)
Confidence: High

Location: REPORT.md lines 17-19; README.md lines 7-8
What is claimed: With H(Z) >= log2(rank) bits, "discrete sources fit the data as well as or better than Gaussian ones" / "Discrete noise with enough entropy fits the data as well as or better than Gaussian noise."
What is true: Only some discrete sources (well-separated atoms at moderate K) beat Gaussian. In results/e3/mix16_pop_*.json, several discrete sources with H >= 4 bits fit worse than best-d Gaussian (d = 4, 0.0555): quantized_d1_m16 (H = 4: 0.1492), quantized_d2_m4 (H = 4: 0.0689), quantized_d1_m256 (H = 8: 0.2282), quantized_d2_m256 (H = 16: 0.0662), categorical_K65536 (H = 16: 0.0608), and most H = 32 discrete sources (0.053-0.062). In E8 (zipf16), every categorical run (K = 16..4096, total KL 0.036-0.454) and Rademacher d = 4..10 (total KL 0.052-0.659) fits worse in total KL than Gaussian d = 2, 4, 8 (0.018-0.027).
Confidence: High

Location: README.md lines 41-43; REPORT.md lines 141-142 (E3 generalization, n = 500)
What is claimed:
- README.md lines 41-43: "H4 (geometry => discrete as good as Gaussian; structureless => worse): ... Supported for generalization and for learning a coupling, where unordered sources fall to chance."
- REPORT.md lines 141-142: "The worst regime is K ~= n, where each atom can memorize one training sample."
What is true:
- In one_step_discrete_plan.md (lines 42, 94), Rademacher is classified as a discrete source WITH geometry (hypercube), while categorical (random embeddings) and shuffled quantized are classified as WITHOUT geometry. In results/e3/mix16_500_*.json, H4 as stated fails for generalization too: Rademacher performs poorly (0.956 at H = 6; 0.805-0.885 at H = 8..16), just as poorly as unordered one-hot (0.769-0.989), while categorical (structureless in the plan) matches Gaussian at H = 16 (0.416 vs 0.415) and shuffled d = 2 m = 16 (0.609) matches quantized d = 2 m = 16 (0.610). Also, in E3 generalization, unordered sources do not "fall to chance" (their KL is 0.77-0.99 vs TC = 2.013); "fall to chance" only describes E4. (REPORT.md line 110 more cautiously writes "partly supported for generalization".)
- For K ~= n (n = 500): across all four source families evaluated at H = 6 (K = 64), the worst generalization KL above the rank threshold occurs at K = 64 (categorical 0.985, onehot m = 4 0.989, rademacher 0.956, quantized m = 4 0.947), not at K = 256 (closest tested K to 500, where KL is already lower: 0.863, 0.898, 0.885, 0.717), even though train_kl_to_empirical is lowest at K = 256.
Confidence: High

Location: REPORT.md lines 150-154 (E5)
What is claimed:
- "Markov ... distant pairs ~1x."
- "mix16 (all pairs equally dependent): uniform co-flipping (lift ~2 for all pairs), as it should be."
What is true:
- mix16 is a single fixed draw (seed = 0), so its 15 position pairs are not equally dependent: exact pairwise MI I(x_i; x_j) varies by 4.6x across pairs (0.0355 to 0.1633 nats).
- Co-flipping lift on mix16 (results/e5/mix16_*_flips.json) is neither uniform nor ~2 for all pairs: mix16_gaussian pair lifts span 0.76 to 5.11 across seeds (s0: 1.24-3.83, s1: 1.15-5.11, s2: 0.76-3.58; seed means 2.13-2.48), and mix16_rademacher pair lifts span 1.42 to 1.87 (seed means 1.55-1.68, closer to 1.6 than 2).
- On markov, distant pairs (dist >= 3) average 1.08 across Gaussian seeds but vary widely per seed (1.61, 0.94, 0.71, down to 0.23 at d = 5 on seed 2), and for markov_rademacher distant pairs are consistently below 1x (0.78, 0.63, 0.71).
Confidence: High

Location: REPORT.md lines 206-207 (E7)
What is claimed: "Student error is governed by I(Z;X), not by H(Z) or the kernel. The uniform and masking curves nearly coincide when plotted against I."
What is true: They are close on mix16 and markov (within ~1%-13%), but diverge substantially on copy (results/e7/copy.json). Across I in [0.4, 2.0) bits, at matched I(Z; X) the uniform kernel has 1.65x to 2.62x higher student KL than the masking kernel (e.g., at I = 0.422 bits: uniform 2.661 vs log-interpolated mask 1.58 [1.68x]; at I = 0.810 bits: 1.491 vs 0.90 [1.65x]; at I = 1.524 bits: 0.437 vs 0.26 [1.67x]; at I = 1.907 bits: 0.0774 vs 0.030 [2.62x]; at I = 1.998 bits: 0.00180 vs 0.00079 [2.30x]). On copy, even one unmasked token pins down the entire sequence with zero uncertainty, whereas uniform noise remains noisy across all positions.
Confidence: High

Location: REPORT.md lines 235-236 (E4)
What is claimed: "The plan's expected ordering holds (ODE > OT > learned VAE > random)."
What is true: one_step_discrete_plan.md lines 104-109 actually predicted "Expect: (a) ~= (d) > (c) > (b) on held-out accuracy and on distributional KL", i.e. ODE ~= VAE > Hamming-OT > random. That predicted ordering failed on both metrics:
- On held-out token accuracy (results/e4/), VAE (0.71 / 0.69) is substantially below both ODE (0.91 / 0.90) and Hamming-OT / code ot (0.85 / 0.84) as well as Gaussian OT (0.82 / 0.77).
- On distributional KL, on mix16 VAE is best (0.382) < Gaussian OT (0.500) < ODE (0.511) < gauss random (1.038) < code random (1.198) < Hamming-OT (1.423), and on markov Hamming-OT KL (0.961) is also worse than code random (0.544).
Confidence: High

Location: REPORT.md lines 39-41, 229-233, 235, 239 (E4 family omission and "chance" baseline)
What is claimed: Lines 39-41 cite "0.865 vs 0.858" and "collapses to chance (0.368)" without specifying the dataset; lines 229-239 call both 0.30/0.27 (random couplings) and 0.37/0.33 (onehot, shuffled KR) "chance".
What is true:
- 0.865, 0.858, and 0.368 are specifically for markov (for mix16 they are 0.822, 0.821, and 0.325).
- 0.368 (0.37 / 0.33) is not uniform random chance (0.25) nor random independent-sample match (sum_v p_i(v)^2 = 0.301 on markov, 0.272 on mix16, which explains gauss/code random getting 0.30/0.27 via training memorization); rather, 0.37 / 0.33 matches the marginal majority-class baseline (mean_i max_v p_i(v) = 0.379 on markov, 0.345 on mix16), i.e., the best z-independent predictor.
Confidence: High (dataset omission), Medium ("chance" terminology ambiguity)

Location: REPORT.md lines 25, 251-258; README.md lines 14, 21 (E8 "frozen"/"unchanged" and omission of K = 16 / d = 4)
What is claimed:
- Allocation error is "frozen from step 1k to step 16k" / "-> unchanged" / "never fix it" (README: "flat lines").
- Table shows only K = 64..4096 and d = 6..12, labeling the last column "best allocation possible with K atoms".
What is true:
- Allocation error is nearly flat, not strictly frozen/unchanged: from 1k to 16k in results/e8/zipf16_*.json, categorical K = 256 increases on all 3 seeds (mean 0.05463 -> 0.05680, seed 0 +5.35% from 0.07359 to 0.07753); categorical K = 4096 drops -3.32% and -4.49% on seeds 0-1 (mean 0.02801 -> 0.02729); Rademacher d = 8 drops -5.93% on seed 2 (0.05124 -> 0.04820, mean 0.05319 -> 0.05217); Rademacher d = 12 drops -3.53% on seed 1 (mean 0.02217 -> 0.02168).
- sweeps/e8_mass.py also ran categorical K = 16 and Rademacher d = 4 (both plotted in e8_mass.png), which REPORT.md omits from the table. At K = 16, categorical gets mean kl_alloc = 0.2810 (with leak = 0.1728), which is 32% BELOW floors["16"] = 0.4137, directly contradicting the table header "best allocation possible with K atoms".
Confidence: High


4. CODE AND METRIC REVIEW: exact.py AND sweeps/e8_mass.py

Location: onestep-discrete/exact.py lines 39-49 (metrics())
What is claimed: Docstring line 44 states "(both are 0 / kl for full-support p)", and the function computes kl = kl_alloc + leak_nats for arbitrary (p, log_q).
What is true:
- Docstring order error: lines 42-43 introduce kl_alloc first and leak_nats second, so for full-support p they are kl / 0, not 0 / kl.
- Unenforced normalization: tc_model normalizes q / q.sum(), whereas kl, tv, and leak_nats assume log_q is already normalized (sum(exp(log_q)) == 1) without subtracting log_q.logsumexp(0). If an unnormalized log_q is passed, leak_nats absorbs -log Z_q (and can be negative) and kl is wrong. (For log_marginal(), log_q is normalized to ~1e-15 so existing JSONs are unaffected apart from -0.0 / ~1e-15 noise on full-support p.)
- Degenerate edge case: if q(supp p) == 0 (log_in == -inf), log_q - log_in yields NaN for kl_alloc instead of handling the zero-support case cleanly.
- MC decomposition caveat: for MC-evaluated models (like E8 Gaussian), the algebraic identity holds for the finite-sample mixture q_hat, but kl_alloc(p || q_hat) has the ~1.14e-4 nats multinomial sampling bias noted above.
Confidence: High

Location: onestep-discrete/sweeps/e8_mass.py lines 3-7, 24-30; onestep-discrete/plots.py lines 215, 221-222; REPORT.md lines 251-258
What is claimed:
- e8_mass.py lines 5-6: "We compare the trained KL with the best allocation in which every atom commits to one codeword (not a strict bound: atoms that spread over codewords can leak a little)".
- e8_mass.py lines 24-30: allocation_floor(w, K) computes the optimal integer allocation greedily.
- REPORT.md lines 251-258 (and plots.py lines 215, 221-222): compare kl_alloc (not total KL) against floors[K], calling it "best allocation possible with K atoms" / "what their atoms could represent" without the one-codeword-per-atom or non-strict-bound caveats.
What is true:
- Greedy optimality of allocation_floor(w, K) IS mathematically exact for K >= len(w) and w_c > 0, and e8_mass.py's claim that it is not a strict bound IS true (verified by direct optimization on zipf16: a K = 16 uniform-weight product mixture achieves total KL = 0.2401, with kl_alloc = 0.1415 and leak = 0.0986, well below floor(16) = 0.4137).
- Problem 1 (e8_mass.py vs plots.py / REPORT.md mismatch): e8_mass.py describes comparing total KL against floor(K), whereas plots.py and REPORT.md compare kl_alloc alone against floor(K). For kl_alloc alone, floor(K) is not a bound at all: for any K >= 16 on zipf16, a K-atom product mixture can achieve kl_alloc = 0 EXACTLY (each atom c puts weight w_c / w_max on codeword c and leaks 1 - w_c / w_max onto a Hamming-distance-1 non-codeword, never colliding with another codeword since min distance is 2). In fact, categorical K = 16 in results/e8/ already has kl_alloc = 0.2810 < floor(16) = 0.4137 (plotted below the dashed curve in e8_mass.png). REPORT.md's label "best allocation possible with K atoms" is therefore false for kl_alloc and omits the "one codeword per atom" restriction even for total KL.
- Problem 2 (minor guard missing in allocation_floor): e8_mass.py lines 26-30 do not check K >= len(w); if called with K < len(w), range(K - len(w)) is empty and it silently returns a finite value with sum(n) = len(w) != K rather than raising an error or returning inf (not triggered by the script's K in [16..4096]).
Confidence: High


5. MATHEMATICAL ARGUMENTS IN REPORT.md (AND README.md)

Location: REPORT.md lines 29-36, 197-199, 208-214, 279-280; README.md lines 15, 26-28 (no deterministic flow on a finite state space; reverse Markov chain realizes forward joint => I(Z; X) = 0)
What is claimed:
- REPORT.md lines 29-36: "A continuous state space has a probability-flow ODE, a deterministic coupling with I(Z;X) = H(X). A finite state space has no such thing. A continuous path in a discrete space is constant, so gradual deterministic transport does not exist, and every diffusion-style discrete sampler must inject fresh randomness as it goes. ... The coupling of a discrete teacher with its own prior noise carries exactly I(Z;X) = 0, even when H(Z) = 12 bits."
- REPORT.md lines 197-199: "A multi-step discrete sampler driven by the exact reverse process realizes the forward joint pi(z, x) = p(x) prod_i k_alpha(z_i|x_i)."
- REPORT.md lines 279-280: "Do not distill from a discrete teacher's own prior noise. Its coupling carries I(Z;X) = 0 however many bits the noise has."
- README.md line 15: "no flow on a finite set, so the teacher coupling has I(Z;X) = 0".
What is true:
- False implication ("no ODE / random jumps => I(Z; X) = 0") and false generalization to all discrete teachers / discrete flows: Injecting fresh randomness via random jumps only forces H(X|Z) > 0, i.e., I(Z; X) < H(X); it does NOT force I(Z; X) = 0. Specifically, a Discrete Flow Matching (DFM) CTMC teacher (e.g., Campbell et al. 2024; Gat et al. 2024) starting from the 12-bit uniform prior X_0 ~ Uniform(V^L) and realizing the exact same marginal path p_t(x) = sum_{x_1} p(x_1) prod_i ((1-t)/V + t 1[x_i = x_{1,i}]) is NOT the time-reversal of the forward uniform-noising Markov chain. Reverse uniform diffusion has a 1/t rate singularity as t -> 0+ (infinite integrated jump rate, completely mixing X_0 and X_1 to I = 0), whereas the marginal DFM CTMC jump rate u_t(y_i | x_t) = (1 / (1-t)) P(X_{1,i} = y_i | X_t = x_t) cancels the (1-t) factor and stays finite at t = 0. Consequently its coupling (X_0, X_1) with its own uniform prior noise has strictly positive I(X_0; X_1) > 0. Verified numerically in this repo's toy setting: on copy (L = 6, V = 4), exact-rate DFM CTMC gives P(X_{1,1} = majority(X_0)) ~= 0.589 (vs 0.25 independent) and I(X_0; X_1) >= 0.65 bits (out of H(X) = 2 bits); on L = 2, V = 4 via exact matrix-exponential integration (TV <= 2.1e-4), DFM CTMC gives I(X_0; X_1) = 0.41 bits on copy (optimal 1-step student KL = 0.91 vs TC = 1.39) and 0.46 bits on markov (student KL = 0.24 vs TC = 0.37). Thus I(Z; X) = 0 holds specifically for the exact reverse-time Markov diffusion chain at T -> inf (alpha = 0) (or a zero-entropy mask prior), not for discrete flow teachers or discrete multi-step samplers in general.
- Missing initialization condition in the reverse-Markov-chain identity (REPORT.md lines 197-199): The exact reverse Markov chain realizes the forward joint pi(z, x) = p(x) prod_i k_alpha(z_i|x_i) ONLY when initialized from the exact finite-time forward marginal p_alpha(z) = sum_x p(x) prod_i k_alpha(z_i|x_i). For alpha > 0 (all non-zero points swept in E7), p_alpha(z) != prior (e.g., on mix16 at alpha = 0.5, masking H(Z) = 11.13 bits vs prior 0 bits; uniform H(Z) = 11.77 vs prior 12 bits), so running the reverse chain from the stationary prior does not realize pi(z, x). At alpha = 0 (T = inf), p_0 = prior and the forward kernel is independent of x, giving I(Z; X) = 0 for the exact same reason a continuous reverse SDE gives I(Z_T; X_0) -> 0 as T -> inf.
- Misstated topological obstruction (REPORT.md lines 31-32, 208-210): Continuity of curves [0, 1] -> X rules out continuous trajectories in X, not deterministic jumps (piecewise-constant paths or discrete-time maps X -> X can change marginals deterministically). The actual obstruction to gradual deterministic transport matching a smooth marginal curve t -> p_t on a finite set is that deterministic maps X -> X cannot split an atom's mass, and because only finitely many such maps exist, pushforward probabilities are quantized to finite sums of initial atom weights and cannot vary continuously in t.
- Continuous-state ODE entropy imprecision (REPORT.md line 30): "A continuous state space has a probability-flow ODE, a deterministic coupling with I(Z;X) = H(X)." For a continuous target X, a deterministic bijection has I(Z; X) = +inf (not finite differential entropy); I(Z; X) = H(X) holds only when X is discrete (e.g., obtained by deterministic argmax/decoding of the continuous ODE endpoint).
Confidence: High

Location: REPORT.md lines 23-27, 261-264; README.md line 14; onestep-discrete/sweeps/e8_mass.py lines 3-5 (mass derivative argument)
What is claimed:
- REPORT.md lines 23-25: "With an atomic source, the probability a generator puts on a mode is (#atoms mapped there)/K. That is piecewise constant in the parameters, so gradient descent cannot move mass between modes."
- REPORT.md lines 261-264: "Mechanism: the mass on a mode is P(z in R_c(theta)). For an atomless P_Z its derivative is the flux of source density through the moving boundary dR_c, which is generically non-zero. For an atomic P_Z it is a sum of indicators, which is piecewise constant."
- e8_mass.py lines 3-5: "A finite source with K equal atoms can only give each codeword a multiple of 1/K, and that allocation is piecewise constant in the parameters (zero gradient)."
What is true:
- For the actual smooth SiLU+softmax MLP generator trained in E8, p_theta(c) = (1/K) sum_k prod_i p_theta(c_i | z_k) is C^inf-smooth and differentiable in theta for both atomic and atomless sources, and is NOT restricted to multiples of 1/K nor piecewise constant with zero gradient. (Indeed, in E8 itself, categorical K = 16 puts non-multiples of 1/16 on codewords to achieve kl_alloc = 0.281 < floor(16) = 0.414, and categorical K = 256..4096 and Rademacher d = 8..12 still move kl_alloc by 1%-6% and cut leak_nats significantly from 1k to 16k.)
- The indicator-sum / piecewise-constant formula P_Z(z in R_c(theta)) applies only in the zero-temperature / hard-partition (zero-leakage) limit, not to the actual objective. In the actual smooth generator, atomic allocations freeze not because p_theta(c) has literally zero derivative everywhere, but because once an atom saturates near a codeword c_1, (i) softmax gradients vanish exponentially and for any codeword c_2 at Hamming distance >= 2 the first derivative of prod_i p_theta(c_{2,i} | z_k) w.r.t. per-position probabilities vanishes at c_1, and (ii) moving a discrete 1/K atom between distance->=2 codewords requires crossing intermediate product distributions that leak onto non-codewords (a non-convex loss barrier, which line 264 notes, contradicting the literal zero-gradient piecewise-constant claim).
- Minor regularity omission in the hard-partition boundary-flux derivative (line 261-262): an atomless P_Z alone does not guarantee a density flux derivative (e.g., singular continuous measures have no density); the Reynolds-transport boundary-flux formula requires a density p_Z with respect to Lebesgue measure and sufficiently smooth moving boundaries dR_c(theta).
Confidence: High
