# BENCHMARKS — every number iobrx ships with, and where each one came from

> Evidence status (2026-09-09): R3–R6 results are campaign-reported measurements. Some cited raw scripts/logs remain outside this repository. They are retained for provenance, not presented as independently rerun release evidence. Current local tests and tutorial timings: [0.3 validation](docs/VALIDATION_0.3.md).

Two measurement eras share one precision standard: **bit-exact vs the frozen
ORIGINAL (IOBRpy 0.2.0) references** — identical index/columns/dtypes/NaN
mask, every numeric cell `max_abs_diff == 0.0`, or byte-/sha256-equal output
files, with individually documented nondeterminism exemptions.

- **Part I — the acceleration stack (iobrx 0.1.0–0.2.0, routes R0–R5).**
  The official-gate numbers for the eight pre-port modules: `cibersort`,
  `calculate_sig_score` (sig_pca / sig_zscore / sig_ssgsea / sig_integration),
  `count2tpm`, `epic`, `estimate_score`, `mcpcounter`, `quantiseq`,
  `anno_eset`. Every artifact cited there ships in [`bench/`](bench/)
  (index: [`bench/README.md`](bench/README.md)).
- **Part II — the port campaign (iobrx 0.3.0, rounds R1–R6).** The 19 IOBRpy
  workflows ported on top of that stack, ending with the R6 formal blind
  benchmark under one unified protocol. Part II sources are campaign research
  artifacts cited as `research/<dir>/<file>` under the campaign root
  `/data/users/lianchong/workspace1/outputs/turn_20260908095156_c4fc9df8a2f541f086dce127383f342a/research/`
  (not bundled in this repository; the repo's own CI re-verifies the parity
  contracts via `tests/test_parity_*.py`).

Route labels **R0–R5 in Part I** and round labels **R1–R6 in Part II** are
two different numbering series; they are not the same timeline.

---

# Part I — the acceleration stack (pre-port modules, routes R0–R5)

This part states the measured performance and precision of every acceleration
route explored for **iobrx 0.1.0**, ending with the route this repository
ships (**R5, `frozen_stack_v1`**). Every number carries its source in
parentheses; all artifact filenames refer to the verbatim copies in
[`bench/`](bench/) (indexed in [`bench/README.md`](bench/README.md)). Numbers
marked *computed* are arithmetic over two cited cells (e.g. a ratio), not new
measurements.

## I.1 Environment & protocol

- **Machine:** Intel Xeon Platinum 8480C, 224 logical cores, ~503 GiB RAM
  (FINAL_SUMMARY.json `environment`). Shared node: background loadavg 5–27
  was recorded with every timing block (FINAL_SUMMARY.json `provenance.timing_protocol`).
- **Software:** Python 3.11.15, numpy 2.2.6, pandas 2.3.3, scipy 1.16.3,
  scikit-learn 1.7.2, gseapy 1.3.0 (FINAL_SUMMARY.json `environment`);
  rustc/cargo 1.97.1, maturin 1.15.0 (scaling_coldstart.json `machine`);
  the shipped Rust extension is the 0.1.0 wheel built 2026-09-07 22:40,
  sha256 `c3b4740b…dafbd` (FINAL_SUMMARY.json `environment.iobrx_rust_sha`).
- **Baseline ("ORIGINAL"):** unmodified IOBRpy 0.2.0, commit `e3808db`
  (FINAL_SUMMARY.json `environment.iobrpy_commit`). PyPI 0.2.1 is numerically
  identical for all ported code paths — ported modules byte-identical,
  cibersort weights/Corr/RMSE bit-exact vs the frozen refs in a fresh
  subprocess (pypi_021_diff.json `port_modules_verdict`, `numeric_confirmation`),
  so no re-baselining was needed.
- **Official workflow (what "total" means):** 11 stages on the official
  data — sig_pca / sig_zscore / sig_ssgsea / sig_integration on
  imvigor210_eset (872×348) × signature_collection; anno_eset
  (eset_stad 60483×10 Ensembl × anno_grch38, `mean`); cibersort
  (LM22, perm=100, QN=True, on eset_stad_symbol 50181×10, original via
  n_jobs=1 temp-CSV); epic (TRef), quantiseq (TIL10/lsei), mcpcounter (HUGO),
  estimate (affy); count2tpm on raw Ensembl eset_stad
  (FINAL_SUMMARY.json `provenance.baseline_definition`).
- **Timing protocol:** fresh subprocess per repetition, data/resource loading
  untimed, stages timed individually with `perf_counter` (with the same
  per-stage input `.copy()` inside the timed region as the baseline harness);
  **1 warmup + 3 timed reps, medians reported**; loadavg recorded per block
  (FINAL_SUMMARY.json `provenance.timing_protocol`, `e2e.protocol`). The
  warm loop is the one exception (one subprocess, 1 warmup + 3 timed passes;
  FINAL_SUMMARY.json `e2e.warm.protocol`). Replicate spread: the six
  fast/steady-state stage legs sit at CV 0.0–1.7% (median 0.6%; sample sd
  over the fast-side `times_s` rep lists in count2tpm_fast_results.json
  `timing.fast`, quantiseq_fast_results.json `timing.wrapper_cached`,
  misc_fast_results.json `anno_eset/epic/mcpcounter/estimate.timing_fast`);
  over ALL thirteen timing rep lists in those three files — adding the six
  ORIGINAL-side lists and quantiseq's `wrapper_cold_pair` steady-state list
  — the range widens to 0.0–28.5% (median 0.8%; dominated by the ORIGINAL
  quantiseq `original_same_process` list still decaying 1.788→1.060 s at
  28.5%, its cold-pair steady list at 5.0% and the ORIGINAL count2tpm list
  at 4.3%), while fresh-subprocess e2e totals on the shared node run at CV
  2.8–4.4% (computed from FINAL_SUMMARY.json
  `e2e.t224/t64/t32.rep_totals_s`) — which is why medians of 3 and loadavg
  logging are used everywhere.
- **Speedup denominator:** the campaign-ledger baseline total of **97.13 s**
  for the ORIGINAL 11-stage workflow (literature_anchors.json
  `internal_reference_point`), i.e. round 1 = 1.0x
  (FINAL_SUMMARY.json `history.per_round_best.round_1`). Stage-level
  ORIGINAL medians below are quoted from the per-stage artifacts, each
  measured in its own run.

## I.2 Route table — what was measured, route by route

| Route | What it is | Official workflow total | Speedup | Precision verdict | Notes (source) |
| --- | --- | --- | --- | --- | --- |
| **R0** baseline | IOBRpy 0.2.0 pure Python, unmodified | **97.13 s** | **1.0x** (reference) | reference | ORIGINAL n_jobs=1 cibersort alone: 87.17 s DataFrame-protocol (rust_cibersort_results.json `original_njobs1_baseline`), 88.82 s setup-facts / 94.9 s temp-CSV-protocol medians (scaling_coldstart.json `reference_baselines_original_iobrpy`), 95.172 s fresh control single run (bench_parallel_results.json `timing.control`); count2tpm median 4.441 s (baseline_timing.json) |
| **R1** parallel-only | ORIGINAL code, joblib `Parallel(n_jobs, prefer='threads')` inside cibersort | 15.1 s *(computed: 97.13/6.42)* | **6.42x** best (@224 threads) (FINAL_SUMMARY.json `history.per_round_best.round_2`) | weights/Corr/RMSE **bit-identical** to the n_jobs=1 reference at every n_jobs incl. the fresh control; only the unseeded P-value column differs (bench_parallel_results.json `identity`) | cibersort stage medians 11.562 s @16T / 8.442 s @64T / 6.864 s @224T = 8.23x / 11.27x / 13.87x stage (bench_parallel_results.json `derived_analysis`); dispatch overhead negligible — 330 no-op tasks median 0.0284–0.0606 s @16–224T (bench_parallel_results.json `dispatch_benchmark`) — but the GIL + coarse task granularity cap it: perm-stage parallel efficiency 0.704 @16T → 0.217 @64T → 0.082 @224T (bench_parallel_results.json `derived_analysis`); joblib threads actively *hurt* the SVD-based sig stages: pca 0.2271 s serial → 0.3315 s @8T → 0.4064 s @32T (sig_compose_results.json `timings`, `notes`) |
| **R2** vectorized-Python stages alone | numpy/pandas rewrites of the non-solver stages, no Rust | stage-level only (no full-stack route was frozen at this stage) | count2tpm **38.7x** stage; quantiseq **16.9x** stage (below) | bit-exact on every rewritten stage (count2tpm_fast_results.json `precision`; quantiseq_fast_results.json `precision`; misc_fast_results.json per-stage gates) | count2tpm 4.680 s → 0.1209 s (count2tpm_fast_results.json `timing`); quantiseq 1.2037 s → 0.0711 s cached (quantiseq_fast_results.json `timing`); smaller wins: anno_eset 3.77x, epic 1.58x, mcpcounter 4.24x, estimate 2.53x (misc_fast_results.json); sig stages then: pca 1.17x, integration 1.05x, zscore 0.92x (sig_compose_results.json `timings`). Combined parallel+vectorized campaign rounds reached 10.56x / 10.74x (FINAL_SUMMARY.json `history.per_round_best.round_3/round_4`) |
| **R3** Rust bit-faithful cibersort | vendored sklearn-1.7.2 libsvm `svm.cpp` + rayon, byte-exact solver semantics | cibersort stage route | **34.3x stage** @224T vs ORIGINAL n_jobs=1 median 88.82 s (2.5866 s; scaling_coldstart.json `1_thread_scaling`), **38.5x** vs 87.17 s DataFrame protocol (2.262 s in-process; rust_cibersort_results.json) | weights/Corr/RMSE **bit-identical** incl. csv-parse semantics: 240/240 non-P-value cells vs the frozen official reference (pool_hoist_results.json `gates.gate1`); outputs **thread-count invariant** 1/64/224T, n_iter_sum 84,984,516 identical (pool_hoist_results.json `gates.gate2`) | single-thread Rust already matches the original (87.42 s vs 87.17 s; rust_cibersort_results.json); phase split @224T: perm stage 0.8112 s of 2.3117 s core time (rust_cibersort_results.json `phase_breakdown_nthreads_224`) |
| **R4** full fast stack, stepwise | Rust cibersort + Rust ssGSEA/integration legs + vectorized stages + cold builders + pool hoist + Rust PCA/z-score | evolution across rounds | **18.0x → 21.4x → 22.1x → 23.0x → 25.9x** (rounds 5–9; FINAL_SUMMARY.json `history.per_round_best`) | every added leg gated bit-exact against the frozen refs before it entered the stack (rust_ssgsea_results.json `gate_rust_vs_ref`; integration_rust_results.json `gate_fast_rust_vs_ref`; cold_builders_results.json `gate_*`; final_round9_results.json `part1_gate_A/B`, `part2_gates`) | see the step table in §I.3 |
| **R5** FINAL = `frozen_stack_v1` **(shipped)** | the R4 endpoint, frozen and re-validated end-to-end | **3.75–4.13 s** fresh-process depending on node load: 3.7515 s (25.89x, final_round9_results.json `e2e.t224`) → 3.8736 s after the audit fixes (25.07x, full_e2e_results.json `t224`) → 4.1265 s under freeze-day load 8–33 (23.54x, FINAL_SUMMARY.json `e2e.t224`); **warm 3.63–3.80 s** (cold_builders_results.json `warm_inproc_e2e`; FINAL_SUMMARY.json `e2e.warm`) | **official gates: 10/11 stages max abs diff 0.0, every numeric cell bit-identical; cibersort 0.0 on all 240 non-P-value cells** (FINAL_SUMMARY.json `official_gates`) | t64 4.8153 s (20.17x), t32 5.8909 s (16.49x) (FINAL_SUMMARY.json `e2e.t64/t32`; ratios computed); per-stage table in §I.4 |
| **Held-out control (BLCA)** | R5 stack vs ORIGINAL on TCGA-BLCA data never used in any optimization round (eset_blca 60483×5) | **55.989 s → 4.899 s** | **11.43x** total | **all 11 stages max abs diff 0.0** (heldout_control_results.json `stages.*.compare.verify`, `totals`; FINAL_SUMMARY.json `heldout_blca`) | stage highlights: cibersort **53.56x** (45.752 s → 0.8542 s), count2tpm **48.84x** (4.0726 s → 0.0834 s), quantiseq 16.47x, mcpcounter 4.97x, anno_eset 3.52x, estimate 3.43x, ssGSEA 2.75x, integration 1.40x, pca 1.32x, epic 1.81x, zscore 0.99x (heldout_control_results.json `stages`); imvigor cibersort subset (100 samples): 7.1x, 2400/2400 cells bit-identical excl P-value (heldout_control_results.json `imvigor_cibersort`) |

## I.3 Route R4 — what each step added

| Round (speedup; FINAL_SUMMARY.json `history.per_round_best`) | Step added | Measured effect (source) |
| --- | --- | --- |
| round 5 (**18.01x**) | Rust cibersort (R3) + the R2 vectorized stages wired as one full stack | e2e t224 median 5.321 s (cold_builders_results.json `summary.previous_run.median_total_s_t224`); mid-campaign snapshot 5.4 s (literature_anchors.json `internal_reference_point`) |
| round 6 (**21.35x**) | Rust ssGSEA core + integration leg | ssGSEA stage 0.6575 s → 0.0571 s = **11.5x** @224T, 10,788/10,788 cells bit-identical (rust_ssgsea_results.json `speedup_vs_original_t1`, `gate_rust_vs_ref`); integration stage 0.9438 s → 0.3239 s, 61,596/61,596 cells (integration_rust_results.json `original_stage_s`, `timing_integration_fast_rust`, `gate_fast_rust_vs_ref`) |
| round 7 (**22.14x**) | cold builders: vectorized HGNC alias-map build + disk-cached packaged count2tpm refs | alias map 0.8531 s → 0.0654 s (**13.0x** build), dict `==`-equal with identical insertion order (cold_builders_results.json `gate_quantiseq.alias_map`); quantiseq stage cold 0.9025–0.9783 s → 0.0571 s warm (cold_builders_results.json `timing_quantiseq`); count2tpm cold-with-disk-cache 0.1623–0.2460 s vs 0.3104–0.5158 s first-ever build (cold_builders_results.json `timing_count2tpm`); e2e t224 5.321 → 4.3241 s (cold_builders_results.json `summary`) |
| round 8 (**23.03x**) | pool hoist: per-call rayon pools → process-global lazily-initialized pool registry shared by cibersort_core and ssgsea_core | steady-state cibersort 224T 2.5866 → 2.434 s, 64T 3.4721 → 3.3408 s (pool_hoist_results.json `before_after`); one-time 224-thread pool build cost ~0.147 s/call now paid once per process (pool_hoist_results.json `pool_build_cost_estimate_224T`); thread-invariance gates re-passed bitwise (pool_hoist_results.json `gates.gate2`) |
| round 9 (**25.89x**) | Rust PCA / z-score legs (`pca_pc1`, `pca_zscore_intermediate`, `rows_colmean_pairwise`, `dgesdd` via the installed scipy's bundled OpenBLAS) | bit-exact on **all 58** official signatures: z intermediate, PC1 and mean_expr each max abs diff 0.0 (final_round9_results.json `part1_gate_A`); full-stage gates re-passed (final_round9_results.json `part1_gate_B`, `part2_gates`); quiet-load stage medians pca 0.1982 s / zscore 0.1158 s / integration 0.2479 s (final_round9_results.json `stage_benchmarks.groups`); **e2e t224 3.7515 s = 25.89x** (final_round9_results.json `e2e.t224`; ratio computed) |

Two later hardening rounds did not change the route, only its fidelity, and
are part of R5: the five adversarial-audit fixes (fix_round11_results.json)
and the csv-parse roundtrip port for cibersort (forensic_blca_results.json).

## I.4 Final stack (`frozen_stack_v1`) — per-stage table, official workflow

ORIGINAL/fast/precision columns are all from artifacts; the speedup column
is *computed* (ORIGINAL ÷ fast) from the two cited cells.

| Stage | ORIGINAL median s (source) | fast median s, freeze e2e t224 (source) | Speedup | Precision (FINAL_SUMMARY.json `official_gates.stages`) | Controlled same-process comparison (source) |
| --- | --- | --- | --- | --- | --- |
| sig_pca | 0.2659 (sig_compose_results.json `timings.orig_pca`) | 0.2600 (FINAL_SUMMARY.json `e2e.t224.per_stage`) | 1.02x | 0.0; 20,532/20,532 cells | 0.2271 s stage, bitwise identical at 1/8/32T (sig_compose_results.json) |
| sig_zscore | 0.1854 (sig_compose_results.json `timings.orig_zscore`) | 0.1473 (FINAL_SUMMARY.json `e2e.t224.per_stage`) | 1.26x | 0.0; 20,532/20,532 cells | Rust leg bit-exact all 58 sigs (final_round9_results.json `part2_gates`) |
| sig_ssgsea | 1.0054 (sig_compose_results.json `timings.orig_ssgsea`) | 0.1532 (FINAL_SUMMARY.json `e2e.t224.per_stage`) | 6.56x | 0.0; 10,788/10,788 cells | 11.5x stage (rust_ssgsea_results.json) |
| sig_integration | 0.9636 (sig_compose_results.json `timings.orig_integration`) | 0.2542 (FINAL_SUMMARY.json `e2e.t224.per_stage`) | 3.79x | 0.0; 61,596/61,596 cells | 0.9438 → 0.3239 s (integration_rust_results.json) |
| anno_eset | 0.0894 (misc_fast_results.json `anno_eset.timing_original`) | 0.1092 (FINAL_SUMMARY.json `e2e.t224.per_stage`) | 0.82x | 0.0; 501,810/501,810 cells | 3.77x in-process (misc_fast_results.json `anno_eset`) |
| cibersort | 94.9 (scaling_coldstart.json `reference_baselines_original_iobrpy.csv_protocol_3rep_median_s`) | 2.7855 (FINAL_SUMMARY.json `e2e.t224.per_stage`) | 34.1x | 0.0 excl P-value; 240/240 non-P cells (P-value column unseeded upstream by design) | 53.56x on held-out BLCA (heldout_control_results.json) |
| epic | 0.0164 (misc_fast_results.json `epic.timing_original`) | 0.0272 (FINAL_SUMMARY.json `e2e.t224.per_stage`) | 0.60x | 0.0; 270/270 cells (epic.pkl: 3 frames) | 1.58x in-process (misc_fast_results.json `epic`) |
| quantiseq | 1.2037 (quantiseq_fast_results.json `timing.original_same_process`) | 0.1617 (FINAL_SUMMARY.json `e2e.t224.per_stage`) | 7.44x | 0.0; 110/110 cells | 16.9x stage cached (quantiseq_fast_results.json) |
| mcpcounter | 0.0072 (misc_fast_results.json `mcpcounter.timing_original`) | 0.0082 (FINAL_SUMMARY.json `e2e.t224.per_stage`) | 0.88x | 0.0; 100/100 cells | 4.24x in-process (misc_fast_results.json `mcpcounter`) |
| estimate | 0.0437 (misc_fast_results.json `estimate.timing_original`) | 0.0485 (FINAL_SUMMARY.json `e2e.t224.per_stage`) | 0.90x | 0.0; 30/30 cells | 2.53x in-process (misc_fast_results.json `estimate`) |
| count2tpm | 4.441 (baseline_timing.json) | 0.1544 (FINAL_SUMMARY.json `e2e.t224.per_stage`) | 28.8x | 0.0; 480,580/480,580 cells | 38.7x stage (count2tpm_fast_results.json) |
| **total** | **97.13 s campaign baseline** (literature_anchors.json `internal_reference_point`) | **4.1265 s** median-of-rep-totals (FINAL_SUMMARY.json `e2e.t224.median_total_s`; sum of the stage medians above is 4.109 s, computed) | **23.5x** (computed) | 10/11 stages max abs diff 0.0 + cibersort 0.0 excl P-value | 25.89x at quiet load (final_round9_results.json `e2e.t224`) |

Reading notes: the freeze e2e per-stage medians were taken in fresh
subprocesses on a node with background load 8.3–32.9 (FINAL_SUMMARY.json
`e2e.protocol`, `e2e.driver_loadavg` — driver start 8.28–8.77, per-block
end peaks up to 32.88), so the four sub-30 ms stages (anno_eset, epic,
mcpcounter, estimate) show e2e ratios at-or-below 1 that are lower bounds —
their controlled same-process comparisons (last column) are the meaningful
stage numbers. The ORIGINAL per-stage medians were each measured in their own
runs; their sum (~103 s, computed) brackets the 97.13 s campaign baseline
within the run-to-run load noise documented above.

## I.5 CIBERSORT thread scaling (shipped stack, fresh subprocess per point)

LM22, perm=100, QN=True, eset_stad_symbol 50181×10, DataFrame-in
(FINAL_SUMMARY.json `cibersort_scaling`):

| n_threads | median s | speedup vs 1T | vs frozen ref (excl unseeded P-value) |
| --- | --- | --- | --- |
| 1 | 95.1732 | 1.0x | 0.0, 240/240 cells bit-identical |
| 8 | 12.9618 | 7.34x | 0.0, 240/240 |
| 16 | 7.4034 | 12.86x | 0.0, 240/240 |
| 32 | 4.5971 | 20.7x | 0.0, 240/240 |
| 64 | 3.3374 | 28.52x | 0.0, 240/240 |
| 128 | 2.9056 | 32.76x | 0.0, 240/240 |
| 224 | 3.0601 | 31.1x | 0.0, 240/240 |

All rows: FINAL_SUMMARY.json `cibersort_scaling.points`. Outputs are
bit-identical at every thread count — the seeded permutations are
thread-count invariant (pool_hoist_results.json `gates.gate2` re-verified
1/64/224 bitwise). 224T runs on logical (HT) cores and is slightly *slower*
than 128T on this 10-sample stage (FINAL_SUMMARY.json
`cibersort_scaling.note`). The pre-fix crate showed the same bend:
13.3224 s @8T → 2.5866 s @224T (scaling_coldstart.json
`1_thread_scaling.results`). Default in iobrx is `n_threads=None` →
`min(8, cores)`; at 8T the official stage is still 7.34x vs 1T.

## I.6 Cold start, memory, and build

- **Cold process to first CIBERSORT result: ~3.1 s** total subprocess wall —
  3.316 / 3.069 / 3.119 s over 3 fresh runs; first call 2.9175 / 2.6701 /
  2.7207 s; importing the compiled module costs ~0.0004 s, numpy+pandas
  ~0.26 s (scaling_coldstart.json `2_cold_process_start`).
- **Memory:** the cibersort call itself adds **+277 MiB VmRSS at 224T**
  (283,520 KiB ≈ 1.24 MiB/thread of faulted-in rayon stacks + per-fit
  buffers) vs **+9.5 MiB VmHWM at 1T** (9,768 KiB); process peak RSS
  (~534 MiB, 547,020 KiB) is set by the numpy/pandas/pyarrow imports and is
  not raised by the call at any thread count (scaling_coldstart.json
  `5_memory_peak_rss`). One-time 224-thread pool build: ~0.147 s per process
  (pool_hoist_results.json `pool_build_cost_estimate_224T`).
- **Wheel & build:** the 0.1.0 extension wheel is 501,213 bytes
  (pool_hoist_results.json `wheel.bytes`); incremental rebuild 9.74–9.97 s,
  full build after `cargo clean` 13.71 s on 224 cores
  (scaling_coldstart.json `4_wheel_build`; pool_hoist_results.json
  `wheel.build_time_s`); a fresh-venv source install including all
  dependencies took 124.4 s (.build_report_round1.json
  `verification.fresh_env`).

## I.7 Precision methodology

**What "bit-exact" means here.** Identical index (incl. name), columns,
dtypes and NaN mask, and max abs diff exactly 0.0 — every numeric cell
bit-identical — versus the frozen ORIGINAL reference outputs
(audit_precision.json `metadata.protocol`; FINAL_SUMMARY.json
`official_gates`). The single excepted column is CIBERSORT's P-value, whose
ORIGINAL permutations are seeded from OS entropy (`SeedSequence()` with no
seed), so *the original itself* is not reproducible run-to-run on that
column; the formula (`count(null_r >= r)/perm`, granularity 1/perm) is
identical and iobrx's seeded P-values are stable and thread-count invariant
(FINAL_SUMMARY.json `official_gates.cibersort_pvalue_column`;
bench_parallel_results.json `identity.identity_verdict`; pypi_021_diff.json
`numeric_confirmation.cibersort`).

**The 83-case adversarial audit.** Fresh subprocesses per side, original vs
fast, on adversarial inputs (NaN rows, duplicated ids, degenerate overlaps,
non-unique indices, log-transformed mixtures, ...). History:
76 PASS + 3 FAIL + 1 ERROR_MISMATCH + 2 BOTH_ERROR_SAME_TYPE +
1 ORIG_ERROR_FAST_OK pre-fix (audit_precision.prefix11.bak.json
`metadata`) → **79 PASS + 3 BOTH_ERROR_SAME_TYPE + 1 ORIG_ERROR_FAST_OK**
after the five fixes (audit_precision.json `metadata`;
fix_round11_results.json `regression.a_audit_battery`) — the five findings
were: a count2tpm keep-mask bug (CT7), an estimate first-vs-last-occurrence
gene-position map (ES5), a 1-ulp Rust-libm `exp2` plus a numpy reduction-order
mismatch in cibersort's X standardization (CB6b), a Rust panic on NaN input
(CB11), and an F-contiguous misread in the exposed QN helper (F5;
`findings.FINDING_5_quantile_normalize_np`, cross-listed in `crate_changes`)
(fix_round11_results.json). The **final freeze rerun** measured
77 PASS + 2 FAIL + 3 BOTH_ERROR_SAME_TYPE + 1 ORIG_ERROR_FAST_OK
(FINAL_SUMMARY.json `audit.verdict_counts`).

**The 2 remaining FAILs are contract artifacts, not solver divergence.**
The audit feeds the FAST side a DataFrame already parsed via
`%.17g` CSV; the shipped fast path additionally applies the OFFICIAL input
semantics (`to_csv` default float format → python-engine reparse), which is
not idempotent on such pre-parsed inputs — 7218/501,810 cells drift 1 ulp
(163 within the 531 LM22-overlap genes), and under QN=False that flips NuSVR
support sets (CB2: 8.61e-3 on 13/250 cells; CB12: 5.05e-1 on 14/260). Under
QN=True — the default and only officially validated mode — outputs stay
bit-identical. Attribution proof: same binary with the roundtrip omitted
scores CB2 250/250 and CB12 260/260 cells bit-identical, while the official
gates (DataFrame contract) and held-out BLCA are 0.0 WITH the roundtrip;
both contracts cannot be satisfied by one code path and the freeze ships the
official one (FINAL_SUMMARY.json `audit.deviation_from_expectation`,
`audit.attribution_proof`).

**Why bit-exact instead of a faster approximate solver.** Varying only the
NuSVR stopping tolerance on the official 531×22 problem: tol 1e-3 vs 1e-5
moves cell-type **fractions by up to 0.0793 in absolute terms (~8 percentage
points**; 127/220 entries differ at rel > 1e-3; raw weights up to 0.0110
abs) (tolerance_results.json `comparisons.fractions_1e-3_vs_1e-05`,
`w_raw_1e-3_vs_1e-05`). Any faster-but-different solver would silently
change results at scientifically meaningful magnitude, so the stack vendors
scikit-learn 1.7.2's `svm.cpp` byte-identical and keeps its semantics.

**The csv-parse finding.** The ORIGINAL cibersort is file-based: it
round-trips your DataFrame through `to_csv` and re-parses with pandas'
python engine, whose `precise_xstrtod` float conversion is *not* the
identity — 1–2 ulp off on ~12% of values (BLCA: 23,930/205,050 cells)
(forensic_blca_results.json `root_cause`). Bit-exactness therefore required
porting `precise_xstrtod` (plus CPython `repr` token rules, including exact
halfway-tie behavior) into `csv_parse_roundtrip`, verified equal to the real
pandas parse bitwise on six datasets — BLCA 205,050/205,050, STAD
501,810/501,810, CB6b 11,750/11,750, CB1 501,810/501,810, imvigor
303,456/303,456, synthetic 130,042/130,042; repr-token parity on
6,198,527/6,198,527 random values — at a steady-state cost of ~9.6 ms per
502k values (median 9.58 ms @224T; forensic_blca_results.json `fix_validation`,
`gates.GATE3_perf.mapping_steady_state_224T.median_ms`).
Known inherited caveat: the parse is not idempotent (pandas' own is not
either), and this is exactly the audit-contract effect above.

**Last-bit fixes that made the Rust core bit-faithful.** numpy's vectorized
`np.exp2` differs from Rust libm `f64::exp2` by 1 ulp on ~5% of entries
(594/11,750 cells measured), and numpy reduces a full-array mean over flat
F-memory order with 8192-element chunked pairwise sums — both were ported
exactly (fix_round11_results.json `findings.FINDING_3_CB6b`).

**Parity pins.** `numpy<2.3` and `scikit-learn<1.8`: upstream numeric drift
breaks bit-parity from the ORIGINAL side — sklearn 1.9.0 moved the
original's NuSVR up to 5.7e-3 on 5/10 official samples while iobrx stayed
240/240 vs the frozen refs, and numpy 2.4 changed reduction rounding by 1 ulp
(a signature-wide `X.mean()`), flipping NuSVR support sets (nSV 417 vs 420 at
nu=0.75) (.build_report_round1.json `blockers` 3–4). The other ten gates pass
bit-exact on numpy 2.2–2.4 and scipy 1.16–1.17 (.build_report_round1.json
`notes`).

## I.8 Literature anchors (claim-class context, not arithmetic comparisons)

From literature_anchors.json (`anchors`, with DOIs and verbatim quotes
there). **None of these numbers is arithmetically comparable to ours** —
every anchor differs in cohort, timing scope, hardware or workload
(literature_anchors.json `recommended_primary_anchor.record_protocol_note`);
they establish claim class only.

| ID | Source (DOI) | Quoted number | Why it is cited (literature_anchors.json) |
| --- | --- | --- | --- |
| A1 | IOBRpy preprint, bioRxiv 2026-07-22 (10.64898/2026.07.17.739055) | CIBERSORT 2,864.8 s @1 thread → 209.7 s @16 on a 936-sample cohort (13.7x thread scaling) | the only published runtime inside IOBRpy itself; in-Python thread scaling, explicitly not Python-vs-R (`anchors.A1`) |
| A2 | Newman 2015, Nat Methods (10.1038/nmeth.3337) | — (no runtime statements) | honest negative: the CIBERSORT paper publishes no timings (`anchors.A2`) |
| A3 | Newman 2019, Nat Biotechnol (10.1038/s41587-019-0114-2) | — (no runtime statements) | honest negative for CIBERSORTx (`anchors.A3`) |
| A4 | IOBR R papers (10.3389/fimmu.2021.687975; 10.1016/j.crmeth.2024.100910; 10.1002/mdr2.70001) | — (no runtime statements) | honest negative: no R-side baseline exists from the authors (`anchors.A4`) |
| A5 | Diaz-Mejia 2019, F1000Research (10.12688/f1000research.18490.3) | CIBERSORT 75–9,330 s (continuous) / 46–1,522 s (binary) on a 2-core i5 | strongest external absolute-runtime anchor for the nu-SVR cost (`anchors.A5`) |
| A6 | GSEApy paper, Bioinformatics 2023 (10.1093/bioinformatics/btac757) | Rust rewrite 3x (small library) to 80x (large) vs their own Numpy version; 50 GB → 1.4 GB | closest peer-reviewed precedent for Python→Rust with comparable statistic; brackets our workflow speedup (`anchors.A6`) |
| A7 | gseapy repo/paper (no DOI) | — (no ssGSEA-specific timing exists) | honest negative for the ssGSEA stage specifically (`anchors.A7`) |
| A8 | scikit-learn issue #11079 (no DOI) | NuSVR(linear) on 547×22 > 3 h without returning | public evidence the exact problem shape is pathologically slow in sklearn (`anchors.A8`) |
| A9 | scikit-learn SVM docs (no DOI) | libsvm QP solver scales O(n_f·n_s²)–O(n_f·n_s³); "LinearSVC … much more efficient" | authoritative complexity anchor; NuSVR has no liblinear counterpart (`anchors.A9`) |
| A10 | ThunderSVM, JMLR 19:797-801 (2018; no DOI) | >10x CPU / >100x GPU vs LibSVM "while producing identical SVMs"; SVR row 9,161 s → 691 s (CPU) | peer-reviewed claim class: native/parallel reimplementation of the same solver, identical outputs (`anchors.A10`) |
| A11 | PyPI iobrpy (no DOI) | latest 0.2.1, uploaded 2026-08-17 | pins the moving target; 0.2.1 numerically identical (see pypi_021_diff.json) (`anchors.A11`) |
| A12 | crates.io + GitHub (no DOI) | 0 crates named "cibersort"; libsvm-rs 335 downloads; python-cibersort-rs exists | novelty check and related work; do not claim first-ever Rust CIBERSORT unqualified (`anchors.A12`) |

## I.9 Verdict (Part I)

**This repository ships R5 — `frozen_stack_v1`: the bit-exact Rust +
vectorized-Python full stack.** Under the precision gate (bit-exact vs the
frozen ORIGINAL references, the only gate any route was allowed to pass), R5
dominates every other route on the official workflow: 23.5–25.9x fresh-process
and 25.6–26.8x warm (FINAL_SUMMARY.json `e2e`;
final_round9_results.json `e2e.t224`; cold_builders_results.json
`warm_inproc_e2e`), versus 6.42x for R1 and stage-level 10.6–10.7x totals for
the R2-era combinations (FINAL_SUMMARY.json `history`), with the held-out
BLCA control confirming 11.43x end-to-end at all-11-stages max abs diff 0.0
(heldout_control_results.json). R1 and R2 remain documented as the cheap
fallbacks: R1 needs zero new code but is capped at 6.42x by the GIL and
coarse tasks (perm-stage efficiency 0.082 @224T; bench_parallel_results.json
`derived_analysis`) and actively slows the SVD stages (sig_compose_results.json
`notes`); R2 buys its two big stages (38.7x / 16.9x) but leaves the
dominant cibersort stage untouched (~87–95 s of the 97.13 s baseline;
rust_cibersort_results.json, scaling_coldstart.json).

## I.10 Traceability (Part I)

Every numeric claim above carries a parenthetical source — a `bench/`
artifact (optionally with a JSON section) or an explicit *computed* tag
(arithmetic over cited cells only). Automated audit of this part at freeze
time: **136 numeric claim rows/lines, 136 traced, 0 untraceable** (counting
every digit-bearing table data row and every prose line whose numbers fall
outside citation parentheses; at logical-block granularity — bullet,
paragraph or table — 19/19 blocks carry a source). Per-section detail and
the audit method are recorded in `.docs_report_round2.json`. The artifact
index is [`bench/README.md`](bench/README.md).

---

# Part II — the port campaign (rounds R1–R6, iobrx 0.3.0)

Part II covers the 19 IOBRpy 0.2.0 workflows ported on top of the Part I
stack: nmf, merge_salmon, merge_star_count, prepare_salmon, log2_eset, ips,
mouse2human, lr_cal, tme_cluster, bayesprism, tme_profile, fastq_qc,
batch_salmon, batch_star_count, trust4, runall, spechla, hla_typing, plus
the calculate_sig_score glue optimization. Gold standard throughout: the
**unmodified IOBRpy 0.2.0 CLI** (`python -m iobrpy.main <subcmd>`, editable
install → `/data/users/lianchong/workspace3/IOBRpy/src`, untouched; verified
byte-identical to the workspace1 checkout for every ported module).

## II.1 Protocol & environment (all rounds)

- **Machine:** 224-core Xeon, Linux 6.8.0-138-generic
  (bench_r3/env.json; bench_r6/results.json `meta.machine`). Shared node —
  background load recorded before/after every candidate; R6 session ran at
  load 1.4–2.2 early, 5–11 mid-session (bench_r6/results.json
  `anomalies.background_load`).
- **Software:** Python 3.11.15 (`/home/lianchong/.omicos/env/.venv/bin/python`),
  numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, iobrpy 0.2.0
  (bench_r3/results.json `meta`; bench_r6/env.json).
- **Port under test:** `iobrx` via `PYTHONPATH=iobrx-github/src`, main repo
  git HEAD `921c55c7f1c52d433c5c3f87edb77a4e9b385f2c` with the R3+R4+R5
  merged working tree (28 APIs, 188 tests); R6's tme_profile_v3 arm is the
  single-file change `_fast/tme_profile_fast.py` sha256 `e2be4bbc…`, which is
  the state merged into the repo for 0.3.0 (bench_r6/results.json `meta.port`;
  bench_r6/INTEGRATION.md §1).
- **Timing discipline:** fresh subprocess **cold start** per timed run;
  1 warmup discarded per side + **3 timed reps, median** (heavy modules
  tme_profile / bayesprism: 2 timed reps, median); arms **interleaved**
  (A,B[,C]) inside one session window; sequential execution across
  candidates; load average logged per candidate
  (bench_r6/results.json `meta.timing_discipline`).
- **Speedup definition:** `t_original_median / t_port_median`
  (bench_r3/results.json `meta`; bench_r6/results.json `meta`).
- **Blind role:** R3 and R6 were measured by blind workers — measurement
  only, no optimization, no repo modifications
  (bench_r3/results.json `meta.worker_role`; bench_r6/results.json
  `meta.worker_role`).
- **Source paths:** `research/<dir>/<file>` relative to the campaign root
  given at the top of this document.

## II.2 R1 — gold-standard baseline: original CLI cold-start wall time (12 modules)

Fresh-subprocess cold-start medians of the unmodified ORIGINAL CLI
(fast modules: 3 runs median; slow modules: single run noted), measured
2026-09-08. These walls double as the parity-golden runs: every output was
frozen with sha256 for later contract checks.

| Module (input) | ORIGINAL median wall s | Source |
| --- | --- | --- |
| CLI import overhead — `import iobrpy.main`, no work | **1.4166** | research/ips_lr/timing.json `OVH_import_iobrpy_main` |
| nmf (cibersort_stad10, kmin 2 kmax 10, features 1:22) | 1.619 (nmf_A; nmf_B variant 2.07) | research/nmf_tme/timing.json `jobs.nmf_A/nmf_B` |
| tme_cluster (tme_stad_transposed 375×235) | 1.468 (A) / 4.176 (B, `--min_nc 2 --max_nc 5`) | research/nmf_tme/timing.json `jobs.tme_cluster_A/B` |
| mouse2human (2000×6 matrix / 2200-row table) | 1.319 / 1.368 | research/nmf_tme/timing.json `jobs.m2h_matrix/m2h_table` |
| tme_profile (TPM_stad10 50181×10, `--threads 16`) | **150.44** single run (determinism rep2 152.98) | research/nmf_tme/timing.json `tme_profile.real_run`, `determinism` |
| IPS (stad TPM symbol) | 1.408 | research/ips_lr/timing.json `B2_IPS_stad_tpm` |
| LR_cal (stad TPM symbol 54658×10) | 5.84 (B3); 6.1061 (A4 eset_stad_tpm); 5.7575 (X2 eset_stad_symbol) | research/ips_lr/timing.json `B3/A4/X2` |
| count2tpm (eset_stad) | 2.2404 | research/ips_lr/timing.json `B1_count2tpm_eset_stad` |
| merge_salmon (fixture8: 8 samples × 60k transcripts quant.sf) | 3.0199 | research/merge/timing.json `[0]` |
| prepare_salmon (fixture8 salmon TPM) | 1.7417 | research/merge/timing.json `[1]` |
| log2_eset (merge fixture) | 1.7236 | research/merge/timing.json `[2]` |
| merge_star_count (fixture8: 8 × ReadsPerGene.out.tab) | 1.8788 | research/merge/timing.json `[3]` |
| bayesprism (134 genes × 20 samples, `--threads 8`) | 8.629 | research/bayesprism/timing.json `timing_default_env` |

**The structural fact R1 established:** every ORIGINAL CLI call pays the
~1.3–1.4 s `iobrpy.main` import graph before any work — small modules'
1.3–1.9 s walls are import-dominated, which set the expectations (and the
floor) for the whole port campaign. The original `tme_profile` runs its nine
sub-steps as nine serial CLI subprocesses, each paying that floor
(research/nmf_tme/module_spec.md; research/bayesprism/module_spec.md;
research/merge/module_spec.md; research/ips_lr/module_spec.md).

## II.3 R3 — first port batch (research/bench_r3/results.json, 2026-09-08T14:41Z)

Six candidates, blind unified protocol (as II.1); all parity PASS:

| Candidate | ORIGINAL median s | port median s | Speedup | Parity contract (verdict PASS) |
| --- | --- | --- | --- | --- |
| nmf_python | 1.5957 | 1.4505 | **1.10x** | clusters.csv / top_features_per_cluster.csv / pca_plot.png sha256 equal across all runs of both arms |
| merge_salmon_python | 3.1912 | 1.6053 | **1.99x** | decompressed tsv.gz token-equal after column-name alignment, both output files; port column order stable, ORIGINAL's not (3 different orders in 3 reps — upstream `as_completed`) |
| lr_cal_rust | 5.9452 | 0.6943 | **8.56x** | output sha256 == frozen gold `a49ddff5…` on every run of both arms |
| tme_cluster_rust | 4.1561 | 0.6030 | **6.89x** | output csv sha256 equal across all runs of both arms (`8684ad4e…`); frozen cmd `--min_nc 2 --max_nc 5` passed explicitly on both sides |
| bayesprism_python | 8.3504 | 3.3729 | **2.48x** | theta.csv / theta_cv.csv / Z_tumor.csv sha256 == frozen hs0 gold (PYTHONHASHSEED=0, `state_order='legacy'`); without hashseed pinning, `state_order='sorted'` gives theta/theta_cv 100% bit-exact and Z_tumor ULP-only (max_abs_err 1.42e-14, 0 discrete flips) — documented expected deviation |
| tme_profile_reuse | 152.4517 | 135.9859 | **1.12x** | 9 contract files vs frozen gold: 7 raw sha256-equal + cibersort_results.csv / deconvo_merged.csv equal after stripping the unseeded `P-value_CIBERSORT` column |

R3 readings: the reuse-first tme_profile (sub-steps in one process, original
solver code untouched) only reached 1.12x because the un-optimized
calculate_sig_score glue still dominated the chain (see II.7); nmf stayed at
1.10x because both arms call the same sklearn NMF (shared-kernel floor);
lr_cal (8.56x) and tme_cluster (6.89x) confirmed the Rust-kernel route where
the hotspot is a per-row Python loop / pure-Python k-means
(bench_r3/results.json `candidates`, `anomalies`).

## II.4 R4 — second batch: Rust engines for merge_salmon and bayesprism

**merge_salmon Rust parse engine** (research/build_merge_salmon_rust/INTEGRATION.md):

- Same-window end-to-end: ORIGINAL/python-fast = 1.953x (reproducing R3's
  1.89–1.99x); ORIGINAL/rust = **2.478x** (rust e2e 1.196 s); python/rust
  = 1.269x.
- Parse segment alone: 0.4418 s → 0.0474 s = **10.32x** — but the e2e gain
  stops at ~2.5x because the byte-level write-out contract (pandas `to_csv`
  serialization + gzip) is the floor: same-window theoretical ceiling ≈2.6x,
  measured 2.478x ≈ 96% of it. This is the **write-out ceiling** (II.7,
  README floor model).
- Parity: rust-lane output bytes identical to the python-fast lane and to
  the recorded sorted-column gold shas (tpm `a7db9059…`, cnt `9d7a6034…`).

**bayesprism Rust Gibbs kernel** (research/build_bayesprism_rust/timing_routes.json,
e2e_report.json; fresh subprocess ×3 median, threads 8, PYTHONHASHSEED=0):

| Lane | median wall s | speedup vs gold CLI |
| --- | --- | --- |
| gold CLI (ORIGINAL) | 8.198 | 1.0x |
| iobrx python-fast | 3.169 | **2.59x** |
| iobrx rust (`backend='rust'`) | 2.034 | **4.03x** (1.56x vs python-fast) |

- Parity: all three hs0 output shas (theta / theta_cv / Z_tumor) equal the
  frozen gold on **both** lanes; the Rust kernel reproduces numpy's full RNG
  chain bit-for-bit (`bp_seedseq_state624` SeedSequence spawn,
  `bp_rng_doubles` PCG64 doubles, `bp_gibbs_phase`).
- Reading: the Gibbs hotspot is only p≈0.24 of the original wall, so the
  Amdahl ceiling over the python lane is ~1.9x — measured 1.56x. Rust bought
  a real but bounded gain; ≥5x needs p≥0.6 (campaign ledger theory
  "Rust 内核算", R4 refinement).

## II.5 R5 — third batch: small tables, merge_star_count, sig glue, orchestration

**(a) Small-table modules** (research/build_small_tables/timing_results.json;
fresh subprocess ×3 median, vs the R1 baseline medians):

| Module | port median s | ORIGINAL baseline median s | Speedup |
| --- | --- | --- | --- |
| ips | 0.4276 | 1.41 | **3.3x** |
| log2_eset | 0.9589 | 1.724 | **1.8x** |
| prepare_salmon | 0.8921 | 1.742 | **1.95x** |
| mouse2human | 0.538 | 1.32 | **2.45x** |

Reading: removing the ~1.4 s `iobrpy.main` import exposes **pandas
read_csv/to_csv as the new floor** — pure-IO modules cap at ~2x, ips (real
net compute) reaches ~3.3x. (Campaign ledger theory "CLI import 地板论", R4
refinement; R6 re-measured all four blind, see II.6.)

**(b) merge_star_count** (research/build_merge_star_count/timing_report.json):
port 0.966 s vs ORIGINAL 1.806 s = **1.87x** (R1 baseline 1.8788 s). Port
decompressed content stable across reps (`8b6e6818…` ×3); ORIGINAL column
order and stat-row anchor nondeterministic run-to-run (3 different sha
prefixes in 3 reps — upstream `as_completed`). Parity: column-aligned
token-level text equality + DataFrame bit equality after reindex, with the
upstream "leading 4 global stat rows not purged" defect **preserved
bug-compatibly** (research/build_merge_star_count/INTEGRATION.md; test
`test_stat_row_pollution_bugcompat`).

**(c) calculate_sig_score glue optimization + tme_profile v2**
(research/build_tme_profile_v2/INTEGRATION.md §5; threads=16, frozen
TPM_stad10, fresh subprocess):

| Measurement | BEFORE | AFTER | Speedup |
| --- | --- | --- | --- |
| calculate_sig_score in-chain step (×2 median) | 147.04 s | **4.86 s** | **30.3x** |
| tme_profile sequential wall, same session | 160.43 s | **20.01 s** | **8.02x** |
| vs ORIGINAL CLI at the current load window (146.64 s) | — | 20.01 s | **7.33x** |
| vs ORIGINAL R3 median (152.98 s) | — | 20.01 s | 7.65x |

- The glue fix (positional `_GlueCtx` index, `frozenset.intersection`,
  `arr[positions]` take, single-pass header, dict assembly) changed **no
  output bytes**: existing 11 bit-exact gates pass, per-signature selection
  identity on 12,567 signatures (`tobytes` equal), 24/24 boundary cases
  bit-exact (research/build_tme_profile_v2/INTEGRATION.md).
- Floor analysis at R5: of the 20.0 s, sig ≈4.6 s, cibersort ≈9.9 s
  (contract-bound ORIGINAL solver), LR_cal ≈4.1 s (still ORIGINAL code),
  light steps ≈0.8 s ⇒ even free sig leaves ~15.4 s (~9.5x ceiling) —
  ≥10x required the LR_cal sub-step, which became R6's v3 (II.7).

**(d) Orchestration, HLA and runall ports — real-data parity at the
orchestration ceiling (speedup ≈1 by design):** full table in II.8.

## II.6 R6 — formal blind benchmark, unified protocol (research/bench_r6/results.json, 2026-09-08T21:18Z)

All 14 core candidates re-measured blind under one protocol (II.1), same
session, arms interleaved; every parity contract PASS. **This is the
canonical speedup table for the 0.3.0 README.**

| Candidate | ORIGINAL median s (reps) | port median s (reps) | **Speedup** | Parity contract (all PASS) |
| --- | --- | --- | --- | --- |
| ips_python | 1.4017 (1.4187/1.3955/1.4017) | 0.4309 (0.4276/0.4309/0.6343) | **3.253x** | sha256 == frozen gold `793b4718…` every run both arms |
| log2_eset_python | 1.7186 | 0.7505 | **2.290x** | sha256 == frozen gold `edc2664a…` |
| prepare_salmon_python | 1.7387 | 0.7351 | **2.365x** | sha256 == frozen gold `d5546a04…` |
| mouse2human_python | 1.3115 | 0.3488 | **3.760x** | sha256 == frozen gold `174aec77…` |
| merge_star_count_python | 1.7755 | 1.0147 | **1.750x** | column-aligned token equality + frame bit equality vs gold `ec09cecc…` and vs fresh ORIGINAL (480,256 cells, 0 mismatches); port deterministic, ORIGINAL column order not (3 orders in 3 reps) |
| merge_salmon_python | 2.8074 | 1.5071 | **1.863x** | decompressed content sha == recorded sorted-column gold (tpm `a7db9059…` / cnt `9d7a6034…`), stable across reps; ORIGINAL column-aligned re-serialization == port bytes every rep |
| merge_salmon_rust | 2.8249 | 1.1707 | **2.413x** | same contract as merge_salmon_python; rust lane bytes == python lane bytes |
| nmf_python | 1.5780 | 1.3034 | **1.211x** | 3-file sha256 equal across ALL runs of BOTH arms and equal to the R3 reference shas |
| tme_cluster_rust | 4.0471 | 0.5642 | **7.173x** | output csv sha `8684ad4e…` equal across all runs of both arms |
| lr_cal_rust | 5.7549 | 0.6783 | **8.484x** | sha256 == frozen gold `a49ddff5…` for EVERY run of BOTH arms |
| bayesprism_python | 7.9518 (×2 reps, shared ORIGINAL arm) | 3.0305 | **2.624x** | theta/theta_cv/Z_tumor sha == frozen hs0 gold every run (PYTHONHASHSEED=0, `state_order='legacy'`, threads 8) |
| bayesprism_rust | 7.9518 (shared arm) | 2.1285 | **3.736x** | same hs0 3-file contract; Rust reproduces the numpy RNG chain bit-for-bit |
| tme_profile_v2 | 153.9316 (×2 reps, shared ORIGINAL arm) | 18.2283 | **8.445x** | 9 contract files: 7 raw sha256-equal + 2 equal after stripping unseeded `P-value_CIBERSORT`; compare_gold rc=0 every rep; ORIGINAL arm re-verified the frozen gold the same day |
| tme_profile_v3 | 153.9316 (shared arm) | 14.5392 | **10.587x** | same 9-file contract, rc=0 ×2 timed + warmup + 1 independent parity run; the v3 LR_cal sub-step output sha `f9d745aa…` == gold byte-for-byte |

Not re-tested in R6 (per the R6 task brief): the orchestration modules
fastq_qc / batch_salmon / batch_star_count / trust4 / runall / spechla /
hla_typing — real-data parity was already established in R5 (II.8) and their
speedup is ≈1 by the orchestration ceiling; re-running the real toolchains
was prohibitively expensive (bench_r6/results.json `meta.not_retested`).

**R3↔R6 cross-check** (bench_r6/results.json `anomalies.cross_check_r3`):
nmf 1.10x→1.21x, tme_cluster 6.89x→7.17x, lr_cal 8.56x→8.48x,
merge_salmon_python 1.99x→1.86x (the R6 ORIGINAL arm was faster in-window:
2.81 s vs R3's 3.19 s; the port arm 1.51 s matches R3's 1.61 s). Every
parity sha equals the R3 / frozen-gold value — the two independent blind
rounds agree.

**Recorded anomalies** (bench_r6/results.json `anomalies`): background load
5–11 mid-session (all comparisons same-window interleaved; the tme_profile
ORIGINAL arm 152.4–155.5 s ≈ the quiet R3 window's 152.98 s); ips port rep3
outlier 0.6343 s vs 0.4276/0.4309 (median robust); bayesprism and
tme_profile candidate pairs share one interleaved multi-arm block (ORIGINAL
measured once per rep, used by both candidates — same-window comparability);
merge_* ORIGINAL column-order nondeterminism handled per contract; the
in-chain cibersort step ran 8.08–8.60 s in this window vs 9.67–10.12 s in
the loaded R5 window (same contract-bound code, machine-load difference —
floor conclusions unaffected).

## II.7 tme_profile bottleneck-shift chain (1.12x → 8.44x → 10.59x)

The campaign's central dynamic finding — each optimization moves the wall
floor to the next incompressible part (multi-stage Amdahl):

| Stage of the chain | Speedup vs ORIGINAL | What was removed | What the floor became |
| --- | --- | --- | --- |
| R3 reuse (v1): 9 CLI subprocesses → one process, ORIGINAL code in-process | **1.12x** (152.45 → 135.99 s; bench_r3/results.json) | 9× cold-start import floor (~1.4 s each) | calculate_sig_score glue: 147 s of the chain (~84%) |
| R5/R6 v2: sig glue optimization (30.3x on the step) | **8.44x** R6 blind (153.93 → 18.23 s; bench_r6/results.json) — 7.33x in the loaded R5 window (146.64 → 20.01 s; build_tme_profile_v2/INTEGRATION.md) | O(n_sig×n_genes) Python/GIL glue (per-signature 44k-gene set rebuilds, `.loc` ×12569 ×2 legs) | cibersort (contract-bound ORIGINAL solver) + LR_cal = ~76% of the wall |
| R6 v3: LR_cal sub-step → `iobrx.lr_cal` (rust gene-filter kernel) | **10.59x** (153.93 → 14.54 s; bench_r6/results.json) | LR_cal in-chain 4.10 → 0.22 s (~18.6x; sub-step output sha == gold `f9d745aa…`) | cibersort 8.55 s (58.6%) + sig rust chain 4.54 s (31%) + light steps 0.74 s + LR_cal 0.22 s |

- v3 chain decomposition and the next floor (bench_r6/results.json
  `bottleneck_shift_conclusion`; bench_r6/INTEGRATION.md §5): even with sig
  free, the chain floor is ~9.3 s ⇒ at most ~16.6x against this window's
  ORIGINAL; **≥12x is practically unreachable without touching the cibersort
  contract** — the R3 blade sample TCGA-BR-A4IV pins the ORIGINAL python
  solver (native and csv-roundtrip alternative routes both flip the NuSVR
  basin; deterministic convergence lands on another valid solution: 13 cells,
  max 2.55e-3 — bit-exact contract forbids it).
- Robustness: against the R5 high-load ORIGINAL (146.64 s) the same v3 wall
  still gives 146.64/14.54 = **10.08x** (bench_r6/results.json
  `bottleneck_shift_conclusion.answer_ge_8x`).
- v2→v3 chain gain: 18.23 → 14.54 s = 1.25x (same session, shared ORIGINAL
  arm).
- The v3 change is exactly one function body (`_step_lr_cal` now calls
  `iobrx.lr_cal`); diff sha256 `caa8773f…`, `git apply --check` verified,
  zero `.so`/API/test changes (bench_r6/INTEGRATION.md §1–2;
  bench_r6/tme_profile_fast.py.v3.diff). This is the state shipped in 0.3.0.

## II.8 Orchestration modules: real-data parity and walls (R5)

Gold standard: official frozen real data + the same external toolchain
binaries for both arms (fastp, MultiQC, salmon, STAR, run-trust4/TRUST4
v1.1.5-r573, samtools 1.24, SpecHLA toolchain). Method: baseline output
trees sha256-manifested, moved aside in place, port re-run into the
**identical absolute paths**, every file compared (sha256 first; on mismatch
a declared run-metadata-only normalization). Sources:
research/build_orchestration/results/PARITY_REPORT.md,
research/build_orchestration/INTEGRATION.md §6,
research/build_hla/INTEGRATION.md §6–7, research/build_runall/INTEGRATION.md.

| Stage | Real-data parity | port wall | baseline wall | ratio (as recorded) |
| --- | --- | --- | --- | --- |
| fastq_qc (2 samples, 12.1 GB cleaned FASTQ) | 34 files: 28 byte-identical + 6 content-identical (run metadata), **0 DIFFERS** | 136.5 s | 137.2 s | 1.00x (0.99x port/baseline convention, same source) |
| batch_salmon (sample 68, quant.sf 51.9 MB) | 14 files: 12 byte-identical + 2 timestamp-only, **0 DIFFERS** | 521.3 s | 1185.8 s (2 samples; sample-68 span +535 s) | **1.03x per-sample** (0.97x port/baseline convention, same source) |
| batch_star_count (sample 68) | 11 files: 6 byte-identical + BAM **alignment-record stream sha-equal** (`samtools view`) + all mapping statistics equal; header @PG/@CO differ by the *declared* `--runThreadN 32→16` deviation | 252.1 s (16 thr) | ~202 s (32 thr) | 1.25x port/baseline, **fully attributable to the halved thread count** (STAR mapping speed 457 vs 594 M reads/h in its own Log.final) |
| trust4 (baseline BAM, -t 16) | **12/12 files BYTE-identical**, incl. the accelerated post-processing outputs trust4_immdata.csv + trust4_immune_indices.csv | 566.8 s | 559.4 s | 1.01x (port/baseline; tool-bound) |
| runall (C2: 2-sample full default chain, rc=0) | end-to-end product parity per step (documented jitter classes only) | 271.9 s | 277.5 s | **1.02x** |
| hla_typing (2.09 GB STAR BAM) | 104 files: 102 byte-identical; `<sample>.realign.sort.bam`+`.bai` differ by samtools-merge @PG random-ID jitter (orig-vs-orig control shows the same jitter) | 87.17 s | 91.31 s | **1.05x** |
| spechla (500k-pair frozen subset) | 87 files: 85 byte-identical; same samtools @PG jitter class | 208.63 s | 210.21 s | **1.01x** |

Python launch layer on mini fixtures (fake tools, median of 3 fresh
subprocesses; research/build_orchestration/results/mini_timing.json): the
port API entry is **14.9x** (fastq_qc), **14.8x** (batch_salmon), **10.6x**
(batch_star_count), **3.3x** (trust4) lighter than the ORIGINAL CLI launch —
the gap is the 1.295 s `import iobrpy.main` graph vs `import iobrx` 0.026 s
(lazy). On real data the external binaries dominate absolutely:
**speedup ≈ 1 is the contract-level expectation; the delivered value is API
consistency, in-process composition, resume/parallel scheduling and
byte-level output fidelity** (campaign ledger theory "编排天花板论";
research/build_runall/INTEGRATION.md structural accounting: the ORIGINAL
runall pays ~15.7 s of pure cold start across 12 child CLI spawns vs 0.33 s
once for the port).

Known tool-internal nondeterminism proven by ORIGINAL-vs-ORIGINAL controls
(not port deviations): fastp HTML duplication-rate 7th significant digit
(an ORIGINAL rerun reproduced the PORT's digit 86.319329 and differed from
its own frozen baseline 86.319322, cleaned FASTQs byte-identical —
research/build_orchestration/results/fastp_html_jitter_check.json);
samtools-merge @PG random IDs; STAR/trust4/salmon log timestamps;
`as_completed` column order in the merge modules (II.6).

## II.9 Literature anchor — R2 round: published thread scaling (a different axis)

The port campaign's round 2 recorded one literature candidate:
**anchor_iobrpy_preprint_cibersort_threads, 13.7x** — the IOBRpy preprint
(bioRxiv 2026-07-22, **doi:10.64898/2026.07.17.739055**; Part I §I.8 anchor
A1): CIBERSORT 2,864.8 s @1 thread → 209.7 s @16 threads on a 936-sample
cohort.

**This number is `published_thread_scaling`: thread scaling *inside the
ORIGINAL* Python implementation, on a different cohort, hardware and timing
scope. It is NOT a port speedup and is not arithmetically comparable to any
row of II.3–II.8.** The campaign ledger flags it as such (protocol-axis
theory: rows with `route=original, protocol=own_coldstart` are speedup 1.0
by definition — they are the denominator; the anchor lives on the
`published_thread_scaling` axis). iobrx's own cibersort thread scaling,
measured under the Part I protocol, is §I.5 (1T 95.17 s → 8T 12.96 s =
7.34x at the shipped default, bit-identical outputs at every thread count).

## II.10 Traceability (Part II)

Every number in Part II cites its source file: `research/bench_r3/results.json`,
`research/bench_r6/results.json` (+ `bench_r6/INTEGRATION.md`,
`bench_r6/env.json`), the four R1 baseline studies
(`research/{nmf_tme,ips_lr,merge,bayesprism}/timing.json`), and the per-build
integration records (`research/build_*/INTEGRATION.md`,
`build_small_tables/timing_results.json`,
`build_merge_star_count/timing_report.json`,
`build_bayesprism_rust/timing_routes.json` + `e2e_report.json`,
`build_orchestration/results/{PARITY_REPORT.md,mini_timing.json,fastp_html_jitter_check.json}`,
`build_hla/INTEGRATION.md`, `build_runall/INTEGRATION.md`) under the campaign
root given at the top. Speedup values quoted to 2–4 significant figures are
verbatim `speedup` fields of the cited JSON (or the ratio of the two cited
median cells, marked where the source table already recorded the ratio).
The repo's own re-verifiable evidence for the parity contracts is the test
suite (`tests/test_parity_*.py`, 188 passed / 15 deselected at 0.3.0).


## Reproducible real-tool follow-up (2026-09-09)

The [real FASTQ/BAM/HLA recipe](benchmarks/real_tools/README.md) supplies 36 fresh-process runs, public input accessions, explicit tool environments, raw logs and per-product checks. Four workflow families match on the declared products; Salmon varies within the original and across wrappers at both 4 and 1 thread. HLA extraction was slower for iobrx on this fixture. These new observations do not turn the historical R3–R6 reports above into independently reproduced results.
