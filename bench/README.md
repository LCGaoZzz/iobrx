# bench/ — frozen measurement artifacts

Every number quoted in [`BENCHMARKS.md`](../BENCHMARKS.md) and in the README
benchmark summary traces to a file in this directory. These are **verbatim
copies** of the benchmark-harness outputs from the development workspace
(`iobrx/bench/`, plus `iobrx/refs/baseline_timing.json`); they are read-only
evidence, not scripts. Any absolute paths inside them refer to that
development workspace; before public release the local username in those path
strings was redacted to `user` (path strings only — every measurement value is
untouched). Nothing in `src/`, `rust/` or `tests/` was modified to
produce them, and no file here was edited after copying except that redaction.

## Index

| File | What it holds |
| --- | --- |
| `FINAL_SUMMARY.json` | Final freeze (2026-09-08): official 11-stage gates, 83-case audit rerun, held-out BLCA control, e2e t224/t64/t32 + warm, cibersort thread scaling 1–224, environment, provenance, campaign round history 1.0→25.89x |
| `baseline_timing.json` | ORIGINAL count2tpm wall times on the official input (median 4.441 s) — from `iobrx/refs/baseline_timing.json` |
| `bench_parallel_results.json` | Route R1: joblib-parallel-only ORIGINAL cibersort (n_jobs 16/64/224), dispatch benchmark, identity vs n_jobs=1 |
| `count2tpm_fast_results.json` | Route R2: vectorized count2tpm stage (38.7x) + precision |
| `quantiseq_fast_results.json` | Route R2: quantiseq alias-map memoization stage (16.9x) + precision |
| `sig_compose_results.json` | Route R2: signature-score composition gates/timings; measured verdict that joblib threads do not help the SVD workload |
| `misc_fast_results.json` | Route R2: anno_eset / epic / mcpcounter / estimate vectorized stages (gates + timings) |
| `small_stages_results.json` | anno_eset / estimate method-variant parity gates (sum/mean, affymetrix platform) |
| `rust_cibersort_results.json` | Route R3: Rust bit-faithful cibersort stage, 1/64/224 threads, phase breakdowns, original n_jobs=1 baseline (87.17 s) |
| `rust_ssgsea_results.json` | Route R4 step: Rust ssGSEA stage gate + timings (10.9–11.5x vs original) |
| `integration_rust_results.json` | Route R4 step: integration stage via Rust ssGSEA leg; per-leg gates and timings |
| `cold_builders_results.json` | Route R4 step: cold builders (vectorized HGNC alias map 13x build, count2tpm disk cache) + e2e before/after |
| `pool_hoist_results.json` | Route R4 step: process-global rayon pool registry; thread-invariance gates; pool build cost 0.147 s; wheel 501,213 B |
| `full_e2e_results.json` | Route R5 (fix-round-11 era) full-stack e2e t224/t64 with 11-stage verification |
| `final_round9_results.json` | Route R5 round 9: Rust PCA/z-score legs (bit-exact, all 58 signatures), stage benchmarks, e2e 3.7515 s |
| `fix_round11_results.json` | The 5 adversarial-audit findings and their fixes; full 83-case battery rerun; regression e2e |
| `forensic_blca_results.json` | Held-out BLCA first-divergence forensics (S0–S10), csv-parse root cause, `precise_xstrtod` port and its bitwise validation |
| `audit_precision.json` | Post-fix 83-case adversarial audit (79 PASS + 3 BOTH_ERROR_SAME_TYPE + 1 ORIG_ERROR_FAST_OK) |
| `audit_precision.prefix11.bak.json` | Pre-fix audit snapshot (76 PASS, 3 FAIL, 1 ERROR_MISMATCH) |
| `tolerance_results.json` | NuSVR tolerance experiment: tol 1e-3 vs 1e-5/1e-8 moves fractions up to 0.0793 abs (~8 pp) |
| `heldout_control_results.json` | Held-out BLCA control: ORIGINAL vs FINAL fast stack, all 11 stages, 11.43x total, all max_abs_diff 0.0 |
| `scaling_coldstart.json` | CIBERSORT thread scaling 8–224T (pre-fix crate), cold-process start (~3.1 s), memory (RSS deltas), wheel build times |
| `final_freeze_gates.json` | Freeze-era official gates detail |
| `final_freeze_e2e_results.json` | Freeze e2e raw (t224/t64/t32) |
| `final_freeze_warm_and_scaling.json` | Freeze warm loop + cibersort 1–224T scaling (post csv-parse-fix crate) |
| `literature_anchors.json` | 12 external literature anchors with DOIs, quotes, protocol caveats; internal 97.13 s reference point |
| `pypi_021_diff.json` | iobrpy 0.2.1 (PyPI) vs local 0.2.0 baseline diff: numerically identical, no re-baselining |

The frozen reference outputs themselves (`refs/*.parquet`, `refs/epic.pkl`) and
the benchmark *scripts* remain in the development workspace; the pytest
`full` marker and `examples/full_workflow.py --verify-parity` re-derive the
parity gates on the fly instead of shipping the binaries.
