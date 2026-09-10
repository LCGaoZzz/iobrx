# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - Unreleased

### Changed

- Reference data is bundled inside the wheel (``iobrx/_resources``); IOBRpy is demoted to the optional ``python`` extra. The default install no longer pulls IOBRpy's transitive dependencies or its exact pins, and ``import iobrx`` no longer imports IOBRpy. quanTIseq and signature scoring reuse upstream IOBRpy helpers and fail with an actionable ``pip install 'iobrx[python]'`` message when it is absent.
- Dependency pins relaxed to ranges so iobrx coexists with shared environments (e.g. the Omicos kernel). The validated bit-exact numerical environment is published as the ``strict`` extra and remains what the parity gates and CI assert.
- Document thread guidance for large cohorts (explicit ``n_threads``; single-call batching for CIBERSORT's fixed per-call overhead).
- Remove committed development artifacts from the repository root.

## [0.3.0] - 2026-09-09

### Reliability and Omicos hardening

- Scope `runall` output checks to calculation products and completion records. Notes and figures no longer block resume; existing schema 2 state is compatible, and partial-output protection remains.
- Add the standalone `extract_hla_read` public API (29 public APIs / 27 workflow functions total), reusing existing extraction helpers. Prepared tools are the default; dependency auto-installation is explicit. Update API docs to reflect current dry-run, resume and NMF output behavior.
- Simplify the Agent/Skill entrypoints: autonomous API/CLI/MCP use and on-demand references, with no mandatory doctor/validate/run/status sequence. Harness provenance defaults to metadata; SHA-256 auditing is opt-in. Allow existing output directories without run-file collisions, and separate artifact inspection from historical execution status.
- Propagate QC/MultiQC failures, make dry runs write-free, and bind resume decisions to input/reference/tool and artifact hashes.
- Require successful per-step records before reusing tables. Retry partial writes and failed merges, invalidate earlier checkpoints for a fresh attempt, and reject legacy run state without these records.
- Require STAR BAM and GeneCounts together; respect `IOBRX_DISABLE_RUST` in Salmon parsing.
- Use the original file-based CIBERSORT solver inside `runall`; retain NMF feature rankings in fresh output directories.
- Add 16 typed Omicos adapters (27 identifiers total), input/output isolation and truthful failed manifests.
- Add 11 executed tutorials (23 total), each with two reviewed figure revisions and recorded API time.
- Enable sequential `tme_profile` and committed-data NMF checks; choose release notes from the actual tag and synchronize container metadata.
- Correct unreleased wheel/PyPI/container claims and distinguish imported campaign reports from locally reproduced evidence.

### Added — 18 workflow APIs and signature-score optimization (campaign R3–R6)

- **Pure-Python ports (bit-exact, chosen route):** `nmf`,
  `merge_star_count` (stat-row bug-compat preserved), `prepare_salmon`,
  `log2_eset`, `ips`, `mouse2human`, and the orchestration stages
  `fastq_qc`, `batch_salmon`, `batch_star_count`, `trust4` (with accelerated
  immune-data post-processing), `runall`, `spechla`, `hla_typing`.
- **Rust-kernel ports (`backend="auto"`, Python fallback):** `lr_cal`
  (`lr_gene_valid_mask` gene filter), `tme_cluster` (`tme_kmeans_best` /
  `tme_kl_index` / `tme_missing_order`), `merge_salmon`
  (`merge_salmon_parse` quant.sf reader).
- **`bayesprism`:** python-fast lane by default; opt-in Rust Gibbs kernel
  (`bp_gibbs_phase` / `bp_rng_doubles` / `bp_seedseq_state624`) reproducing
  numpy's full RNG chain bit-for-bit. Default `state_order="sorted"` removes
  the original's `PYTHONHASHSEED` dependence (theta/theta_cv 100% bit-exact
  vs a `PYTHONHASHSEED=0` original; Z_tumor ULP-only, max abs 1.42e-14,
  zero discrete flips); `state_order="legacy"` + `PYTHONHASHSEED=0`
  reproduces the frozen gold shas exactly.
- **`tme_profile`:** the full 9-step chain (sig scores + 6 deconvolutions +
  LR_cal) in one process (`reuse_fast`, `cibersort_backend="original"`);
  the shipped v3 routes the LR_cal sub-step through `iobrx.lr_cal`
  (sub-step 4.10 s → 0.22 s, output sha256 == gold byte-for-byte).
- **`calculate_sig_score` glue optimization:** the in-chain step went
  147.04 s → 4.86 s (**30.3×**) with output bytes unchanged (11 existing
  bit-exact gates + per-signature selection identity on 12,567 signatures +
  24/24 boundary cases).
- 4 new Rust kernel families in total: `lr_gene_valid_mask`, `tme_kmeans`,
  `merge_salmon_parse`, `bp_gibbs`.
- 15 new parity test modules (`tests/test_parity_{nmf,merge_salmon,
  merge_star_count,lr_cal,tme_cluster,tme_profile,bayesprism,
  bayesprism_rust,ips,log2_eset,prepare_salmon,mouse2human,
  orchestration,hla,runall}.py`); the default suite is now
  **188 passed, 15 deselected**.

### Benchmarks — R6 formal blind benchmark, unified cold-start protocol

Source: `research/bench_r6/results.json` (14 candidates, interleaved arms,
medians; all parity contracts PASS) — full tables and per-round history in
[BENCHMARKS.md](BENCHMARKS.md) Part II:

- `tme_profile` **10.59×** (153.93 s → 14.54 s); bottleneck-shift chain
  1.12× (R3 reuse) → 8.44× (R5 sig-glue fix) → 10.59× (R6 LR_cal→rust).
  Next floor ~9.3 s (~16.6×); ≥12× unreachable without breaking the
  contract-bound original CIBERSORT solver.
- `lr_cal` **8.48×**, `tme_cluster` **7.17×**, `mouse2human` **3.76×**,
  `ips` **3.25×**, `bayesprism` **2.62×** (python) / **3.74×** (rust),
  `merge_salmon` **2.41×** (rust; write-out ceiling ~2.5×),
  `prepare_salmon` **2.37×**, `log2_eset` **2.29×**,
  `merge_star_count` **1.75×**, `nmf` **1.21×** (shared sklearn floor).
- Orchestration stages (`fastq_qc`, `batch_salmon`, `batch_star_count`,
  `trust4`, `runall`, `spechla`, `hla_typing`): real-data parity verified
  end-to-end on the official frozen data (0 DIFFERS per stage, declared
  run-metadata normalization only); speedup ≈1 by the orchestration ceiling
  (0.99–1.25× measured, the STAR residual fully attributable to a declared
  thread deviation); the Python launch layer is 3.3–14.9× lighter.

### Changed

- README (EN + zh-CN): three-route comparison table for all 28 APIs
  (original CLI / pure-Python / Rust, final choice, blind speedup, parity
  contract), the floor-model explanation of why speedups differ, the
  bit-exact contract and nondeterminism ledger, external-tool requirements,
  dependency-pin rationale, and 0.3.0 install routes (pip / maturin).
  README timing tables keep the "Parity vs IOBRpy" column (generated by
  `scripts/update_tutorial_index.py`, so re-benchmarking preserves it).
- BENCHMARKS.md restructured into Part I (acceleration stack, official
  gates — numbers unchanged, every figure still cited to `bench/`
  artifacts) and Part II (port campaign rounds R1–R6 — every number cited
  to its campaign research artifact).

## [0.2.0] - 2026-09-08

### Changed

- Removed mandatory AVX-512 compilation and linking. Quantile normalization
  and its parity helper delegate sorting/reductions to the installed NumPy,
  preserving local tie order on an AVX2-only desktop CPU.
- Made native import optional. Added `backend="auto"|"rust"|"python"` to
  CIBERSORT and signature scoring, `IOBRX_DISABLE_RUST`, and `backend_info()`.
  Auto CIBERSORT falls back to IOBRpy for unsupported BLAS layouts; PCA can
  use sklearn when the matching bundled LAPACK entry point is unavailable.
- Resolve NumPy and SciPy libraries from their own installation directories,
  including split/overlay environments.
- CI runs smoke/portability tests on every runner rather than skipping CPUs
  without AVX-512, plus official parity, disabled-native and baseline-sorting
  checks with the validated dependency constraints.
- Rewrote the English README and added a Chinese README with actual desktop
  timings, tutorial links, input-scale guidance, and explicit platform limits.
- Corrected documentation of annotation duplicate selection, the upstream
  `zscore` method, RNA-seq quantile-normalization choices, and ESTIMATE's exact
  `affymetrix` platform spelling. Existing numerical defaults are retained.

### Added

- Release workflow for a CPython 3.11 manylinux2014 x86-64 wheel and source
  distribution, PyPI Trusted Publishing, GitHub Release assets/checksums,
  and a tested versioned GHCR container with hash-locked dependencies.
- Binary-only installation checks in a clean environment, source-archive
  rebuild checks, and all parity/portability gates inside the container.
- Third-party attribution and full vendored license texts are included in
  both binary wheels and source archives and checked before publication.
- PyPI and Tsinghua installation instructions. Core numerical dependencies
  now select the exact validated versions during an ordinary pip install.
- Twelve executed notebooks: eleven individual analyses and a complete TME
  workflow, public fixtures with provenance/checksums, compact result tables,
  36 figure exports, and a documented two-round visual review.
- Tutorial generation/execution, integrity validation, repeated benchmarking,
  and README/gallery synchronization scripts.
- Tests for tied/NaN sorting, array layouts, explicit/automatic backend
  selection, missing BLAS and disabled native imports.

### Validation scope

- Built and executed on WSL2 Ubuntu / Python 3.11 / i9-13900KF without AVX-512.
- Native CIBERSORT non-P-value columns remain exact against IOBRpy on the
  tested fixtures; seeded native P-values differ from upstream unseeded draws.
- Removing the CPU instruction requirement does not certify all operating
  systems: upstream IOBRpy packaging still limits installation combinations.

## [0.1.0] - 2026-09-08

### Fixed

- README Quickstart now starts from a self-sufficient seeded synthetic cohort
  (~200 genes × 20 samples; verified verbatim rc=0 in a fresh venv with no
  testdata and no network); the official IMvigor210 path follows as a clearly
  marked second block that skips cleanly when neither `IOBRX_TESTDATA` nor a
  reachable mirror exists (round-3 review B1).
- Official-data download now goes through a mirror list (github.com direct,
  gh-proxy.com, ghproxy.net; override with `$IOBRX_TESTDATA_MIRRORS`) and
  fails with `iobrx.OfficialDataUnavailable`, which spells out the
  remediation, instead of a raw urllib traceback (B2). Shared by
  `examples/`, the `full`-marker tests and the new public
  `iobrx.load_official()`.
- Plain `pytest -q` is green by default (pyproject `addopts = "-m 'not
  full'"` deselects the 11 official-data gates); the documented `--run-full`
  flag is now real (registered in `tests/conftest.py`) (B3, B4).
- README badge row: the 404-linking `[PyPI 0.1.0]` badge is replaced by a
  static version badge; Install states "not yet on PyPI — build from
  source" (B5).
- BENCHMARKS.md citation precision: stage-CV range stated for both the
  fast/steady legs (0.0–1.7%) and all rep lists (0.0–28.5%);
  `gates.GATE3_perf.mapping_steady_state_224T.median_ms`;
  `findings.FINDING_5_quantile_normalize_np`; background-load range
  corrected to 8.3–32.9 (`e2e.driver_loadavg`).

### Added

- `iobrx.load_official()` / `iobrx.OfficialDataUnavailable` public
  data-resolution helper (env var → local cache → mirror download).
- GitHub Actions CI (`.github/workflows/ci.yml`): ubuntu-latest, Python 3.11,
  cargo cache, maturin wheel build via `pip install .[test]`, `pytest -q`.
- Initial packaged release of the finished fast stack as the installable
  library `iobrx` (hybrid maturin build: `src/iobrx` Python package +
  `iobrx_rust` cdylib from `rust/`).
- Public API (bit-exact drop-ins over IOBRpy, see README):
  `cibersort`, `calculate_sig_score`, `count2tpm`, `quantiseq` /
  `deconvolute_quantiseq`, `epic`, `mcpcounter`, `estimate_score`,
  `anno_eset`, plus `set_threads` / `get_threads` and `__version__`.
- Rust extension vendoring scikit-learn 1.7.2's dense libsvm
  (`svm.cpp`/`svm.h` + its two sklearn headers) and Intel's x86-simd-sort
  argsort kernel, compiled with bit-exactness-preserving flags.
- Tests: `tests/test_api_smoke.py` (synthetic, original-vs-fast bit parity,
  no network) and `tests/test_parity_official.py` (`full` marker; the 11
  official gates against original-on-the-fly, data from `IOBRX_TESTDATA` or
  the IOBR `data-v1.0` release).
- Examples: `examples/quickstart.py` and `examples/full_workflow.py`
  (11 official stages, per-stage timing table, optional `--verify-parity`).
- Docs: README (install, quickstart, API, caveats), BENCHMARKS.md
  placeholder, MIT LICENSE, third-party notices for vendored sources.
