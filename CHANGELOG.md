# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- README timing tables (English and Chinese) now carry a "Parity vs IOBRpy"
  column stating each analysis's verified numerical relationship to the
  original — bit-identical everywhere except CIBERSORT's P-value — generated
  by `scripts/update_tutorial_index.py` so re-benchmarking keeps it. Both
  READMEs, `tutorials/BENCHMARKS.md` and the fidelity sections now spell out
  the gate's exact assertion (equal labels; every numeric cell
  `max_abs_diff == 0.0` vs the same-environment original), the validation
  record, and that CI re-runs all 11 gates on every push and pull request.

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
