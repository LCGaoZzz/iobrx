# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- CI: smoke tests are gated on the runner CPU exposing `avx512f` — the
  extension's argsort path is deliberately compiled for the AVX-512 kernel
  numpy dispatches (bit-exactness contract), which SIGILLs on CPUs without
  AVX-512; build + install remain verified on every runner, tests run
  whenever the ISA is present, and a `::warning::` is logged on skip.
- README "Install" now states the AVX-512 CPU as a hard requirement instead
  of only "validated on AVX-512 hardware".

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
