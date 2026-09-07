# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-07

### Added

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
