# iobrx 0.3 hardening: validation and limits

Audit date: 2026-09-09. Starting PR commit: `186a15d`.
Machine-readable results and sanitized JUnit records are in
[validation/0.3.0](validation/0.3.0/summary.json). They include a digest of the
Python package/harness sources, versions and observed suite durations.

## Reproduced checks

| Check | Result | Scope |
| --- | --- | --- |
| Default package and harness suite | 246 passed; 15 full tests deselected | Numerical/contract tests, CLI/MCP, relocated Skill, failure handling and typed adapters |
| Full parity suite | 15 passed, no skips | Original numerical comparator; NMF uses the committed tutorial CIBERSORT table; sequential and parallel tme_profile are enabled |
| Installed wheel with Rust disabled | 19 passed | Public API smoke plus QC/resume/reference regressions, including the Salmon parser switch |
| Installed harness wheel, new adapters | 16 passed | Actual built companion distribution; numerical and file-contract adapters |
| Tutorial integrity | 23 executed notebooks, 69 figure exports, 4 dataset checksums | Includes 11 new notebooks, each executed at figure revisions 0/1/2 |
| Omicos catalog validator | 1 Skill, zero errors/warnings | Repository-specific frontmatter and resource rules |
| Packaging | Main and companion wheels built and installed | Local main wheel is CPython 3.11 manylinux_2_34; release CI builds the older manylinux2014 baseline |

The generic Codex Skill validator does not recognize Omicos's required `id`,
`tier` and runtime metadata. The applicable `omicos-admin` validator passed
without changing that metadata. No catalog deployment or production change
was performed.

The existing WSL validation dependencies were reused through an isolated
environment path. The new notebooks ran with the existing Omicos interpreter
and the validated numerical stack. The shared Omicos environment was not
upgraded or repinned. GitHub Actions separately performs fresh wheel/container
installation; a local build is not a claim that a public distribution exists.

```bash
python -m pytest -q tests agent-harness/tests
IOBRX_TME_PROFILE_SEQ=1 IOBRX_TESTDATA="$PWD/tutorials/data" python -m pytest -q -m full tests
IOBRX_DISABLE_RUST=1 python -m pytest -q tests/test_api_smoke.py tests/test_workflow_reliability.py
python scripts/validate_tutorials.py
python scripts/check_release.py --source --installed
```

Install the companion wheel (or expose its packaged script directory) before
running harness tests. `--installed` requires a wheel import, not `PYTHONPATH=src`.

## Deliberate reliability differences from upstream

- QC and MultiQC failures return nonzero status; runall does not write a completion marker after them.
- Dry runs print a complete plan and do not write results, directories or flags.
- QC, Salmon and STAR resume only after matching SHA-256 records for inputs, reference indices, tool binaries and products. STAR needs both BAM and GeneCounts. Linked reference directories are included, with cycle protection.
- `runall` binds the whole output tree to its input/configuration. Old trees without state, changed inputs/references, or edited/deleted outputs require a fresh output directory. Whole-file hashing adds I/O cost for large data; this is intentionally conservative.
- `runall` uses the original file-based CIBERSORT solver, matching the conservative tme_profile default. The reported campaign corner case is not reclassified as solved in the Rust kernel.
- NMF saves feature rankings even when its output directory did not previously exist.
- `backend="python"` remains the untouched upstream escape hatch for orchestration. It does not inherit the hardened auto-path behavior; the harness uses the hardened path.

## Scientific and delivery limits

Exact equality applies to explicit matched-method fixtures and parameters,
not arbitrary datasets or every pipeline byte. Existing exceptions remain:
CIBERSORT stochastic P-values, upstream asynchronous column ordering, gzip and
tool timestamps, BayesPrism state-order effects, and the documented runall
signature rounding tolerance. The NMF harness adapter permits floating-point
roundoff after memory-layout conversion (`rtol=1e-12, atol=1e-14`).

R3–R6 speedups in BENCHMARKS are campaign-reported. Some cited raw scripts,
logs and frozen files have not been archived in this repository. Those claims
are preserved as historical reports and were not independently rerun here.
The new tutorial timing JSON files are actual small local API calls, with
their input shapes, thread request and fixture description recorded.

External-tool tests use deterministic stubs. No new real fastp/Salmon/STAR/
TRUST4/SpecHLA end-to-end validation or real-data external-tool notebooks were
completed in this audit. Reference preparation and full tool environments are
still required. The file-merging tutorials explicitly use synthetic data;
the BayesPrism tutorial uses a shortened chain for execution demonstration,
not a converged scientific estimate.

The harness exposes 27 identifiers. HLA/SpecHLA and custom-reference
BayesPrism remain direct Python API capabilities. HLA wrappers retain upstream
auto-install behavior; they need a prepared isolated environment before an
automatic Omicos adapter can be added responsibly.

At the audit date, PyPI's iobrx JSON endpoint returned 404; GitHub's latest
release was v0.1.0 with no binary attachments. The wheel/PyPI/container
workflow exists, but publication and Tsinghua synchronization remain separate
work. This PR updates code, documentation and validation; it does not publish
a release or deploy Omicos content.
