# iobrx 0.3 hardening: validation and limits

## Release follow-up: real external tools

After PR #5 was merged, the release follow-up added actual FASTQ, BAM and HLA
comparisons, three repetitions per arm, and tutorials 24–28. The full input,
tool, timing and output record is in [the real-tool evidence](../benchmarks/real_tools/README.md).
This supersedes the external-tool evidence gap in the earlier audit below;
it does not extend that evidence to TRUST4, CRAM, every reference layout or
independent HLA genotype accuracy. Salmon variability and the slower small
HLA extraction case are explicitly retained. No scientific kernel was changed
by this release/evidence work.

The remaining sections describe their historical audit snapshots. Public
distribution status is tracked on the release and PyPI pages, independently
of those earlier checks.

Audit date: 2026-09-09. Starting PR commit: `186a15d`.
Machine-readable results and sanitized JUnit records are in
[validation/0.3.0](validation/0.3.0/summary.json). They include a digest of the
Python package/harness sources, versions and observed suite durations.

## Follow-up: scoped resume and standalone extraction

The final review of `27df5c1` found that unrelated notes and figures in a
`runall` output directory still blocked resume. The output inventory now
checks calculation products and completion records: successful table outputs,
cleaned reads, quantification/count files, BAMs and TCR/BCR results. Output
paths in tool sidecars cover custom read suffixes. Unrelated files are not
hashed. Existing schema 2 records are projected onto this scope when read,
so they do not force a new run merely because the inventory changed.

Regression tests cover additions/edits/deletions of notes and figures,
backward reading of whole-tree schema 2 state, changed/deleted calculation
products, an added quantification sample, and custom-named reads. The earlier
partial-write and retry tests remain enabled.

`extract_hla_read` now provides the missing standalone Python API, reusing the
existing extraction helpers. Both backends default to `auto_install=False`.
Tests compare hg19/hg38 command arguments and output bytes against the original
entrypoint using stub scripts, and cover dependency, argument and exit-code
failures. This is interface validation, not a new real HLA dataset benchmark.

The public `runall` and NMF docstrings now describe the corrected default
behavior, including dry runs without writes and feature output in fresh
directories. The untouched upstream backend remains explicitly distinguished.

Validation on WSL2 / Python 3.11.15, using an isolated environment with the
validated numerical stack (the shared Omicos environment was not modified):

| Check | Result |
| --- | --- |
| Focused resume/HLA contracts | 102 passed; 1 full test deselected; 2.26 s |
| Rebuilt, installed wheel: default package + harness | 302 passed; 15 full tests deselected; 56.34 s |
| Installed wheel: full runall numerical comparator | 1 passed; 18 default tests deselected; 22.17 s |
| Installed wheel: API/reliability/extraction with Rust disabled | 70 passed; 9.08 s |
| Packaging and metadata | Local Linux x86-64 CPython 3.11 wheel built/installed; both changed modules match the source; `pip check` and release source/installed checks passed |
| Omicos Skill validator | 1 Skill, 0 errors, 0 warnings |

Commands from the checkout after installing the wheel and companion package:

```bash
python -m pytest -q tests agent-harness/tests
IOBRX_TESTDATA="$PWD/tutorials/data" python -m pytest -q -m full tests/test_parity_runall.py
IOBRX_DISABLE_RUST=1 python -m pytest -q tests/test_api_smoke.py tests/test_workflow_reliability.py tests/test_extract_hla_read.py
python -m pip check
python scripts/check_release.py --source --installed
```

These checks do not rebuild a portable release image or repeat every historical
benchmark locally. GitHub's CI, harness and release workflows all passed at
the preceding head `27df5c1`; the new commit's checks are separate evidence.

## Earlier follow-up: partial-output resume repair

The final audit of `9b63dac` found that a failed signature writer could leave
a nonempty CSV, which a later resume skipped and reported as successful.
The installed-wheel reproducer returned 7 on its first attempt and incorrectly
returned 0 on retry. This was a gap in the initial hardening tests below.

Run-state schema 2 now records each successful table-producing step with its
output hash. Failed steps have no completion record and are retried; successful
upstream steps remain reusable. A new run without `resume` discards previous
table checkpoints. Legacy state without these records requires a new output
directory. At this repair, an uncatchable process kill left unverified files
that the whole-tree guard rejected. The follow-up above scopes that guard to
calculation products; it still cannot authorize reuse of unrecorded tables.

The repaired, rebuilt and installed wheel keeps returning 7 while the simulated
writer fails, records `failed`, and re-executes that writer. Regression tests
also demonstrate successful recovery once the failure clears.

| Follow-up check | Result |
| --- | --- |
| New failure/retry scenarios | 17 cases: both pipeline modes, all table steps, repeated failure, exceptions, missing outputs, interrupted merge, fresh attempts and legacy state |
| Focused scheduler/reliability suite | 41 passed; 1 full test deselected |
| Default package and harness | 263 passed; 15 full tests deselected |
| Full runall numerical comparison | 1 passed; 18 default tests deselected |
| Rebuilt installed wheel, Rust disabled | 36 passed |
| Release metadata and tutorial integrity | Passed; 23 notebooks, 69 figures, 4 data checksums |

[Follow-up source digest, wheel hash and JUnit records](validation/0.3.0/resume-fix/summary.json)
identify this repair separately from the initial evidence. Scientific kernels
are unchanged; external-tool tests still use stubs. Documentation now agrees
with the actual 27 harness identifiers, exact dependency pins and unpublished
release status.

## Initial reproduced checks at `9b63dac`

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
- `runall` binds calculation products and completion records to its input/configuration. Unrelated notes and figures may be added, edited or deleted. Schema 2 state remains readable; older trees without per-step records, changed inputs/references, or modified calculation products require a fresh output directory. Hashing the actual calculation files still adds I/O cost for large data.
- Table reuse also requires a successful per-step completion record. Nonempty partial outputs from failed steps and interrupted merges are retried; pre-schema-2 state cannot authorize reuse.
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
BayesPrism remain direct Python API capabilities. Existing typing wrappers
retain upstream auto-install behavior. The new standalone `extract_hla_read`
uses prepared dependencies by default (`auto_install=False`); no automatic
HLA adapter is added by this repair.

At the audit date, PyPI's iobrx JSON endpoint returned 404; GitHub's latest
release was v0.1.0 with no binary attachments. The wheel/PyPI/container
workflow exists, but publication and Tsinghua synchronization remain separate
work. This PR updates code, documentation and validation; it does not publish
a release or deploy Omicos content.
