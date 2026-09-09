# 0.3 extension audit

The current harness is 0.2.0 and requires iobrx 0.3.x. It exposes 27 typed
analysis identifiers. The combined default suite passed 246 tests, and the
16 new-adapter tests also passed against installed main/companion wheels.
The Omicos catalog validator reported zero errors or warnings. See
[current evidence and limits](../docs/VALIDATION_0.3.md) for exact scope.

The original 0.2 integration record below is historical. New HLA and
custom-reference BayesPrism adapters are not claimed, and the external-tool
tests in this audit use stubs.

# Harness validation record

Validated on **2026-09-08**, WSL2 Ubuntu / Linux x86-64, Intel Core i9-13900KF
(no AVX-512), Python 3.11.15. This record tests the harness adapter, separately
from the existing [11 official upstream parity gates](https://github.com/LCGaoZzz/iobrx/tree/main/tests).

## Numerical execution and interfaces

- All **11 analyses** ran through the JSON CLI in fresh subprocesses. Every
  Parquet table matched direct iobrx API results with `check_exact=True`,
  including labels, index, column order and dtypes. EPIC checked all three
  result tables. CIBERSORT used `perm=0`, `QN=false`; this is not a validation
  of permutation P-values.
- The installed companion wheel passed **36 tests in 28.31 s**, including a
  real MCP SDK 1.30.0 stdio handshake, discovery, validation, numerical run,
  result status and rejected workspace escapes.
- The existing **Omicos Python environment** passed **35 CLI/Skill tests in
  35.46 s**. iobrx's tested manylinux wheel and the already provisioned
  IOBRpy resources were supplied through isolated overlays outside the
  environment; no system/site-package replacement was needed. The optional
  MCP SDK was tested separately in the installed-wheel environment.
- Negative tests covered unknown parameters, wrong scale/species, malformed
  matrices, duplicate/missing IDs, non-finite values, output collisions,
  changed results, Python fallback and an unavailable explicit Rust request.
- The whole Skill was copied to a different directory with spaces and ran
  count-to-TPM from an unrelated working directory. Its canonical wrapper
  used the colocated runtime rather than the installed companion package.
- The pure-Python wheel and source distribution both built successfully;
  the wheel contains both CLI/MCP launchers and the source archive includes
  the Agent, Skill, installer and request example. `pip check` and actionlint
  passed. Upstream warnings about unavailable effective lengths are retained
  in run manifests and do not change the returned numerical results.

Numerical versions: iobrx 0.2.0, IOBRpy 0.2.1, NumPy 2.2.6, pandas 2.3.3,
SciPy 1.16.3, scikit-learn 1.7.2, GSEApy 1.3.1, PyArrow 20.0.0.

## Observed Omicos run times

One fresh CLI process per analysis; **2 requested threads**. These are
single-run engineering checks on subsets, not repeated benchmark medians.
API time includes first-call initialization; total time includes imports,
input validation/hashing and output writes, but excludes process startup.
Do not compare these directly with the full-size warm notebook benchmarks.

| Analysis | Features × samples | API time (s) | Total (s) |
| --- | ---: | ---: | ---: |
| Annotation | 60,483 × 3 | 0.149 | 0.659 |
| Count → TPM | 60,483 × 3 | 0.080 | 0.600 |
| CIBERSORT (`perm=0`, `QN=false`) | 42,415 × 3 | 2.924 | 3.310 |
| EPIC | 42,415 × 3 | 0.481 | 0.877 |
| quanTIseq | 42,415 × 3 | 0.613 | 1.031 |
| MCP-counter | 42,415 × 3 | 0.003 | 0.434 |
| ESTIMATE | 42,415 × 3 | 0.023 | 0.406 |
| PCA signatures | 872 × 12 | 0.850 | 1.235 |
| Mean-based signatures | 872 × 12 | 0.712 | 1.110 |
| ssGSEA signatures | 872 × 12 | 0.707 | 1.086 |
| Integrated signatures | 872 × 12 | 0.734 | 1.163 |

Fixtures are the first three STAD samples and first twelve IMvigor210 samples
from the committed public Parquet files. TPM uses `check_data=true` and
`remove_version=true`; the smaller sample subset changes zero-variance gene
filtering and therefore the resulting feature count. Marker/ESTIMATE inputs
are explicitly log2(TPM+1). IMvigor210 is declared `preprocessed`, since its
provided signed values are not a linear TPM matrix.

## Omicos integration

- The local omicos-admin shared Skill validator accepted the shipped Skill:
  **0 errors, 0 warnings, 0 informational findings**.
- A separate local omicos-core instance discovered the installed bundle:
  `/api/agents` returned `iobrx_analyst`, tier `community`, skill list `[iobrx]`;
  `/api/skills` returned the `iobrx` entry and its expected description/tier.
  The temporary core instance was stopped after the check.
- Interface references were inspected in local omicos-core at
  `3a2be8e2723ad96119f777cae32577ea58c140ff` and omicos-admin at
  `09117ca4e42e852e30aebeec74ed0756426db8da`. Those checkouts had unrelated
  local work; no source changes were made to either. Discovery used the
  available local debug binary, not a newly built binary from those commits.

This establishes local loading and execution. It does not claim a production
catalog deployment, a live LLM routing evaluation, other operating-system
support, or a new scientific parity guarantee for untested versions/settings.
